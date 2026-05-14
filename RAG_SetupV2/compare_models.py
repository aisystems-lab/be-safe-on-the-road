from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any, Optional

from config import settings
from evaluate import EVAL_SET, _retrieved_ids, _reciprocal_rank, _prf_at_k
from judge import Judge, JudgeVerdict, aggregate, context_str
from llm import (
    available_providers,
    build_llm,
    build_openrouter_llm,
    list_configured_openrouter_models,
    openrouter_display_name,
    provider_display_name,
)
from rag_chain import RagChain


logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")


def _make_chain_for_provider(provider: str) -> RagChain:
    base = getattr(_make_chain_for_provider, "_base", None)
    if base is None:
        raise RuntimeError(
            "Internal: call site must set _make_chain_for_provider._base "
            "to a fully-built RagChain before invoking this helper."
        )
    chain = RagChain.__new__(RagChain) 
    chain.vs = base.vs
    chain.retriever = base.retriever
    chain.llm_router = build_llm(provider, fallback_name="none")
    if settings.use_structured_output and chain.llm_router.has_llm:
        from schemas import SafetyAnswer
        chain.structured_llm = chain.llm_router.with_structured_output(SafetyAnswer)
    else:
        chain.structured_llm = None
    return chain


def _make_chain_for_openrouter_model(model_id: str) -> RagChain:
    """Same as _make_chain_for_provider but for one specific OpenRouter model ID."""
    base = getattr(_make_chain_for_provider, "_base", None)
    if base is None:
        raise RuntimeError(
            "Internal: call site must set _make_chain_for_provider._base "
            "first."
        )
    chain = RagChain.__new__(RagChain)
    chain.vs = base.vs
    chain.retriever = base.retriever
    chain.llm_router = build_openrouter_llm(model_id)
    if settings.use_structured_output and chain.llm_router.has_llm:
        from schemas import SafetyAnswer
        chain.structured_llm = chain.llm_router.with_structured_output(SafetyAnswer)
    else:
        chain.structured_llm = None
    return chain


def _run_eval_for_chain(
    chain: RagChain,
    identifier: str,        
    display: str,          
    judge: Optional[Judge],
    trials: int,
    cooldown: float,
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    if not chain.llm_router.has_llm:
        logger.warning("LLM unavailable for %s — skipping.", identifier)
        return {"provider": identifier, "display": display, "skipped": True, "rows": []}

    rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    verdicts: list[JudgeVerdict] = []

    prf_by_k: dict[int, list[tuple[float, float, float]]] = {k: [] for k in k_values}
    mrrs: list[float] = []

    FAST_FAIL_PROBE = 3
    fast_fail_misses = 0

    for query_idx, item in enumerate(EVAL_SET):
        q = item["query"]
        rel = set(item["relevant_ids"])
        ctx = item.get("context", {})

        per_trial_latency: list[float] = []
        last_result: Optional[dict[str, Any]] = None
        for t in range(trials):
            t0 = time.time()
            try:
                res = chain.ask(q, context=ctx)
                latency_ms = (time.time() - t0) * 1000.0
            except Exception as e:  
                logger.error("[%s] %s -> ERROR: %s", identifier, q, e)
                res = None
                latency_ms = float("nan")
            per_trial_latency.append(latency_ms)
            last_result = res or last_result
            if cooldown > 0 and t < trials - 1:
                time.sleep(cooldown)

        if query_idx < FAST_FAIL_PROBE:
            llm_worked = bool(last_result and last_result.get("used_llm", False))
            if not llm_worked:
                fast_fail_misses += 1
            if query_idx == FAST_FAIL_PROBE - 1 and fast_fail_misses >= FAST_FAIL_PROBE:
                logger.error(
                    "[%s] LLM failed on all %d probe queries — upstream provider "
                    "is unhealthy. Skipping the remaining %d queries to save quota. "
                    "Re-run this model later with: "
                    "./run_compare_openrouter.sh --openrouter-models %s",
                    identifier, FAST_FAIL_PROBE,
                    len(EVAL_SET) - FAST_FAIL_PROBE,
                    identifier.replace("openrouter:", ""),
                )
        
                rows.append({
                    "provider": identifier,
                    "display": display,
                    "query": "<fast-fail>",
                    "answer": "",
                    "action": "",
                    "urgency": "",
                    "latency_ms": float("nan"),
                    "error": f"fast-failed after {FAST_FAIL_PROBE} LLM-failed probes",
                })
                return {
                    "provider": identifier,
                    "display": display,
                    "rows": rows,
                    "summary": {
                        "provider": identifier,
                        "display": display,
                        "n_queries": len(EVAL_SET),
                        "n_judged": 0,
                        "fast_failed": True,
                        "correctness": float("nan"),
                        "groundedness": float("nan"),
                        "instruction_following": float("nan"),
                        "latency_mean_ms": float("nan"),
                        "latency_p50_ms": float("nan"),
                        "latency_p95_ms": float("nan"),
                        "mrr": 0.0,
                        **{f"precision@{k}": 0.0 for k in k_values},
                        **{f"recall@{k}": 0.0 for k in k_values},
                        **{f"f1@{k}": 0.0 for k in k_values},
                    },
                }

        if last_result is None:
            rows.append({
                "provider": identifier,
                "display": display,
                "query": q,
                "answer": "",
                "action": "",
                "urgency": "",
                "latency_ms": float("nan"),
                "error": "all-trials-failed",
            })
            continue

        finite_latencies = [x for x in per_trial_latency if x == x]
        mean_latency_ms = (
            statistics.mean(finite_latencies) if finite_latencies else float("nan")
        )
        latencies_ms.append(mean_latency_ms)

        retrieved = _retrieved_ids(last_result.get("chunks_detailed", []))
        rr = _reciprocal_rank(retrieved, rel)
        mrrs.append(rr)
        for k in k_values:
            prf_by_k[k].append(_prf_at_k(retrieved, rel, k))

        verdict_dict: dict[str, Any] = {}
        if judge is not None:
            try:
                v = judge.score(
                    question=q,
                    context=context_str(last_result.get("chunks_detailed", [])),
                    answer=last_result.get("answer", ""),
                    action=last_result.get("action", ""),
                    urgency=last_result.get("urgency", ""),
                )
                if v is None:
                    logger.warning(
                        "[%s] judge returned None on %r (likely empty answer); "
                        "row will lack a verdict.",
                        identifier, q,
                    )
                    verdict_dict = {"judge_error": "judge_returned_none"}
                else:
                    verdicts.append(v)
                    verdict_dict = v.model_dump()
            except Exception as e:  
                logger.error("[%s] judge failed on %r: %s", identifier, q, e)
                verdict_dict = {"judge_error": str(e)}

        rows.append({
            "provider": identifier,
            "display": display,
            "query": q,
            "answer": last_result.get("answer", ""),
            "action": last_result.get("action", ""),
            "urgency": last_result.get("urgency", ""),
            "confidence": last_result.get("confidence", 0.0),
            "latency_ms": round(mean_latency_ms, 2),
            "trials": trials,
            "retrieved_ids": retrieved,
            "relevant_ids": sorted(rel),
            "reciprocal_rank": round(rr, 4),
            "verdict": verdict_dict,
        })

        if cooldown > 0:
            time.sleep(cooldown)

    summary = aggregate(verdicts) if verdicts else {
        "correctness": float("nan"),
        "groundedness": float("nan"),
        "instruction_following": float("nan"),
    }
    summary["provider"] = identifier
    summary["display"] = display
    summary["n_queries"] = len(EVAL_SET)
    summary["n_judged"] = len(verdicts)
    summary["latency_mean_ms"] = (
        round(statistics.mean(latencies_ms), 2) if latencies_ms else float("nan")
    )
    summary["latency_p50_ms"] = (
        round(statistics.median(latencies_ms), 2) if latencies_ms else float("nan")
    )
    summary["latency_p95_ms"] = (
        round(_pct(latencies_ms, 0.95), 2) if latencies_ms else float("nan")
    )
    summary["mrr"] = round(statistics.mean(mrrs), 4) if mrrs else 0.0
    for k in k_values:
        vals = prf_by_k[k]
        summary[f"precision@{k}"] = round(statistics.mean(v[0] for v in vals), 4) if vals else 0.0
        summary[f"recall@{k}"] = round(statistics.mean(v[1] for v in vals), 4) if vals else 0.0
        summary[f"f1@{k}"] = round(statistics.mean(v[2] for v in vals), 4) if vals else 0.0

    return {"provider": identifier, "display": display, "rows": rows, "summary": summary}


def run_for_provider(
    provider: str,
    judge: Optional[Judge],
    trials: int,
    cooldown: float,
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    """Legacy multi-provider path: one row per provider name."""
    display = provider_display_name(provider)
    logger.info("=== %s ===", display)
    chain = _make_chain_for_provider(provider)
    return _run_eval_for_chain(chain, provider, display, judge, trials, cooldown, k_values)


def run_for_openrouter_model(
    model_id: str,
    judge: Optional[Judge],
    trials: int,
    cooldown: float,
    k_values: tuple[int, ...],
) -> dict[str, Any]:
    """OpenRouter path: one row per model ID, all on the same gateway."""
    display = openrouter_display_name(model_id)
    identifier = f"openrouter:{model_id}"
    logger.info("=== %s ===", display)
    chain = _make_chain_for_openrouter_model(model_id)
    return _run_eval_for_chain(chain, identifier, display, judge, trials, cooldown, k_values)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    idx = min(len(xs_sorted) - 1, int(round(p * (len(xs_sorted) - 1))))
    return xs_sorted[idx]


PAPER_COLUMNS = [
    "Model",
    "Correctness",
    "Groundedness",
    "Instruction following",
    "Latency (ms)",
]


def _load_existing_rows(out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    existing_rows: list[dict[str, Any]] = []
    jsonl_path = out_dir / "model_comparison.jsonl"
    if jsonl_path.exists():
        try:
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        existing_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except Exception as e:  
            logger.warning("Could not read existing JSONL for append: %s", e)

    existing_summaries: list[dict[str, Any]] = []
    full_path = out_dir / "model_comparison_full.csv"
    if full_path.exists():
        try:
            with full_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    out: dict[str, Any] = {}
                    for k, v in row.items():
                        if v in ("", None):
                            out[k] = float("nan")
                            continue
                        try:
                            out[k] = float(v)
                        except (TypeError, ValueError):
                            out[k] = v
                    existing_summaries.append(out)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not read existing summary CSV for append: %s", e)

    return existing_rows, existing_summaries


def write_outputs(
    per_provider: list[dict[str, Any]],
    out_dir: Path,
    append: bool = False,
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    new_keys: set[str] = {
        p.get("display", "") for p in per_provider if p.get("display")
    }

    combined_rows: list[dict[str, Any]] = []  
    combined_summaries: list[dict[str, Any]] = []  

    if append:
        old_rows, old_summaries = _load_existing_rows(out_dir)
        for r in old_rows:
            if r.get("display", "") not in new_keys:
                combined_rows.append(r)
        for s in old_summaries:
            if s.get("display", "") not in new_keys:
                combined_summaries.append(s)
        if old_rows or old_summaries:
            logger.info(
                "Append mode: kept %d old per-query rows and %d old summaries from previous runs.",
                len(combined_rows), len(combined_summaries),
            )

    for prov in per_provider:
        for row in prov.get("rows", []):
            combined_rows.append(row)
        if "summary" in prov:
            combined_summaries.append(prov["summary"])

    jsonl_path = out_dir / "model_comparison.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in combined_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _sort_key(s: dict[str, Any]) -> tuple:
        def safe(v, default):
            try:
                f = float(v)
                if f != f:  
                    return default
                return f
            except (TypeError, ValueError):
                return default
        return (
            -safe(s.get("correctness"), -1e9),
            -safe(s.get("groundedness"), -1e9),
            -safe(s.get("instruction_following"), -1e9),
            safe(s.get("latency_mean_ms"), 1e18),
        )

    ranked = sorted(combined_summaries, key=_sort_key)

    csv_path = out_dir / "model_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(PAPER_COLUMNS)
        for s in ranked:
            def fmt(key, places=4):
                v = s.get(key)
                try:
                    fv = float(v)
                    if fv != fv:  
                        return "—"
                    return f"{fv:.{places}f}"
                except (TypeError, ValueError):
                    return "—"
            w.writerow([
                s.get("display", ""),
                fmt("correctness"),
                fmt("groundedness"),
                fmt("instruction_following"),
                fmt("latency_mean_ms", 2),
            ])

    ext_path = out_dir / "model_comparison_full.csv"
    if combined_summaries:
        all_keys: list[str] = []
        seen: set[str] = set()
        for s in combined_summaries:
            for k in s.keys():
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        with ext_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            for s in ranked:
                w.writerow(s)

    md_path = out_dir / "model_comparison.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(PAPER_COLUMNS) + " |\n")
        f.write("|" + "|".join(["---"] * len(PAPER_COLUMNS)) + "|\n")
        for s in ranked:
            def fmt(key, places=4):
                v = s.get(key)
                try:
                    fv = float(v)
                    if fv != fv:
                        return "—"
                    return f"{fv:.{places}f}"
                except (TypeError, ValueError):
                    return "—"
            f.write(
                f"| {s.get('display', '')} | {fmt('correctness')} | "
                f"{fmt('groundedness')} | {fmt('instruction_following')} | "
                f"{fmt('latency_mean_ms', 2)} |\n"
            )

    mode = "APPENDED" if append else "WROTE"
    print(f"\n{mode}:\n  {csv_path}\n  {md_path}\n  {ext_path}\n  {jsonl_path}")
    print(f"  ({len(ranked)} models in table)")


def _print_console_table(per_provider: list[dict[str, Any]]) -> None:
    print("\n=== Model comparison (Table 7 shape) ===")
    fmt = "{:<32} {:>11} {:>13} {:>22} {:>13}"
    print(fmt.format(*PAPER_COLUMNS))
    print("-" * 100)
    ranked = sorted(
        (p for p in per_provider if "summary" in p),
        key=lambda p: (
            -p["summary"]["correctness"],
            -p["summary"]["groundedness"],
            -p["summary"]["instruction_following"],
            p["summary"]["latency_mean_ms"],
        ),
    )
    for p in ranked:
        s = p["summary"]
        print(
            fmt.format(
                s["display"][:32],
                f"{s['correctness']:.4f}",
                f"{s['groundedness']:.4f}",
                f"{s['instruction_following']:.4f}",
                f"{s['latency_mean_ms']:.2f}",
            )
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--providers",
        nargs="+",
        default=None,
        help="Providers to compare (default: auto-detect from API keys). "
             "Ignored if --openrouter is used.",
    )
    ap.add_argument(
        "--openrouter",
        action="store_true",
        help="OpenRouter mode: iterate over OPENROUTER_MODELS instead of "
             "PROVIDERS. One key, one gateway, many models. SAFEST OPTION.",
    )
    ap.add_argument(
        "--openrouter-models",
        nargs="+",
        default=None,
        help="Override the OPENROUTER_MODELS list (only used with --openrouter). "
             "Pass full model IDs, e.g. 'meta-llama/llama-3.3-70b-instruct:free'.",
    )
    ap.add_argument(
        "--allow-paid",
        action="store_true",
        help="Disable the :free safety guard. WARNING: this can spend money "
             "on the OpenRouter key. Only use if your professor has approved "
             "a budget AND set a credit cap on the dashboard.",
    )
    ap.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM-as-judge step. Only retrieval + latency are reported.",
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append results to existing output files instead of overwriting. "
             "Lets you run one model at a time over multiple sessions and "
             "build up the comparison table incrementally. Re-running the "
             "same model overwrites just that model's row.",
    )
    ap.add_argument(
        "--trials",
        type=int,
        default=settings.compare_trials,
        help="Trials per (model, query) — latencies are averaged.",
    )
    ap.add_argument(
        "--cooldown",
        type=float,
        default=None,
        help="Sleep between calls (seconds). Default: 4.0 in --openrouter mode "
             "(stays under 20 req/min/model), else settings.compare_cooldown_sec.",
    )
    ap.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=list(settings.eval_k_values),
        help="Precision/Recall/F1 cutoffs.",
    )
    args = ap.parse_args()

    if args.allow_paid:
        settings.openrouter_free_only = False
        logger.warning(
            "--allow-paid passed: :free guard DISABLED. Paid OpenRouter "
            "models can now be invoked. Make sure the dashboard credit cap "
            "is set sensibly."
        )

    logger.info("Loading retriever / vector store (one-time)...")
    _make_chain_for_provider._base = RagChain()  

    judge: Optional[Judge] = None
    if not args.no_judge:
        try:
            judge = Judge()
            logger.info("Judge ready: provider=%s", settings.judge_provider)
        except Exception as e:  
            logger.error(
                "Judge could not be constructed (%s). Re-run with --no-judge "
                "or fix the judge provider configuration.",
                e,
            )
            raise

    per_provider: list[dict[str, Any]] = []

    # ---------------- OpenRouter mode ----------------
    if args.openrouter:
        if not settings.openrouter_api_key:
            raise SystemExit(
                "OPENROUTER_API_KEY is not set. Add it to .env before "
                "running with --openrouter."
            )
        models = args.openrouter_models or list_configured_openrouter_models()
        if not models:
            raise SystemExit(
                "No OpenRouter models configured. Set OPENROUTER_MODELS in .env "
                "or pass --openrouter-models."
            )
        from llm import _check_openrouter_model_id
        for m in models:
            _check_openrouter_model_id(m)

        cooldown = (
            args.cooldown
            if args.cooldown is not None
            else settings.openrouter_per_model_cooldown_sec
        )
        logger.info(
            "OpenRouter mode: %d models, cooldown=%.1fs/call (rate-limit safe).",
            len(models),
            cooldown,
        )
        for m in models:
            try:
                res = run_for_openrouter_model(
                    m, judge, args.trials, cooldown, tuple(args.k)
                )
            except Exception as e:  
                logger.error("OpenRouter model %s crashed: %s", m, e)
                res = {
                    "provider": f"openrouter:{m}",
                    "display": openrouter_display_name(m),
                    "error": str(e),
                    "rows": [],
                }
            per_provider.append(res)

        write_outputs(per_provider, settings.data_dir, append=args.append)
        _print_console_table(per_provider)
        return

    if args.providers:
        providers = list(args.providers)
    elif settings.compare_providers.strip():
        providers = [p.strip() for p in settings.compare_providers.split(",") if p.strip()]
    else:
        providers = available_providers()

    if not providers:
        raise SystemExit(
            "No providers available. Configure at least one API key, "
            "or pass --providers ollama, or use --openrouter (recommended)."
        )

    cooldown = (
        args.cooldown if args.cooldown is not None else settings.compare_cooldown_sec
    )
    logger.info("Comparing providers: %s", ", ".join(providers))
    for prov in providers:
        try:
            res = run_for_provider(prov, judge, args.trials, cooldown, tuple(args.k))
        except Exception as e:  
            logger.error("Provider %s crashed: %s", prov, e)
            res = {
                "provider": prov,
                "display": provider_display_name(prov),
                "error": str(e),
                "rows": [],
            }
        per_provider.append(res)

    write_outputs(per_provider, settings.data_dir, append=args.append)
    _print_console_table(per_provider)


if __name__ == "__main__":
    main()
