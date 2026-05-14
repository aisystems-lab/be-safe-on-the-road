"""
evaluate.py
-----------

Metrics
-------
- Precision@k, Recall@k, F1@k for k in settings.eval_k_values
- MRR (Mean Reciprocal Rank)
- Retrieval latency (mean, p50, p95)
- LLM latency (mean, p50, p95)
- Answer length (sanity check)

Outputs
-------
- Per-query results -> data/eval_results.jsonl
- Summary table     -> data/eval_summary.csv  (paper-ready)
- Console summary

Usage
-----
    python evaluate.py
    python evaluate.py --no-llm          # retrieval-only
    python evaluate.py --k 1 3 5 10      # custom cutoffs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

from config import settings
from rag_chain import RagChain


logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")


EVAL_SET: list[dict[str, Any]] = [
    {
        "query": "Why did you warn me? I'm at risk level 3.",
        "relevant_ids": ["risk_lvl_3", "act_lvl_3"],
        "context": {"risk_level": 3},
    },
    {
        "query": "What does critical risk mean?",
        "relevant_ids": ["risk_lvl_4", "act_lvl_4"],
        "context": {"risk_level": 4},
    },
    {
        "query": "Is this safe? I see level 0 on the screen.",
        "relevant_ids": ["risk_lvl_0", "act_lvl_0"],
        "context": {"risk_level": 0},
    },
    {
        "query": "What should I do right now?",
        "relevant_ids": ["act_lvl_3", "risk_lvl_3"],
        "context": {"risk_level": 3},
    },
    {
        "query": "Why did the risk jump from low to moderate?",
        "relevant_ids": ["risk_transition_low_to_mod"],
        "context": {"risk_level": 2},
    },
    {
        "query": "Is texting really that bad?",
        "relevant_ids": ["beh_text_right", "beh_text_left"],
        "context": {"behavior": "texting"},
    },
    {
        "query": "Can I take a phone call while driving?",
        "relevant_ids": ["beh_phone_right", "beh_phone_left"],
        "context": {"behavior": "phone"},
    },
    {
        "query": "I'm feeling sleepy behind the wheel.",
        "relevant_ids": ["beh_drowsy"],
        "context": {},
    },
    {
        "query": "The driver in front keeps tailgating me.",
        "relevant_ids": ["sc_tailgater", "beh_aggressive"],
        "context": {},
    },
    {
        "query": "What's the best way to drive in heavy rain?",
        "relevant_ids": ["wx_rain_heavy", "wx_hydroplane", "rule_4_6_second"],
        "context": {},
    },
    {
        "query": "My car is skidding on the rain.",
        "relevant_ids": ["wx_hydroplane"],
        "context": {},
    },
    {
        "query": "How should I handle fog?",
        "relevant_ids": ["wx_fog"],
        "context": {},
    },
    {
        "query": "Is black ice real?",
        "relevant_ids": ["wx_ice"],
        "context": {},
    },
    {
        "query": "How to drive in snow safely?",
        "relevant_ids": ["wx_snow", "rule_4_6_second"],
        "context": {},
    },
    {
        "query": "Sun is blinding me.",
        "relevant_ids": ["wx_sun_glare"],
        "context": {},
    },
    {
        "query": "What is the three-second rule?",
        "relevant_ids": ["rule_3_second"],
        "context": {},
    },
    {
        "query": "Can I turn right on red?",
        "relevant_ids": ["rule_red_light"],
        "context": {},
    },
    {
        "query": "Should I speed up on a yellow light?",
        "relevant_ids": ["rule_yellow_light"],
        "context": {},
    },
    {
        "query": "What happens if a school bus stops with red lights?",
        "relevant_ids": ["rule_school_bus"],
        "context": {},
    },
    {
        "query": "An ambulance is behind me.",
        "relevant_ids": ["sc_emergency_vehicle"],
        "context": {},
    },
    {
        "query": "How do I merge onto the highway?",
        "relevant_ids": ["sc_highway_merge"],
        "context": {},
    },
    {
        "query": "There's a cyclist next to me in traffic.",
        "relevant_ids": ["sc_bicycle"],
        "context": {},
    },
    {
        "query": "How do I share the road with trucks?",
        "relevant_ids": ["sc_large_truck"],
        "context": {},
    },
    {
        "query": "A deer ran in front of me.",
        "relevant_ids": ["sc_animal_on_road"],
        "context": {},
    },
    {
        "query": "Driving through a school zone.",
        "relevant_ids": ["sc_school_zone", "sc_pedestrian_crossing"],
        "context": {},
    },
    {
        "query": "The temperature gauge is going up.",
        "relevant_ids": ["mech_overheat"],
        "context": {},
    },
    {
        "query": "My tire just blew out.",
        "relevant_ids": ["mech_blowout"],
        "context": {},
    },
    {
        "query": "The check engine light is flashing.",
        "relevant_ids": ["mech_check_engine"],
        "context": {},
    },
    {
        "query": "Brake pedal feels soft.",
        "relevant_ids": ["mech_brake_warning"],
        "context": {},
    },
    {
        "query": "How does the car measure distance to other vehicles?",
        "relevant_ids": ["faq_stereo_depth"],
        "context": {},
    },
    {
        "query": "How did you detect I was texting?",
        "relevant_ids": ["faq_driver_cnn"],
        "context": {"behavior": "texting"},
    },
    {
        "query": "Does this system work without internet?",
        "relevant_ids": ["faq_offline"],
        "context": {},
    },
    {
        "query": "Can I trust these warnings?",
        "relevant_ids": ["faq_false_positive", "faq_not_replacement"],
        "context": {},
    },
    {
        "query": "Why did you warn me at low speed?",
        "relevant_ids": ["faq_why_warn_low_speed"],
        "context": {"risk_level": 2, "speed_kmh": 25},
    },
]


def _parent_id(meta_id: str) -> str:
    """Chunks carry IDs like 'foo#c0'; the parent is 'foo'."""
    return meta_id.split("#", 1)[0]


def _retrieved_ids(chunks_detailed: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for c in chunks_detailed:
        pid = _parent_id(c.get("id", ""))
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def _prf_at_k(retrieved: list[str], relevant: set[str], k: int) -> tuple[float, float, float]:
    top = retrieved[:k]
    if not top:
        return 0.0, 0.0, 0.0
    hits = sum(1 for r in top if r in relevant)
    precision = hits / min(k, len(top))
    recall = hits / len(relevant) if relevant else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def _reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def run_eval(use_llm: bool, k_values: tuple[int, ...]) -> dict[str, Any]:
    if not use_llm:
        settings.use_structured_output = False
        settings.llm_provider = "none"
        settings.llm_fallback = "none"

    chain = RagChain()

    per_query_rows: list[dict[str, Any]] = []
    retrieval_latencies: list[float] = []
    total_latencies: list[float] = []
    mrrs: list[float] = []

    prf_by_k: dict[int, list[tuple[float, float, float]]] = {k: [] for k in k_values}

    for item in EVAL_SET:
        q = item["query"]
        rel = set(item["relevant_ids"])
        ctx = item.get("context", {})

        t0 = time.time()
        res = chain.ask(q, context=ctx)
        total = time.time() - t0

        retrieved = _retrieved_ids(res.get("chunks_detailed", []))
        rr = _reciprocal_rank(retrieved, rel)
        mrrs.append(rr)

        row = {
            "query": q,
            "relevant_ids": sorted(rel),
            "retrieved_ids": retrieved,
            "answer": res["answer"],
            "used_llm": res["used_llm"],
            "latency_sec": res["latency_sec"],
            "total_sec": round(total, 4),
            "reciprocal_rank": round(rr, 4),
        }
        for k in k_values:
            p, r, f = _prf_at_k(retrieved, rel, k)
            prf_by_k[k].append((p, r, f))
            row[f"precision@{k}"] = round(p, 4)
            row[f"recall@{k}"] = round(r, 4)
            row[f"f1@{k}"] = round(f, 4)

        per_query_rows.append(row)
        retrieval_latencies.append(res["latency_sec"])
        total_latencies.append(total)

    def _pct(xs: list[float], p: float) -> float:
        if not xs:
            return 0.0
        xs_sorted = sorted(xs)
        idx = min(len(xs_sorted) - 1, int(round(p * (len(xs_sorted) - 1))))
        return xs_sorted[idx]

    summary: dict[str, Any] = {
        "n_queries": len(EVAL_SET),
        "mrr": round(statistics.mean(mrrs), 4) if mrrs else 0.0,
        "latency_mean_sec": round(statistics.mean(retrieval_latencies), 4),
        "latency_p50_sec": round(_pct(retrieval_latencies, 0.5), 4),
        "latency_p95_sec": round(_pct(retrieval_latencies, 0.95), 4),
        "total_latency_mean_sec": round(statistics.mean(total_latencies), 4),
        "vector_store": settings.vector_store,
        "embedding_model": settings.embedding_model,
        "reranker": settings.reranker_model if settings.use_reranker else None,
        "used_llm": use_llm,
        "llm_provider": chain.llm_router.active_provider if use_llm else "none",
    }
    for k in k_values:
        vals = prf_by_k[k]
        summary[f"precision@{k}"] = round(statistics.mean(v[0] for v in vals), 4)
        summary[f"recall@{k}"] = round(statistics.mean(v[1] for v in vals), 4)
        summary[f"f1@{k}"] = round(statistics.mean(v[2] for v in vals), 4)

    out_jsonl = settings.data_dir / "eval_results.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in per_query_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_csv = settings.data_dir / "eval_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    return summary


def _print_summary(s: dict[str, Any]) -> None:
    print("\n=== Evaluation summary ===")
    print(f"  queries          : {s['n_queries']}")
    print(f"  vector_store     : {s['vector_store']}")
    print(f"  embeddings       : {s['embedding_model']}")
    print(f"  reranker         : {s['reranker']}")
    print(f"  used_llm         : {s['used_llm']} ({s['llm_provider']})")
    print()
    print(f"  MRR              : {s['mrr']:.4f}")
    for key in sorted(k for k in s if k.startswith(("precision@", "recall@", "f1@"))):
        print(f"  {key:16s} : {s[key]:.4f}")
    print()
    print(f"  retrieval latency: mean={s['latency_mean_sec']}s "
          f"p50={s['latency_p50_sec']}s p95={s['latency_p95_sec']}s")
    print(f"  total latency    : mean={s['total_latency_mean_sec']}s")
    print(f"\n  Per-query results: {settings.data_dir / 'eval_results.jsonl'}")
    print(f"  Summary CSV      : {settings.data_dir / 'eval_summary.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="Retrieval-only evaluation")
    ap.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=list(settings.eval_k_values),
        help="Precision/Recall/F1 cutoffs",
    )
    args = ap.parse_args()

    summary = run_eval(use_llm=not args.no_llm, k_values=tuple(args.k))
    _print_summary(summary)


if __name__ == "__main__":
    main()
