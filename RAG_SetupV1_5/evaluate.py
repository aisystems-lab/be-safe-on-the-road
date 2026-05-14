"""
evaluate.py  —  V1.5
====================

Usage:
  python build_index.py          # once
  python evaluate.py
  python evaluate.py --no-llm
  python evaluate.py --k 1 3 5 10
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

from rag_chain import RagChain


EVAL_SET: list[dict[str, Any]] = [
    {"query": "Why did you warn me? I'm at risk level 3.",
     "relevant_ids": ["risk_lvl_3", "act_lvl_3"], "context": {"risk_level": 3}},
    {"query": "What does critical risk mean?",
     "relevant_ids": ["risk_lvl_4", "act_lvl_4"], "context": {"risk_level": 4}},
    {"query": "Is this safe? I see level 0 on the screen.",
     "relevant_ids": ["risk_lvl_0", "act_lvl_0"], "context": {"risk_level": 0}},
    {"query": "What should I do right now?",
     "relevant_ids": ["act_lvl_3", "risk_lvl_3"], "context": {"risk_level": 3}},
    {"query": "Why did the risk jump from low to moderate?",
     "relevant_ids": ["risk_transition_low_to_mod"], "context": {"risk_level": 2}},
    {"query": "Is texting really that bad?",
     "relevant_ids": ["beh_text_right", "beh_text_left"], "context": {"behavior": "texting"}},
    {"query": "Can I take a phone call while driving?",
     "relevant_ids": ["beh_phone_right", "beh_phone_left"], "context": {"behavior": "phone"}},
    {"query": "I'm feeling sleepy behind the wheel.",
     "relevant_ids": ["beh_drowsy"], "context": {}},
    {"query": "The driver in front keeps tailgating me.",
     "relevant_ids": ["sc_tailgater", "beh_aggressive"], "context": {}},
    {"query": "What's the best way to drive in heavy rain?",
     "relevant_ids": ["wx_rain_heavy", "wx_hydroplane", "rule_4_6_second"], "context": {}},
    {"query": "My car is skidding on the rain.",
     "relevant_ids": ["wx_hydroplane"], "context": {}},
    {"query": "How should I handle fog?",
     "relevant_ids": ["wx_fog"], "context": {}},
    {"query": "Is black ice real?",
     "relevant_ids": ["wx_ice"], "context": {}},
    {"query": "How to drive in snow safely?",
     "relevant_ids": ["wx_snow", "rule_4_6_second"], "context": {}},
    {"query": "Sun is blinding me.",
     "relevant_ids": ["wx_sun_glare"], "context": {}},
    {"query": "What is the three-second rule?",
     "relevant_ids": ["rule_3_second"], "context": {}},
    {"query": "Can I turn right on red?",
     "relevant_ids": ["rule_red_light"], "context": {}},
    {"query": "Should I speed up on a yellow light?",
     "relevant_ids": ["rule_yellow_light"], "context": {}},
    {"query": "What happens if a school bus stops with red lights?",
     "relevant_ids": ["rule_school_bus"], "context": {}},
    {"query": "An ambulance is behind me.",
     "relevant_ids": ["sc_emergency_vehicle"], "context": {}},
    {"query": "How do I merge onto the highway?",
     "relevant_ids": ["sc_highway_merge"], "context": {}},
    {"query": "There's a cyclist next to me in traffic.",
     "relevant_ids": ["sc_bicycle"], "context": {}},
    {"query": "How do I share the road with trucks?",
     "relevant_ids": ["sc_large_truck"], "context": {}},
    {"query": "A deer ran in front of me.",
     "relevant_ids": ["sc_animal_on_road"], "context": {}},
    {"query": "Driving through a school zone.",
     "relevant_ids": ["sc_school_zone", "sc_pedestrian_crossing"], "context": {}},
    {"query": "The temperature gauge is going up.",
     "relevant_ids": ["mech_overheat"], "context": {}},
    {"query": "My tire just blew out.",
     "relevant_ids": ["mech_blowout"], "context": {}},
    {"query": "The check engine light is flashing.",
     "relevant_ids": ["mech_check_engine"], "context": {}},
    {"query": "Brake pedal feels soft.",
     "relevant_ids": ["mech_brake_warning"], "context": {}},
    {"query": "How does the car measure distance to other vehicles?",
     "relevant_ids": ["faq_stereo_depth"], "context": {}},
    {"query": "How did you detect I was texting?",
     "relevant_ids": ["faq_driver_cnn"], "context": {"behavior": "texting"}},
    {"query": "Does this system work without internet?",
     "relevant_ids": ["faq_offline"], "context": {}},
    {"query": "Can I trust these warnings?",
     "relevant_ids": ["faq_false_positive", "faq_not_replacement"], "context": {}},
    {"query": "Why did you warn me at low speed?",
     "relevant_ids": ["faq_why_warn_low_speed"], "context": {"risk_level": 2, "speed_kmh": 25}},
]


def _parent_id(meta_id: str) -> str:
    return meta_id.split("#", 1)[0]


def _retrieved_ids(chunks_detailed: list[dict[str, Any]]) -> list[str]:
    out, seen = [], set()
    for c in chunks_detailed:
        pid = _parent_id(str(c.get("id", "")))
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def _prf_at_k(retrieved: list[str], relevant: set[str], k: int) -> tuple[float, float, float]:
    top = retrieved[:k]
    if not top:
        return 0.0, 0.0, 0.0
    hits = sum(1 for r in top if r in relevant)
    p = hits / min(k, len(top))
    r = hits / len(relevant) if relevant else 0.0
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f


def _rr(retrieved: list[str], relevant: set[str]) -> float:
    for i, x in enumerate(retrieved, start=1):
        if x in relevant:
            return 1.0 / i
    return 0.0


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def run_eval(use_llm: bool, k_values: tuple[int, ...]) -> dict[str, Any]:
    HERE = Path(__file__).parent.resolve()
    data_dir = HERE / "data"
    data_dir.mkdir(exist_ok=True)

    chain = RagChain(k=max(k_values))

    rows, lats, walls, rrs = [], [], [], []
    prf_by_k: dict[int, list[tuple[float, float, float]]] = {k: [] for k in k_values}
    successes = 0

    for item in EVAL_SET:
        rel = set(item["relevant_ids"])
        t0 = time.time()
        try:
            res = chain.ask(item["query"], use_llm=use_llm)
            ok = True
        except Exception as e:  # noqa: BLE001
            res = {"answer": f"ERROR: {e}", "chunks_detailed": [],
                   "latency_sec": 0.0, "used_llm": False, "llm_provider": "none"}
            ok = False
        wall = round(time.time() - t0, 4)
        if ok:
            successes += 1

        retrieved = _retrieved_ids(res.get("chunks_detailed", []))
        rr = _rr(retrieved, rel)
        rrs.append(rr)
        lats.append(float(res.get("latency_sec", wall)))
        walls.append(wall)

        row = {
            "query":           item["query"],
            "relevant_ids":    sorted(rel),
            "retrieved_ids":   retrieved,
            "answer":          (res.get("answer") or "").strip(),
            "used_llm":        res.get("used_llm", False),
            "latency_sec":     res.get("latency_sec", 0.0),
            "total_sec":       wall,
            "reciprocal_rank": round(rr, 4),
        }
        for k in k_values:
            p, r, f = _prf_at_k(retrieved, rel, k)
            prf_by_k[k].append((p, r, f))
            row[f"precision@{k}"] = round(p, 4)
            row[f"recall@{k}"]    = round(r, 4)
            row[f"f1@{k}"]        = round(f, 4)
        rows.append(row)

    summary = {
        "n_queries":              len(EVAL_SET),
        "isr":                    round(successes / len(EVAL_SET), 4),
        "mrr":                    round(statistics.mean(rrs), 4) if rrs else 0.0,
        "latency_mean_sec":       round(statistics.mean(lats), 4),
        "latency_p50_sec":        round(_pct(lats, 0.50), 4),
        "latency_p95_sec":        round(_pct(lats, 0.95), 4),
        "total_latency_mean_sec": round(statistics.mean(walls), 4),
        "vector_store":           "faiss",
        "embedding_model":        "sentence-transformers/all-MiniLM-L6-v2",
        "reranker":               None,
        "used_llm":               use_llm,
        "llm_provider":           ("ollama" if use_llm else "none"),
        "backend":                "V1.5",
    }
    for k in k_values:
        v = prf_by_k[k]
        summary[f"precision@{k}"] = round(statistics.mean(x[0] for x in v), 4)
        summary[f"recall@{k}"]    = round(statistics.mean(x[1] for x in v), 4)
        summary[f"f1@{k}"]        = round(statistics.mean(x[2] for x in v), 4)

    with (data_dir / "eval_results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (data_dir / "eval_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    args = ap.parse_args()
    s = run_eval(use_llm=not args.no_llm, k_values=tuple(args.k))
    print("\n=== V1.5 evaluation summary ===")
    for k in ("backend", "n_queries", "mrr", "isr",
              "latency_mean_sec", "latency_p50_sec", "latency_p95_sec",
              "used_llm", "llm_provider"):
        print(f"  {k:24s}: {s.get(k)}")
    for k in sorted(x for x in s if x.startswith(("precision@", "recall@", "f1@"))):
        print(f"  {k:24s}: {s[k]:.4f}")
    print("\n  -> data/eval_results.jsonl")
    print("  -> data/eval_summary.csv")


if __name__ == "__main__":
    main()
