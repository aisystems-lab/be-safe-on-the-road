from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from config import settings
from knowledge_base import build_knowledge_base, kb_stats


logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")

MAX_LSTM_CARDS = 50


def _load_curated() -> list[dict[str, Any]]:
    records = build_knowledge_base()
    for r in records:
        r["source"] = "curated"
    return records


def _load_lstm_cards(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path or not csv_path.exists():
        logger.info("No LSTM CSV at %s; skipping.", csv_path)
        return []

    cards: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return []

    sample_keys = {k.lower() for k in rows[0].keys()}
    speed_col = next(
        (k for k in rows[0] if k.lower() in {"speed", "speed_kmh", "speed_mph"}),
        None,
    )
    dist_col = next(
        (k for k in rows[0] if k.lower() in {"distance", "depth", "depth_m", "headway"}),
        None,
    )
    risk_col = next(
        (k for k in rows[0] if "risk" in k.lower()),
        None,
    )
    beh_col = next(
        (k for k in rows[0] if "behav" in k.lower() or "driver" in k.lower()),
        None,
    )

    if not (speed_col and dist_col and risk_col):
        logger.warning(
            "LSTM CSV missing expected columns (speed/distance/risk); "
            "found %s. Skipping cards.",
            list(rows[0].keys()),
        )
        return []

    step = max(1, len(rows) // MAX_LSTM_CARDS)
    sampled = rows[::step][:MAX_LSTM_CARDS]

    for i, row in enumerate(sampled):
        try:
            risk = int(float(row[risk_col]))
        except (ValueError, TypeError):
            continue
        speed = row.get(speed_col, "NA")
        dist = row.get(dist_col, "NA")
        beh = row.get(beh_col, "NA") if beh_col else "NA"
        text = (
            f"Observed driving context: speed={speed}, "
            f"distance={dist}, behavior={beh}. "
            f"The system assigned risk level {risk}."
        )
        cards.append(
            {
                "id": f"lstm_{i:04d}",
                "text": text,
                "category": "observed_context",
                "risk_level": risk,
                "tags": ["observed", "lstm", f"level-{risk}"],
                "source": "lstm_csv",
            }
        )

    logger.info("Loaded %d LSTM context cards from %s", len(cards), csv_path)
    return cards


def build() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records.extend(_load_curated())
    records.extend(_load_lstm_cards(settings.lstm_csv_path))
    return records


def main() -> None:
    settings.ensure_dirs()
    records = build()

    settings.kb_json_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.kb_json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    logger.info("Wrote %d KB entries to %s", len(records), settings.kb_json_path)
    print(f"✅ Knowledge base written: {settings.kb_json_path} ({len(records)} docs)")
    print("   Curated breakdown:")
    for category, count in kb_stats().items():
        print(f"   - {category:20s} {count:4d}")


if __name__ == "__main__":
    main()
