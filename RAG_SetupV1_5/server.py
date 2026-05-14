from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS

from rag_chain import RagChain


HERE = Path(__file__).parent.resolve()
PORT = int(os.environ.get("PORT", "8002"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("v15.server")

app = Flask(__name__)
CORS(app)

chain = RagChain(k=3)

EVENT_LOG_PATH   = HERE / "event_log.json"
RESULTS_CSV_PATH = HERE / "results.csv"
event_log: list[dict[str, Any]] = []
if EVENT_LOG_PATH.exists():
    try:
        event_log = json.loads(EVENT_LOG_PATH.read_text())
    except Exception:
        event_log = []


def _save_events() -> None:
    try:
        EVENT_LOG_PATH.write_text(
            json.dumps(event_log[-500:], indent=2, ensure_ascii=False)
        )
    except Exception as e:
        logger.warning("save events failed: %s", e)


def _empty_response(t_q: float) -> dict[str, Any]:
    ts = int(t_q * 1000)
    return {
        "answer": "", "chunks": [], "chunks_detailed": [],
        "action": "", "urgency": "low", "confidence": 0.0,
        "latency_sec": 0.0, "used_llm": False, "llm_provider": "none",
        "success": False, "timestamp_query": ts, "timestamp_resp": ts,
    }


@app.get("/health")
def health():
    return jsonify({"ok": True, "version": "v1.5"})


@app.post("/ask")
def ask():
    data = request.get_json(force=True, silent=True) or {}
    query   = (data.get("query") or "").strip()
    use_llm = bool(data.get("use_llm", True))
    _ctx = data.get("context") or {}

    t_q = time.time()
    if not query:
        return jsonify(_empty_response(t_q))

    try:
        out = chain.ask(query, use_llm=use_llm)
        out["success"] = True
    except Exception as e:
        logger.exception("chain failed")
        out = {
            "answer": f"Temporary issue: {e}",
            "chunks": [], "chunks_detailed": [],
            "action": "stay alert", "urgency": "medium", "confidence": 0.0,
            "latency_sec": 0.0, "used_llm": False, "llm_provider": "none",
            "success": False,
        }
    t_r = time.time()
    out["timestamp_query"] = int(t_q * 1000)
    out["timestamp_resp"]  = int(t_r * 1000)

    event_log.append({"type": "ask", "query": query, **out})
    _save_events()
    return jsonify(out)


@app.get("/events")
def events():
    return jsonify([e for e in event_log[-50:] if isinstance(e, dict)])


@app.get("/export_csv")
def export_csv():
    fields = ["timestamp_query", "timestamp_resp", "latency_sec", "success",
              "query", "answer", "used_llm", "llm_provider"]
    rows = [{k: e.get(k, "") for k in fields}
            for e in event_log if isinstance(e, dict) and e.get("type") == "ask"]
    with RESULTS_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return jsonify({"ok": True, "rows": len(rows), "path": str(RESULTS_CSV_PATH)})


if __name__ == "__main__":
    logger.info("V1.5 server listening on :%d", PORT)
    app.run(host="0.0.0.0", port=PORT)
