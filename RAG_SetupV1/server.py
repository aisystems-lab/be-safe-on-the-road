"""
server.py  —  V1 (RAG without LangChain)
========================================

Listens on :8001 by default.

  POST /ask          -> {answer, chunks, chunks_detailed, ...}
  GET  /health
  GET  /events
  GET  /export_csv
"""

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

from retriever import Retriever
from generator import OllamaGenerator


HERE = Path(__file__).parent.resolve()
PORT = int(os.environ.get("PORT", "8001"))
USE_LLM_DEFAULT = os.environ.get("USE_LLM", "true").lower() == "true"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("v1.server")

app = Flask(__name__)
CORS(app)

retriever = Retriever(kb_path=HERE / "kb.json")
generator = OllamaGenerator(
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
)

EVENT_LOG_PATH = HERE / "event_log.json"
RESULTS_CSV_PATH = HERE / "results.csv"
event_log: list[dict[str, Any]] = []
if EVENT_LOG_PATH.exists():
    try:
        event_log = json.loads(EVENT_LOG_PATH.read_text())
    except Exception:  # noqa: BLE001
        event_log = []


def _save_events() -> None:
    try:
        EVENT_LOG_PATH.write_text(
            json.dumps(event_log[-500:], indent=2, ensure_ascii=False)
        )
    except Exception as e:  # noqa: BLE001
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
    return jsonify({
        "ok": True, "version": "v1",
        "embedding": retriever.model_name,
        "kb_size": len(retriever.docs),
        "llm": generator.model,
    })


@app.post("/ask")
def ask():
    data = request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()
    use_llm = bool(data.get("use_llm", USE_LLM_DEFAULT))
    _ctx = data.get("context") or {}      # accepted but unused in V1

    t_q = time.time()
    if not query:
        return jsonify(_empty_response(t_q))

    hits = retriever.retrieve(query, top_k=3)

    used_llm, answer = False, ""
    if use_llm and hits:
        try:
            answer = generator.generate(query, [h["text"] for h in hits])
            used_llm = True
        except Exception as e:  # noqa: BLE001
            logger.warning("LLM call failed (%s); falling back to top chunk.", e)
            answer = hits[0]["text"]
    elif hits:
        answer = hits[0]["text"]
    else:
        answer = "Please slow down and stay alert."

    t_r = time.time()
    out = {
        "answer":          answer.strip(),
        "chunks":          [h["text"] for h in hits],
        "chunks_detailed": hits,
        "action":          "stay alert",
        "urgency":         "medium",
        "confidence":      0.5,
        "latency_sec":     round(t_r - t_q, 4),
        "used_llm":        used_llm,
        "llm_provider":    "ollama" if used_llm else "none",
        "success":         True,
        "timestamp_query": int(t_q * 1000),
        "timestamp_resp":  int(t_r * 1000),
    }
    event_log.append({"type": "ask", **out, "query": query})
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
    logger.info("V1 server listening on :%d  (kb=%d docs)", PORT, len(retriever.docs))
    app.run(host="0.0.0.0", port=PORT)
