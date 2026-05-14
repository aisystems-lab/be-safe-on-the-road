from __future__ import annotations

import csv
import json
import logging
import os
import time
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.WARNING)

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from config import settings
from rag_chain import RagChain
from report import save_report
from risk import risk_to_message


logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


app = Flask(__name__)
CORS(app)

rag = RagChain()

event_log: list[dict[str, Any]] = []
if settings.event_log_path.exists():
    try:
        with settings.event_log_path.open("r", encoding="utf-8") as f:
            event_log = json.load(f)
    except Exception as e:  
        logger.warning("Could not load event log (%s); starting fresh.", e)
        event_log = []


def _save_events() -> None:
    try:
        with settings.event_log_path.open("w", encoding="utf-8") as f:
            json.dump(event_log[-500:], f, indent=2, ensure_ascii=False)
    except Exception as e: 
        logger.warning("Could not persist events: %s", e)


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "llm_provider": rag.llm_router.active_provider,
            "vector_store": settings.vector_store,
            "embedding_model": settings.embedding_model,
            "reranker": settings.reranker_model if settings.use_reranker else None,
        }
    )


@app.post("/risk_alert")
def risk_alert():
    data = request.get_json(force=True, silent=True) or {}
    try:
        lvl = int(data.get("risk_level", 0))
    except (ValueError, TypeError):
        lvl = 0
    msg = risk_to_message(lvl)
    ts_ms = int(time.time() * 1000)
    event_log.append(
        {
            "type": "risk_alert",
            "timestamp_query": ts_ms / 1000.0,
            "timestamp_resp": ts_ms / 1000.0,
            "level": lvl,
            "message": msg,
            "success": True,
        }
    )
    _save_events()
    return jsonify({"message": msg, "timestamp_ms": ts_ms})


@app.post("/ask")
def ask():
    data = request.get_json(force=True, silent=True) or {}
    q = (data.get("query") or "").strip()
    ctx = data.get("context") or {}

    t_q = time.time()

    if not q:
        return jsonify(
            {
                "answer": "",
                "chunks": [],
                "chunks_detailed": [],
                "action": "",
                "urgency": "low",
                "confidence": 0.0,
                "latency_sec": 0.0,
                "success": False,
                "timestamp_query": int(t_q * 1000),
                "timestamp_resp": int(t_q * 1000),
                "used_llm": False,
                "llm_provider": "none",
            }
        )

    try:
        out = rag.ask(q, context=ctx)
        out["success"] = True
    except Exception as e:
        logger.exception("RAG call failed")
        out = {
            "answer": f"Temporary issue: {e}",
            "chunks": [],
            "chunks_detailed": [],
            "action": "stay alert",
            "urgency": "medium",
            "confidence": 0.0,
            "latency_sec": 0.0,
            "used_llm": False,
            "llm_provider": "none",
            "success": False,
        }

    t_r = time.time()
    out["timestamp_query"] = int(t_q * 1000)
    out["timestamp_resp"] = int(t_r * 1000)

    event_log.append(
        {
            "type": "ask",
            "timestamp_query": t_q,
            "timestamp_resp": t_r,
            "query": q,
            "answer": out["answer"],
            "chunks": out["chunks"],
            "chunks_detailed": out.get("chunks_detailed", []),
            "success": out["success"],
            "used_llm": out["used_llm"],
            "llm_provider": out.get("llm_provider", "none"),
            "latency_sec": out["latency_sec"],
            "action": out.get("action", ""),
            "urgency": out.get("urgency", ""),
            "confidence": out.get("confidence", 0.0),
            "context": ctx,
        }
    )
    _save_events()
    return jsonify(out)


@app.get("/events")
def events():
    lines: list[str] = []
    for e in event_log[-50:]:
        if not isinstance(e, dict):
            lines.append(str(e))
            continue
        ts = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(float(e.get("timestamp_query", 0))),
        )
        if e.get("type") == "ask":
            q = (e.get("query") or "")[:48]
            a = (e.get("answer") or "")[:64]
            lines.append(f"{ts} | Q: {q} | A: {a}")
        elif e.get("type") == "risk_alert":
            lines.append(f"{ts} | L{e.get('level', '?')} | {e.get('message', '')}")
        else:
            lines.append(str(e))
    return jsonify(lines)


@app.get("/report")
def report():
    entries = []
    for e in event_log:
        if not isinstance(e, dict):
            continue
        if e.get("type") not in ("risk_alert", "ask"):
            continue
        ts_ms = int(float(e.get("timestamp_query", 0)) * 1000)
        lvl = e.get("level", -1)
        msg = e.get("message") or e.get("answer") or ""
        entries.append((ts_ms, lvl, msg))
    path = save_report(entries, out_path="trip_report.pdf")
    return send_file(path, as_attachment=True)


@app.get("/export_csv")
def export_csv():
    try:
        out_path = str(settings.results_csv_path)
        fieldnames = [
            "timestamp_query",
            "timestamp_resp",
            "latency_sec",
            "success",
            "query",
            "answer",
            "used_llm",
            "llm_provider",
            "action",
            "urgency",
            "confidence",
        ]
        rows: list[dict[str, Any]] = []
        for e in event_log:
            if not isinstance(e, dict) or e.get("type") != "ask":
                continue
            rows.append(
                {
                    "timestamp_query": e.get("timestamp_query", 0),
                    "timestamp_resp": e.get("timestamp_resp", 0),
                    "latency_sec": e.get("latency_sec", 0),
                    "success": e.get("success", False),
                    "query": (e.get("query") or "").replace("\n", " ").strip(),
                    "answer": (e.get("answer") or "").replace("\n", " ").strip(),
                    "used_llm": e.get("used_llm", False),
                    "llm_provider": e.get("llm_provider", ""),
                    "action": e.get("action", ""),
                    "urgency": e.get("urgency", ""),
                    "confidence": e.get("confidence", 0.0),
                }
            )
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return jsonify({"ok": True, "rows": len(rows), "path": os.path.abspath(out_path)})
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/kb_stats")
def kb_stats_route():
    """Lets the eval harness and README sanity-check the KB."""
    from knowledge_base import kb_stats

    by_category = kb_stats()
    return jsonify(
        {
            "total_curated": sum(by_category.values()),
            "by_category": by_category,
            "vector_store": settings.vector_store,
            "embedding_model": settings.embedding_model,
        }
    )

if __name__ == "__main__":
    logger.info(
        "Starting server on %s:%d (vector=%s, llm=%s)",
        settings.host,
        settings.port,
        settings.vector_store,
        rag.llm_router.active_provider,
    )
    app.run(host=settings.host, port=settings.port)
