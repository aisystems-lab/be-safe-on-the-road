from __future__ import annotations

import logging
import time
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config import settings
from llm import get_llm
from retrievers import HybridRetriever
from schemas import DrivingContext, SafetyAnswer
from vector_store import load_vector_store, index_exists


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are "Be Safe on the Road", an in-car driving safety assistant.

Your audience is a driver who is currently operating a vehicle. They
cannot read long text. Keep answers calm, concrete, and short.

Hard rules:
1. Always ground your answer in the RETRIEVED CONTEXT below. If the
   context does not cover the question, say so briefly and give a
   conservative default (slow down, increase following distance).
2. Keep `answer` to 1-2 short sentences a driver can hear in traffic.
3. `action` must be a single imperative (e.g. "slow down",
   "increase following distance", "pull over safely",
   "no action needed").
4. `urgency` must be exactly one of: low, medium, high, critical.
   - risk_level 0   -> low
   - risk_level 1-2 -> medium
   - risk_level 3   -> high
   - risk_level 4   -> critical
   - unknown        -> use your best judgment
5. Never invent numeric thresholds that are not in the context.
6. Do not moralize or lecture. No "please drive safely" filler.
"""

FEW_SHOT_EXAMPLES = """\
You always respond with a single JSON object and nothing else.
No preamble, no commentary, no markdown fences, no "let me think".
Start with `{{` and end with `}}`.

Example 1
---------
Telemetry: risk_level=3, speed_kmh=65, behavior=texting
Question: Why did you warn me?
Response:
{{"answer": "You were texting while closing distance on the vehicle ahead.", "action": "put the phone down and increase following distance", "urgency": "high", "confidence": 0.9}}

Example 2
---------
Telemetry: no telemetry provided
Question: What should I do in heavy rain?
Response:
{{"answer": "Drop your speed well below the limit and extend the gap ahead to about 5 seconds.", "action": "reduce speed and increase following distance", "urgency": "medium", "confidence": 0.85}}
"""

USER_PROMPT = """\
Telemetry: {telemetry}

Retrieved context (most relevant safety knowledge):
{context}

Driver question: {question}

Respond with ONLY a JSON object. No prose, no explanation, no thinking out loud.
The JSON must have exactly these four fields: answer, action, urgency, confidence.
Begin your response with the character `{{` and end it with `}}`.
"""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT + "\n" + FEW_SHOT_EXAMPLES),
        ("human", USER_PROMPT),
    ]
)


def _format_context(docs: list[Document]) -> str:
    if not docs:
        return "(no retrieved context available)"
    lines: list[str] = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        tag = meta.get("category", "misc")
        lines.append(f"[{i}] ({tag}) {d.page_content}")
    return "\n".join(lines)


def _enrich_query(query: str, context: DrivingContext) -> str:
    """Add telemetry facts to the query so BM25 and dense both see them."""
    facts = context.as_prompt_lines()
    if facts == "no telemetry provided":
        return query
    return f"{query} [context: {facts}]"


def _chunks_to_api(docs: list[Document]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in docs:
        meta = d.metadata or {}
        out.append(
            {
                "text": d.page_content,
                "score": float(meta.get("score", 0.0)),
                "id": str(meta.get("id", "")),
                "category": str(meta.get("category", "")),
                "risk_level": int(meta.get("risk_level", -1)),
            }
        )
    return out


def _urgency_from_risk(risk_level: Optional[int]) -> str:
    if risk_level is None:
        return "medium"
    mapping = {0: "low", 1: "medium", 2: "medium", 3: "high", 4: "critical"}
    return mapping.get(risk_level, "medium")


def _retrieval_only_answer(docs: list[Document], risk_level: Optional[int]) -> SafetyAnswer:
    """Sensible answer when no LLM is configured or all providers failed."""
    if docs:
        action_docs = [d for d in docs if d.metadata.get("category") == "action"]
        chosen = action_docs[0] if action_docs else docs[0]
        answer_text = chosen.page_content
    else:
        answer_text = (
            "Please slow down, increase following distance, and stay alert."
        )
    return SafetyAnswer(
        answer=answer_text,
        action="increase following distance",
        urgency=_urgency_from_risk(risk_level),
        confidence=0.4,
    )


class RagChain:
    """
    Public class used by server.py. Keeps a single retriever and LLM
    alive for the lifetime of the Flask process.
    """

    def __init__(self):
        if not index_exists():
            raise RuntimeError(
                "Vector index not found. Run `python prepare_kb.py && "
                "python build_index.py` first."
            )
        self.vs = load_vector_store()
        self.retriever = HybridRetriever(self.vs)
        self.llm_router = get_llm()

        if settings.use_structured_output and self.llm_router.has_llm:
            self.structured_llm = self.llm_router.with_structured_output(SafetyAnswer)
        else:
            self.structured_llm = None

    def ask(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
        k: Optional[int] = None,
    ) -> dict[str, Any]:
        t0 = time.time()
        ctx_model = DrivingContext(**(context or {}))

        enriched_query = _enrich_query(query, ctx_model)

        docs = self.retriever.retrieve(
            enriched_query,
            risk_level=ctx_model.risk_level,
        )
        if k is not None:
            docs = docs[:k]

        used_llm = False
        provider_used = "none"

        if self.structured_llm is not None:
            try:
                prompt_val = _prompt.invoke(
                    {
                        "telemetry": ctx_model.as_prompt_lines(),
                        "context": _format_context(docs),
                        "question": query,
                    }
                )
                safety: SafetyAnswer = self.structured_llm.invoke(prompt_val)
                if safety is None:
                    logger.warning(
                        "Structured LLM (%s) returned None; using retrieval only.",
                        self.llm_router.active_provider,
                    )
                    safety = _retrieval_only_answer(docs, ctx_model.risk_level)
                else:
                    used_llm = True
                    provider_used = self.llm_router.active_provider
            except Exception as e:  
                logger.warning("Structured LLM call failed: %s; using retrieval only.", e)
                safety = _retrieval_only_answer(docs, ctx_model.risk_level)
        else:
            safety = _retrieval_only_answer(docs, ctx_model.risk_level)

        latency = round(time.time() - t0, 4)

        chunks_text = [d.page_content for d in docs]
        return {
            "answer": safety.answer.strip(),
            "chunks": chunks_text,
            "chunks_detailed": _chunks_to_api(docs),
            "action": safety.action,
            "urgency": safety.urgency,
            "confidence": float(safety.confidence),
            "latency_sec": latency,
            "used_llm": used_llm,
            "llm_provider": provider_used,
        }

    def retrieve_only(self, query: str, k: int = 5) -> list[Document]:
        return self.retriever.retrieve(query)[:k]