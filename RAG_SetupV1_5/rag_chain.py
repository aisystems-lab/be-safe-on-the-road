from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings


logger = logging.getLogger("v15.rag_chain")

HERE = Path(__file__).parent.resolve()
INDEX_DIR = HERE / "data" / "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT_STR = """\
You are a driver-safety assistant. Answer in 1-2 short sentences a
driver can hear in traffic. Ground your answer in the context below.

Context:
{context}

Question: {question}

Answer:"""

_prompt = PromptTemplate.from_template(PROMPT_STR)


def _make_ollama():
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        logger.warning("langchain_ollama not installed")
        return None
    try:
        return ChatOllama(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
            temperature=0.2,
            num_predict=200,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("ChatOllama construct failed: %s", e)
        return None


def _make_flant5():
    try:
        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline
        pipe = pipeline(
            "text2text-generation",
            model=os.environ.get("FLAN_MODEL", "google/flan-t5-base"),
            max_new_tokens=200,
        )
        return HuggingFacePipeline(pipeline=pipe)
    except Exception as e:  # noqa: BLE001
        logger.warning("FLAN-T5 construct failed: %s", e)
        return None


def _make_llm():
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    if provider == "ollama":
        return _make_ollama()
    if provider == "flant5":
        return _make_flant5()
    if provider == "none":
        return None
    return _make_ollama()


def _docs_to_detailed(docs: list[Document], scores: Optional[list[float]] = None) -> list[dict[str, Any]]:
    out = []
    for i, d in enumerate(docs):
        m = d.metadata or {}
        out.append({
            "text":       d.page_content,
            "id":         str(m.get("id", "")),
            "score":      float(scores[i]) if scores and i < len(scores) else 0.0,
            "category":   m.get("category", ""),
            "risk_level": int(m.get("risk_level", -1)),
        })
    return out


def _format_context(docs: list[Document]) -> str:
    return "\n".join(f"- {d.page_content}" for d in docs)


class RagChain:
    """Loaded once at server startup and reused across requests."""

    def __init__(self, k: int = 3) -> None:
        self.k = k
        embed = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        if not INDEX_DIR.exists():
            raise RuntimeError(
                f"FAISS index not found at {INDEX_DIR}. "
                "Run `python build_index.py` first."
            )
        self.vs = FAISS.load_local(
            str(INDEX_DIR), embed, allow_dangerous_deserialization=True
        )
        # NOTE: plain similarity search — no MMR, no filter, no hybrid.
        self.retriever = self.vs.as_retriever(search_kwargs={"k": k})
        self.llm = _make_llm()
        self.chain = (_prompt | self.llm | StrOutputParser()) if self.llm is not None else None
        logger.info(
            "V1.5 chain ready (llm=%s, kb_docs=%s)",
            "on" if self.llm is not None else "off",
            self.vs.index.ntotal if hasattr(self.vs, "index") else "?",
        )

    def ask(self, query: str, use_llm: bool = True) -> dict[str, Any]:
        t0 = time.time()

        scored = self.vs.similarity_search_with_score(query, k=self.k)
        docs   = [d for d, _ in scored]
        scores = [float(s) for _, s in scored]

        used_llm = False
        answer   = ""
        if use_llm and self.chain is not None and docs:
            try:
                answer = self.chain.invoke({
                    "context":  _format_context(docs),
                    "question": query,
                })
                used_llm = True
            except Exception as e:  
                logger.warning("LLM invoke failed: %s; using top chunk.", e)
                answer = docs[0].page_content
        elif docs:
            answer = docs[0].page_content
        else:
            answer = "Please slow down and stay alert."

        return {
            "answer":          (answer or "").strip(),
            "chunks":          [d.page_content for d in docs],
            "chunks_detailed": _docs_to_detailed(docs, scores),
            "action":          "stay alert",
            "urgency":         "medium",
            "confidence":      0.5,
            "latency_sec":     round(time.time() - t0, 4),
            "used_llm":        used_llm,
            "llm_provider":    os.environ.get("LLM_PROVIDER", "ollama") if used_llm else "none",
        }
