from __future__ import annotations

import logging
from typing import Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_core.vectorstores import VectorStore

from config import settings


logger = logging.getLogger(__name__)

def _all_documents(vs: VectorStore) -> list[Document]:
    """
    Pull every stored document out of the vector store so BM25 can
    be built over the same corpus. Both Chroma and FAISS expose this
    in slightly different ways; we handle both.
    """
    # Chroma
    if hasattr(vs, "_collection"): 
        try:
            data = vs._collection.get(include=["documents", "metadatas"])  
            return [
                Document(page_content=t, metadata=m or {})
                for t, m in zip(data.get("documents", []), data.get("metadatas", []))
            ]
        except Exception as e: 
            logger.warning("Chroma enumerate failed: %s", e)

    # FAISS
    if hasattr(vs, "docstore"):
        try:
            store = vs.docstore._dict  
            return list(store.values())
        except Exception as e:  
            logger.warning("FAISS docstore enumerate failed: %s", e)

    logger.warning("Could not enumerate documents; BM25 will be empty.")
    return []


def _build_reranker_compressor():
    """Cross-encoder reranker wrapped as a LangChain compressor."""
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    encoder = HuggingFaceCrossEncoder(model_name=settings.reranker_model)
    return CrossEncoderReranker(model=encoder, top_n=settings.reranker_top_n)


class HybridRetriever:
    """
    Wraps the ensemble + optional reranker, and exposes a single
    `retrieve(query, risk_level=None)` method.

    The metadata filter is applied at query time on the vector-store
    retriever; BM25 filters are applied in Python post-hoc because
    LangChain's BM25Retriever doesn't natively support metadata filters.
    """

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store
        self._all_docs = _all_documents(vector_store)
        if self._all_docs:
            self.bm25 = BM25Retriever.from_documents(self._all_docs)
            self.bm25.k = settings.retrieval_top_k
        else:
            self.bm25 = None

        self._compressor = None
        if settings.use_reranker:
            try:
                self._compressor = _build_reranker_compressor()
                logger.info("Cross-encoder reranker loaded: %s", settings.reranker_model)
            except Exception as e: 
                logger.warning("Reranker disabled (%s); using ensemble only.", e)

    def _dense_retriever(self, risk_level: Optional[int]) -> BaseRetriever:
        search_kwargs: dict = {"k": settings.retrieval_top_k}
        if settings.use_mmr:
            search_type = "mmr"
            search_kwargs["fetch_k"] = max(settings.retrieval_top_k * 3, 20)
            search_kwargs["lambda_mult"] = settings.mmr_lambda
        else:
            search_type = "similarity"

        if risk_level is not None:
            search_kwargs["filter"] = {
                "$or": [
                    {"risk_level": {"$eq": int(risk_level)}},
                    {"risk_level": {"$eq": -1}},
                ]
            }

        return self.vs.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    def _ensemble(self, risk_level: Optional[int]) -> BaseRetriever:
        dense = self._dense_retriever(risk_level)
        if self.bm25 is None:
            return dense
        return EnsembleRetriever(
            retrievers=[self.bm25, dense],
            weights=[settings.bm25_weight, settings.dense_weight],
        )

    def retrieve(
        self,
        query: str,
        risk_level: Optional[int] = None,
    ) -> list[Document]:
        base = self._ensemble(risk_level)

        safe_query = (query or "").strip()
        if not safe_query:
            logger.warning("retrieve() got empty query; returning empty doc list.")
            return []

        if self._compressor is not None:
            pipeline = ContextualCompressionRetriever(
                base_retriever=base,
                base_compressor=self._compressor,
            )
            try:
                docs = pipeline.invoke(safe_query)
            except BaseException as e:  
                logger.warning(
                    "Reranker failed on query (%s); falling back to ensemble only. "
                    "query=%r",
                    type(e).__name__,
                    safe_query[:80],
                )
                try:
                    docs = base.invoke(safe_query)
                except BaseException as e2:  
                    logger.error(
                        "Ensemble retrieval also failed (%s); returning empty list.",
                        type(e2).__name__,
                    )
                    return []
        else:
            try:
                docs = base.invoke(safe_query)
            except BaseException as e: 
                logger.error(
                    "Ensemble retrieval failed (%s); returning empty list.",
                    type(e).__name__,
                )
                return []

        return docs[: settings.final_top_k]
