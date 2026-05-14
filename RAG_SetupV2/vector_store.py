from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from config import settings
from embeddings import get_embeddings


logger = logging.getLogger(__name__)


def build_faiss(docs: Sequence[Document]) -> VectorStore:
    from langchain_community.vectorstores import FAISS

    if settings.faiss_index_dir.exists():
        shutil.rmtree(settings.faiss_index_dir)
    settings.faiss_index_dir.mkdir(parents=True, exist_ok=True)

    emb = get_embeddings()
    vs = FAISS.from_documents(list(docs), emb)
    vs.save_local(str(settings.faiss_index_dir))
    logger.info("Built FAISS index at %s with %d docs", settings.faiss_index_dir, len(docs))
    return vs


def build_chroma(docs: Sequence[Document]) -> VectorStore:
    from langchain_chroma import Chroma

    if settings.chroma_index_dir.exists():
        shutil.rmtree(settings.chroma_index_dir)
    settings.chroma_index_dir.mkdir(parents=True, exist_ok=True)

    emb = get_embeddings()
    vs = Chroma.from_documents(
        documents=list(docs),
        embedding=emb,
        collection_name=settings.chroma_collection_name,
        persist_directory=str(settings.chroma_index_dir),
    )
    logger.info(
        "Built Chroma index at %s with %d docs", settings.chroma_index_dir, len(docs)
    )
    return vs


def build_vector_store(docs: Sequence[Document]) -> VectorStore:
    backend = settings.vector_store
    if backend == "faiss":
        return build_faiss(docs)
    if backend == "chroma":
        return build_chroma(docs)
    raise ValueError(f"Unknown vector_store backend: {backend!r}")


def load_faiss() -> VectorStore:
    from langchain_community.vectorstores import FAISS

    emb = get_embeddings()
    return FAISS.load_local(
        str(settings.faiss_index_dir),
        emb,
        allow_dangerous_deserialization=True,
    )


def load_chroma() -> VectorStore:
    from langchain_chroma import Chroma

    emb = get_embeddings()
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=emb,
        persist_directory=str(settings.chroma_index_dir),
    )


def load_vector_store() -> VectorStore:
    backend = settings.vector_store
    if backend == "faiss":
        return load_faiss()
    if backend == "chroma":
        return load_chroma()
    raise ValueError(f"Unknown vector_store backend: {backend!r}")


def index_exists() -> bool:
    """Does the configured index exist on disk?"""
    if settings.vector_store == "faiss":
        return (settings.faiss_index_dir / "index.faiss").exists()
    if settings.vector_store == "chroma":
        return any(settings.chroma_index_dir.iterdir()) if settings.chroma_index_dir.exists() else False
    return False
