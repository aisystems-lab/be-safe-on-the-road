from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings

from config import settings


logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _load_model(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model %s on %s", model_name, device)
    return SentenceTransformer(model_name, device=device)


class BGEEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings with optional query prefix.

    If `query_prefix` is empty, behaves exactly like standard
    SentenceTransformers embeddings (useful for non-BGE models).
    """

    def __init__(
        self,
        model_name: str = settings.embedding_model,
        device: str = settings.embedding_device,
        query_prefix: str = settings.embedding_query_prefix,
        normalize: bool = True,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.query_prefix = query_prefix or ""
        self.normalize = normalize
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model(self.model_name, self.device)
        return self._model

    # -- LangChain API -------------------------------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self.model.encode(
            list(texts),
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        prefixed = f"{self.query_prefix}{text}" if self.query_prefix else text
        vec = self.model.encode(
            [prefixed],
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        return vec.tolist()


def get_embeddings() -> BGEEmbeddings:
    """Factory used across the backend. Keeps a single instance alive."""
    return _embeddings_singleton()


@lru_cache(maxsize=1)
def _embeddings_singleton() -> BGEEmbeddings:
    return BGEEmbeddings()
