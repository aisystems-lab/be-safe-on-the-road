from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    """Dense retriever: SentenceTransformers + FAISS IndexFlatIP."""

    def __init__(
        self,
        kb_path: Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.kb_path = Path(kb_path)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.docs: list[dict[str, Any]] = self._load_kb()
        self.index = self._build_index()

    def _load_kb(self) -> list[dict[str, Any]]:
        records = json.loads(self.kb_path.read_text(encoding="utf-8"))
        for r in records:
            r.setdefault("category", "")
            r.setdefault("risk_level", -1)
            r.setdefault("tags", [])
        return records

    def _build_index(self) -> faiss.Index:
        texts = [d["text"] for d in self.docs]
        embs = self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        index = faiss.IndexFlatIP(embs.shape[1])   
        index.add(embs)
        return index

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        q_emb = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        scores, idxs = self.index.search(q_emb, top_k)
        out: list[dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], idxs[0])):
            if idx < 0 or idx >= len(self.docs):
                continue
            d = self.docs[int(idx)]
            out.append({
                "text":       d["text"],
                "id":         d["id"],
                "score":      float(score),
                "category":   d.get("category", ""),
                "risk_level": int(d.get("risk_level", -1)),
            })
        return out
