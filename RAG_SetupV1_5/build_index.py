from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


HERE = Path(__file__).parent.resolve()
KB_PATH = HERE / "kb.json"
INDEX_DIR = HERE / "data" / "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("v15.build_index")


def build() -> None:
    records = json.loads(KB_PATH.read_text(encoding="utf-8"))
    logger.info("loaded %d KB records", len(records))

    docs = [
        Document(
            page_content=r["text"],
            metadata={
                "id":         r["id"],
                "category":   r.get("category", ""),
                "risk_level": int(r.get("risk_level", -1)),
                "tags":       ",".join(r.get("tags", [])), 
            },
        )
        for r in records
    ]

    embed = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    vs = FAISS.from_documents(docs, embed)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_DIR))
    logger.info("wrote FAISS index -> %s", INDEX_DIR)


if __name__ == "__main__":
    build()
