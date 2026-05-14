from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from vector_store import build_vector_store


logger = logging.getLogger(__name__)
logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")


def _load_kb() -> list[dict[str, Any]]:
    if not settings.kb_json_path.exists():
        raise FileNotFoundError(
            f"KB JSON not found at {settings.kb_json_path}. "
            f"Run `python prepare_kb.py` first."
        )
    with settings.kb_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_documents(entries: list[dict[str, Any]]) -> list[Document]:
    """Wrap KB rows as Documents, chunking any that exceed chunk_size."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs: list[Document] = []
    for entry in entries:
        text: str = entry["text"]
        base_meta = {
            "id": entry["id"],
            "category": entry.get("category", "misc"),
            "risk_level": int(entry.get("risk_level", -1)),
            "tags": ",".join(entry.get("tags", [])),
            "source": entry.get("source", "curated"),
        }

        if len(text) <= settings.chunk_size:
            docs.append(Document(page_content=text, metadata=base_meta))
            continue

        for i, chunk in enumerate(splitter.split_text(text)):
            meta = dict(base_meta)
            meta["id"] = f"{entry['id']}#c{i}"
            meta["parent_id"] = entry["id"]
            docs.append(Document(page_content=chunk, metadata=meta))

    return docs


def main() -> None:
    settings.ensure_dirs()
    entries = _load_kb()
    docs = _to_documents(entries)

    logger.info(
        "Indexing %d chunks from %d KB entries using backend=%s, embeddings=%s",
        len(docs),
        len(entries),
        settings.vector_store,
        settings.embedding_model,
    )
    build_vector_store(docs)

    print(
        f"✅ Built {settings.vector_store} index with {len(docs)} chunks "
        f"from {len(entries)} KB entries."
    )
    print(f"   Embedding model: {settings.embedding_model}")
    if settings.vector_store == "faiss":
        print(f"   Path: {settings.faiss_index_dir}")
    else:
        print(f"   Path: {settings.chroma_index_dir}")


if __name__ == "__main__":
    main()
