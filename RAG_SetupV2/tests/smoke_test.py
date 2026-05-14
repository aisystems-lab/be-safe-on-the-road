"""
tests/smoke_test.py
-------------------
End-to-end smoke test that exercises the whole pipeline WITHOUT
downloading any HuggingFace models. It monkey-patches `get_embeddings`
to return a tiny deterministic hash-based embedder, then runs:

    prepare_kb -> build_index -> RagChain.ask (retrieval-only)

This validates that every module imports cleanly, the Document/metadata
plumbing is correct, both FAISS and Chroma work, BM25 integrates, and
the chain returns the expected API shape.

Run from the project root:
    python tests/smoke_test.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

# Make the project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force retrieval-only + no reranker so we don't need HF downloads
os.environ["LLM_PROVIDER"] = "none"
os.environ["LLM_FALLBACK"] = "none"
os.environ["USE_RERANKER"] = "false"
os.environ["USE_STRUCTURED_OUTPUT"] = "false"


import numpy as np
from langchain_core.embeddings import Embeddings


# ---------------------------------------------------------------------------
# Fake embeddings: hash-based, deterministic, no downloads.
# Good enough to validate plumbing; not good enough to judge retrieval quality.
# ---------------------------------------------------------------------------
class FakeEmbeddings(Embeddings):
    DIM = 32  # SHA-256 gives 32 bytes

    def _embed(self, text: str) -> list[float]:
        # Deterministic pseudo-random vector from the text's SHA-256.
        h = hashlib.sha256(text.lower().encode("utf-8")).digest()
        arr = np.frombuffer(h, dtype=np.uint8)[: self.DIM].astype(np.float32)
        arr = (arr / 127.5) - 1.0
        # Bag-of-words signal so retrieval isn't random
        for tok in text.lower().split():
            idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM
            arr[idx] += 0.5
        # L2-normalize
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


# Patch before any module that uses embeddings is loaded
import embeddings as emb_mod

emb_mod.get_embeddings = lambda: FakeEmbeddings()          # type: ignore[assignment]
emb_mod._embeddings_singleton = lambda: FakeEmbeddings()   # type: ignore[attr-defined]


def _clean():
    from config import settings
    for p in [settings.faiss_index_dir, settings.chroma_index_dir,
              settings.kb_json_path, settings.event_log_path]:
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()


def test_prepare_kb():
    print("\n[1/5] prepare_kb.py ...")
    import prepare_kb
    prepare_kb.main()
    from config import settings
    assert settings.kb_json_path.exists()
    docs = json.loads(settings.kb_json_path.read_text())
    assert len(docs) >= 91, f"expected >=91 KB entries, got {len(docs)}"
    required_fields = {"id", "text", "category", "risk_level", "tags", "source"}
    for d in docs[:5]:
        assert required_fields.issubset(d.keys()), f"missing fields on {d}"
    print(f"    ✔ {len(docs)} entries written to {settings.kb_json_path.name}")


def test_build_and_query(backend: str):
    print(f"\n[backend={backend}] build + query ...")
    os.environ["VECTOR_STORE"] = backend

    # Reload config so settings.vector_store picks up the env change
    import importlib
    import config
    importlib.reload(config)
    import vector_store
    importlib.reload(vector_store)
    import retrievers
    importlib.reload(retrievers)
    import rag_chain
    importlib.reload(rag_chain)

    # Re-patch after reloads
    import embeddings as emb_mod
    emb_mod.get_embeddings = lambda: FakeEmbeddings()  # type: ignore[assignment]
    emb_mod._embeddings_singleton = lambda: FakeEmbeddings()  # type: ignore[attr-defined]

    import build_index
    build_index.main()

    chain = rag_chain.RagChain()
    result = chain.ask(
        "Why did you warn me?",
        context={"risk_level": 3, "behavior": "texting"},
    )

    # Shape checks
    for key in ("answer", "chunks", "chunks_detailed", "action",
                "urgency", "confidence", "latency_sec", "used_llm",
                "llm_provider"):
        assert key in result, f"missing key: {key}"
    assert isinstance(result["chunks"], list) and len(result["chunks"]) > 0
    assert isinstance(result["chunks_detailed"], list)
    assert result["chunks_detailed"][0]["id"]  # ID propagated through metadata
    assert result["used_llm"] is False         # we disabled the LLM
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["urgency"] in {"low", "medium", "high", "critical"}
    assert result["latency_sec"] > 0

    print(f"    ✔ answer (first 80 chars): {result['answer'][:80]}...")
    print(f"    ✔ chunks retrieved      : {len(result['chunks'])}")
    print(f"    ✔ top chunk id          : {result['chunks_detailed'][0]['id']}")
    print(f"    ✔ top chunk category    : {result['chunks_detailed'][0]['category']}")
    print(f"    ✔ urgency               : {result['urgency']}")
    print(f"    ✔ latency_sec           : {result['latency_sec']}")


def test_risk_and_report():
    print("\n[4/5] risk.py + report.py ...")
    from risk import risk_to_message
    assert "Critical" in risk_to_message(4)
    assert "Very low" in risk_to_message(0)
    assert "Unknown" in risk_to_message(99)

    from report import save_report
    from config import settings
    out = settings.data_dir / "test_report.pdf"
    entries = [
        (1_729_000_000_000, 0, "Safe conditions"),
        (1_729_000_010_000, 3, "High risk detected"),
        (1_729_000_020_000, 4, "Critical!"),
    ]
    p = save_report(entries, out_path=str(out))
    assert Path(p).exists() and Path(p).stat().st_size > 500
    print(f"    ✔ risk messages OK")
    print(f"    ✔ PDF report generated: {Path(p).name} ({Path(p).stat().st_size} bytes)")


def test_server_routes_import():
    print("\n[5/5] server.py import ...")
    # Chromadb 0.5.x keeps an in-process singleton keyed by the persist
    # path, and complains if settings differ between reloads within the
    # same Python process. In production the server is its own fresh
    # process, so to mirror that we spawn a subprocess here.
    import subprocess, textwrap
    script = textwrap.dedent(
        f"""
        import os, sys
        os.environ["LLM_PROVIDER"] = "none"
        os.environ["LLM_FALLBACK"] = "none"
        os.environ["USE_RERANKER"] = "false"
        os.environ["USE_STRUCTURED_OUTPUT"] = "false"
        os.environ["VECTOR_STORE"] = "chroma"
        sys.path.insert(0, "{PROJECT_ROOT}")

        # Monkey-patch embeddings before server.py imports RagChain
        import hashlib, numpy as np
        from langchain_core.embeddings import Embeddings
        class _Fake(Embeddings):
            DIM = 32
            def _e(self, t):
                h = hashlib.sha256(t.lower().encode()).digest()
                a = np.frombuffer(h, dtype=np.uint8)[: self.DIM].astype(np.float32)
                a = (a / 127.5) - 1.0
                for tok in t.lower().split():
                    i = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM
                    a[i] += 0.5
                n = float(np.linalg.norm(a))
                return (a / n).tolist() if n > 0 else a.tolist()
            def embed_documents(self, ts): return [self._e(t) for t in ts]
            def embed_query(self, t): return self._e(t)
        import embeddings as em
        em.get_embeddings = lambda: _Fake()
        em._embeddings_singleton = lambda: _Fake()

        import server
        routes = {{r.rule for r in server.app.url_map.iter_rules()}}
        expected = {{"/health","/ask","/risk_alert","/events","/report","/export_csv","/kb_stats"}}
        missing = expected - routes
        if missing:
            print("MISSING:", missing); sys.exit(2)
        print("OK:", sorted(expected))
        """
    )
    res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert res.returncode == 0, (
        f"server subprocess failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )
    print(f"    ✔ all expected routes registered (subprocess import OK)")


def main():
    _clean()
    test_prepare_kb()
    test_build_and_query("faiss")
    test_build_and_query("chroma")
    test_risk_and_report()
    test_server_routes_import()
    print("\n✅ All smoke tests passed.")


if __name__ == "__main__":
    main()
