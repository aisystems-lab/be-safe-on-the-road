"""
tests/smoke_eval.py
-------------------
Runs the evaluation harness with fake embeddings so we can verify
metric computation end-to-end without downloading HF models.

Numbers produced here are NOT meaningful for the paper — the whole
point is just to prove evaluate.py executes cleanly and produces the
expected output artifacts.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["LLM_PROVIDER"] = "none"
os.environ["LLM_FALLBACK"] = "none"
os.environ["USE_RERANKER"] = "false"
os.environ["USE_STRUCTURED_OUTPUT"] = "false"
os.environ["VECTOR_STORE"] = "faiss"

import numpy as np
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    DIM = 32

    def _embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.lower().encode()).digest()
        arr = np.frombuffer(h, dtype=np.uint8)[: self.DIM].astype(np.float32)
        arr = (arr / 127.5) - 1.0
        for tok in text.lower().split():
            idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM
            arr[idx] += 0.5
        n = float(np.linalg.norm(arr))
        return (arr / n).tolist() if n > 0 else arr.tolist()

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


import embeddings as emb_mod
emb_mod.get_embeddings = lambda: FakeEmbeddings()         # type: ignore
emb_mod._embeddings_singleton = lambda: FakeEmbeddings()  # type: ignore


# Ensure index exists
import prepare_kb
prepare_kb.main()
import build_index
build_index.main()


# Now run the evaluator
import evaluate
summary = evaluate.run_eval(use_llm=False, k_values=(1, 3, 5))
evaluate._print_summary(summary)

from config import settings
assert (settings.data_dir / "eval_results.jsonl").exists()
assert (settings.data_dir / "eval_summary.csv").exists()

# Spot-check metric keys
for k in (1, 3, 5):
    for metric in ("precision", "recall", "f1"):
        key = f"{metric}@{k}"
        assert key in summary, f"missing {key}"
        assert 0.0 <= summary[key] <= 1.0, f"bad range for {key}: {summary[key]}"

assert 0.0 <= summary["mrr"] <= 1.0
assert summary["n_queries"] == 34

print("\n✅ Evaluation smoke test passed (34 queries, all metrics in [0,1]).")
