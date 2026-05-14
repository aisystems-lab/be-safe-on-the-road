# RAG Setup V2

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # then edit as needed
```

### Optional: local LLM via Ollama

```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.1
ollama serve             # leave this running
```

### Optional: cloud LLM

Add a key to `.env`:

```bash
OPENROUTER_API_KEY=sk-or-v1-...   # or OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
```

---

## Running the server

```bash
# 1. Build the knowledge base JSON
python prepare_kb.py

# 2. Build the vector index (Chroma by default)
python build_index.py

# 3. Start the server (pick one)
./run_server_ollama.sh         # Ollama primary
./run_server_openai.sh         # OpenAI primary
./run_server_openrouter.sh     # OpenRouter primary
python server.py               # use whatever .env says
```

Sanity-check it's up:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/kb_stats
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"Why did you warn me?","context":{"risk_level":3,"behavior":"texting"}}'
```

### Pointing the Android app at it

Get your machine's LAN IP (`ipconfig getifaddr en0` on macOS, `hostname -I` on Linux). Open the Android app → Settings → Backend Server → set `http://<your-ip>:8000`. Make sure your phone and laptop are on the same network.

---

## Configuration — swap any component via `.env`

```bash
VECTOR_STORE=faiss                       # or chroma
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5    # scale up if you have GPU
USE_RERANKER=false                       # ablation: turn off cross-encoder
LLM_PROVIDER=openai                      # ollama / openai / anthropic / groq / mistral / google
BM25_WEIGHT=0.0                          # ablation: dense-only retrieval
RETRIEVER_K=5                            # how many chunks to send to the LLM
```

Any ablation row in the evaluation table can be reproduced from one env change and a rerun of `evaluate.py`.

---

## Evaluation

```bash
# Full eval with LLM
python evaluate.py

# Retrieval-only (no LLM needed — fast, deterministic)
python evaluate.py --no-llm

# Custom k values
python evaluate.py --k 1 3 5 10
```

Outputs to `data/`:
- `eval_results.jsonl` — per-query retrieved IDs, ranks, answer, latency
- `eval_summary.csv` — one-row summary

Metrics reported: Precision@k, Recall@k, F1@k for each k; MRR; latency mean / p50 / p95.

### Multi-model comparison

`compare_models.py` runs the full eval set against every configured LLM and emits a side-by-side table:

| Model | Correctness ↑ | Groundedness ↑ | Instruction following ↑ | Latency (ms) ↓ |

Supported providers (enabled by their respective API key):

| Provider | Default model |
|---|---|
| `ollama` (local) | `llama3.1` |
| `openai` | `gpt-4o-mini` |
| `anthropic` | `claude-3-5-sonnet-latest` |
| `google` | `gemini-2.5-flash` |
| `groq` | `llama-3.3-70b-versatile` |
| `mistral` | `mistral-large-latest` |
| `openrouter` | configurable per `OPENROUTER_MODEL` |

**Metrics.** Correctness, Groundedness, and Instruction-following are 0/1 verdicts from an LLM-as-judge (defaults to `openai gpt-4o`, override with `JUDGE_PROVIDER`). Latency is the end-to-end RAG round-trip, averaged across `--trials` per query. Retrieval metrics (P/R/F1@k, MRR) are also captured per model in the extended CSV.

Run it:

```bash
# Auto-detect providers from your .env:
bash run_compare.sh

# Or explicitly:
python compare_models.py --providers ollama openai anthropic google groq mistral --trials 3

# Latency-only sanity run (no judge):
python compare_models.py --no-judge
```

Outputs (`data/`):
- `model_comparison.csv` / `model_comparison.md` — the four-column table
- `model_comparison_full.csv` — extended summary (p50/p95 latency, retrieval P/R/F1@k, MRR)
- `model_comparison.jsonl` — per-(model, query) records with judge verdicts and rationale


---

## Summary in one paragraph

The retrieval-augmented explanation layer is a LangChain LCEL pipeline: driver queries are enriched with telemetry, retrieved from a curated 91-entry safety knowledge base via a hybrid (BM25 + bge-small dense with MMR) retriever, reranked by a bge-reranker-base cross-encoder, and composed into short answers by a local Llama 3.1 (Ollama) model with cloud-LLM fallbacks. Answers are Pydantic-validated into structured fields (`answer`, `action`, `urgency`, `confidence`) before being returned to the Android client.