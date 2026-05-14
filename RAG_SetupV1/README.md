**Install** new requirements:
   ```bash
   pip install -r requirements.txt
   ```
**Pull the LLM** if you haven't already:
   ```bash
   ollama pull llama3.1
   ollama serve &
   ```

## Run the server

```bash
python server.py             # listens on :8001
```


## Run the evaluation

```bash
python evaluate.py                   # full eval with LLM
python evaluate.py --no-llm          # retrieval-only, fast
python evaluate.py --k 1 3 5 10      # custom cutoffs
```

Outputs land in `./data/`:
- `eval_results.jsonl` — one JSON object per query (retrieved IDs,
  answer, latency, P/R/F1, RR)
- `eval_summary.csv` — one-row paper-ready summary
  (mean P/R/F1@k, MRR, latency mean/p50/p95, ISR)


## Why this version is intentionally "primitive"

V1 is the strawman the paper compares V1.5 and V2 against. It uses
the simplest reasonable pipeline:
- one embedding model (MiniLM)
- one index type (FAISS IndexFlatIP, dense top-k only)
- no MMR, no metadata filter, no reranker, no hybrid retrieval
- one LLM call, no fallback, no structured output
- no orchestration framework

