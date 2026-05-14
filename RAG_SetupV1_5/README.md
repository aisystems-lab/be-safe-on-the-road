Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
Pull the LLM:
   ```bash
   ollama pull llama3.1
   ollama serve &
   ```

## Run

```bash
# Once:
python build_index.py

# Server:
python server.py                 # listens on :8002

# Eval:
python evaluate.py               # full
python evaluate.py --no-llm      # retrieval-only, fast
```

Eval outputs (to `./data/`):
- `eval_results.jsonl` — one row per query
- `eval_summary.csv`   — paper-ready one-row summary

## FLAN-T5 alternative

The original V1.5 offered FLAN-T5 as an alternative local generator.
That path is preserved:

```bash
export LLM_PROVIDER=flant5
export FLAN_MODEL=google/flan-t5-base
python server.py
```
