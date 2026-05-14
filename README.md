# Be Safe on the Road 🚗

A driver safety system combining an Android app, three iterations of a retrieval-augmented generation (RAG) backend, and the supporting ML / data-analysis notebooks.

## Repository layout

```
SafeDriveMonitor/
├── SafeDriveMonitor_App/    # Android app (Kotlin, Gradle) — the user-facing client
├── RAG_SetupV1/             # RAG backend, plain Python (no LangChain)
├── RAG_SetupV1_5/           # RAG backend, partial LangChain adoption
├── RAG_SetupV2/             # RAG backend, full LangChain pipeline — FINAL, integrated with the app
└── notebooks/               # ML and data-analysis notebooks
    ├── Driver_Behaviour_Analysis.ipynb
    ├── Risk_Level_Prediction.ipynb
    └── depth_final.ipynb
```

## The three RAG setups

The three backends exist so the project can compare RAG architectures on the same 34-question evaluation set. They expose the same Flask endpoints (`/ask`, `/risk_alert`, `/events`, `/report`, `/export_csv`), so the Android app talks to any of them without code changes.

| Setup | Stack | Purpose |
| --- | --- | --- |
| `RAG_SetupV1` | Plain Python, FAISS, hand-rolled retrieval | Baseline |
| `RAG_SetupV1_5` | Partial LangChain (retriever + LCEL) | Isolates "using a framework" from advanced techniques |
| `RAG_SetupV2` | Full LangChain: hybrid BM25 + dense, cross-encoder reranking, structured Pydantic output, multi-LLM router | **Production version — integrated with the Android app** |

See each folder's `README.md` for setup and run instructions.

## Android app

`SafeDriveMonitor_App/` is the Kotlin/Gradle Android project. Open it in Android Studio (File → Open → select the folder). It speaks to the v2 RAG backend over HTTP.

Features include 9-language voice support, light/dark mode, color-blind-safe risk indicators, offline speech recognition via Vosk, and trip-report PDF generation. See `SafeDriveMonitor_App/README.md` for details.

## Notebooks

The three notebooks cover the ML components developed alongside the RAG work — driver behavior clustering, risk-level prediction, and the depth-estimation model. They are exploratory and self-contained.

## Configuration

`RAG_SetupV2` needs an LLM API key. Copy `RAG_SetupV2/.env.example` to `RAG_SetupV2/.env` and fill in your key (OpenRouter, OpenAI, etc.). The real `.env` is gitignored and must never be committed.
