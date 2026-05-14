from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).parent.resolve()


ProviderName = Literal[
    "ollama", "openai", "anthropic", "google", "groq", "mistral",
    "openrouter", "none"
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -------- Server --------
    host: str = "0.0.0.0"
    port: int = 8003
    log_level: str = "INFO"

    # -------- Paths --------
    data_dir: Path = PROJECT_ROOT / "data"
    kb_json_path: Path = PROJECT_ROOT / "data" / "rag_meta.json"
    faiss_index_dir: Path = PROJECT_ROOT / "data" / "faiss_index"
    chroma_index_dir: Path = PROJECT_ROOT / "data" / "chroma_index"
    event_log_path: Path = PROJECT_ROOT / "event_log.json"
    results_csv_path: Path = PROJECT_ROOT / "results.csv"
    lstm_csv_path: Optional[Path] = PROJECT_ROOT / "lstm_data.csv"

    # -------- Vector store --------
    # "chroma" -> persistent, good metadata filtering (recommended default)
    # "faiss"  -> faster in-memory, limited metadata filtering
    vector_store: Literal["chroma", "faiss"] = "chroma"
    chroma_collection_name: str = "besafe_kb"

    # -------- Embeddings --------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"
    embedding_query_prefix: str = (
        "Represent this sentence for searching relevant passages: "
    )

    # -------- Reranker (cross-encoder) --------
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_top_n: int = 3

    # -------- Retrieval --------
    retrieval_top_k: int = 10
    final_top_k: int = 3
    bm25_weight: float = 0.4
    dense_weight: float = 0.6
    use_mmr: bool = True
    mmr_lambda: float = 0.5

    # -------- LLM (production routing) --------
    # Primary provider tried first, fallback used on failure.
    # Server.py and the production RagChain use these.
    llm_provider: ProviderName = "ollama"
    llm_fallback: ProviderName = "openai"

    # ---- Ollama ----
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1"
    ollama_temperature: float = 0.2
    ollama_timeout_sec: int = 60

    # ---- OpenAI ----
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.2
    openai_timeout_sec: int = 30

    # ---- Anthropic (Claude) ----
    anthropic_api_key: Optional[str] = Field(default=None)
    anthropic_model: str = "claude-3-5-sonnet-latest"
    anthropic_temperature: float = 0.2
    anthropic_timeout_sec: int = 60

    # ---- Google (Gemini) ----
    google_api_key: Optional[str] = Field(default=None)
    google_model: str = "gemini-2.5-flash"
    google_temperature: float = 0.2
    google_timeout_sec: int = 60

    # ---- Groq (very low-latency hosted Llama / Mixtral) ----
    groq_api_key: Optional[str] = Field(default=None)
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.2
    groq_timeout_sec: int = 60

    # ---- Mistral ----
    mistral_api_key: Optional[str] = Field(default=None)
    mistral_model: str = "mistral-large-latest"
    mistral_temperature: float = 0.2
    mistral_timeout_sec: int = 60

    openrouter_api_key: Optional[str] = Field(default=None)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    openrouter_temperature: float = 0.2
    openrouter_timeout_sec: int = 60
    # Optional, recommended by OpenRouter for leaderboard attribution.
    openrouter_referer: str = "https://besafe-research.local"
    openrouter_app_title: str = "Be Safe on the Road"

    openrouter_models: str = (
        "nvidia/nemotron-3-super-120b-a12b:free,"
        "google/gemma-3-27b-it:free,"
        "minimax/minimax-m2.5:free,"
        "z-ai/glm-4.5-air:free,"
        "meta-llama/llama-3.3-70b-instruct:free",
        "qwen/qwen3-coder:free"
    )
    
    openrouter_judge_model: str = "openai/gpt-oss-120b:free"
    openrouter_free_only: bool = True
    openrouter_per_model_cooldown_sec: float = 4.0
    openrouter_max_retries: int = 5
    openrouter_retry_base_delay_sec: float = 8.0

    # -------- Generation --------
    max_answer_tokens: int = 200
    use_structured_output: bool = True  # Pydantic-validated JSON

    # -------- Chunking (for future doc ingestion) --------
    chunk_size: int = 400
    chunk_overlap: int = 60

    # -------- Evaluation --------
    eval_k_values: tuple[int, ...] = (1, 3, 5)

    # -------- Multi-model comparison (compare_models.py) --------
    compare_providers: str = ""

    # LLM-as-judge settings.
    # The judge is itself an LLM. To keep the comparison fair we use a
    # provider distinct from the system under test where possible.
    # Default: OpenAI gpt-4o (stronger than gpt-4o-mini we test).
    judge_provider: ProviderName = "openai"
    judge_model_openai: str = "gpt-4o"
    judge_model_anthropic: str = "claude-3-5-sonnet-latest"
    judge_model_google: str = "gemini-2.5-pro"
    judge_temperature: float = 0.0  # deterministic-ish judging
    judge_timeout_sec: int = 60

    # How many trials per (model, query). Latencies are averaged across trials.
    compare_trials: int = 1
    # Cooldown between calls to the same provider (seconds) to dodge rate limits.
    compare_cooldown_sec: float = 0.0

    # -------- Observability --------
    langsmith_tracing: bool = False
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "besafe-road"

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_index_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_index_dir.mkdir(parents=True, exist_ok=True)


# Singleton instance imported by the rest of the backend
settings = Settings()
settings.ensure_dirs()
