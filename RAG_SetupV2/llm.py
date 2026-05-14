from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from config import settings


logger = logging.getLogger(__name__)


def _make_ollama() -> Optional[BaseChatModel]:
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        logger.warning("langchain_ollama not installed; Ollama disabled.")
        return None
    try:
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.ollama_temperature,
            num_predict=settings.max_answer_tokens,
            timeout=settings.ollama_timeout_sec,
        )
    except Exception as e:
        logger.warning("Failed to construct ChatOllama: %s", e)
        return None


def _make_openai() -> Optional[BaseChatModel]:
    if not settings.openai_api_key:
        logger.info("OPENAI_API_KEY not set; OpenAI disabled.")
        return None
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        logger.warning("langchain_openai not installed; OpenAI disabled.")
        return None
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        max_tokens=settings.max_answer_tokens,
        timeout=settings.openai_timeout_sec,
    )


def _make_anthropic() -> Optional[BaseChatModel]:
    if not settings.anthropic_api_key:
        logger.info("ANTHROPIC_API_KEY not set; Anthropic disabled.")
        return None
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        logger.warning("langchain_anthropic not installed; Anthropic disabled.")
        return None
    return ChatAnthropic(
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
        temperature=settings.anthropic_temperature,
        max_tokens=settings.max_answer_tokens,
        timeout=settings.anthropic_timeout_sec,
    )


def _make_google() -> Optional[BaseChatModel]:
    if not settings.google_api_key:
        logger.info("GOOGLE_API_KEY not set; Google Gemini disabled.")
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        logger.warning(
            "langchain_google_genai not installed; Google Gemini disabled."
        )
        return None
    return ChatGoogleGenerativeAI(
        google_api_key=settings.google_api_key,
        model=settings.google_model,
        temperature=settings.google_temperature,
        max_output_tokens=settings.max_answer_tokens,
        timeout=settings.google_timeout_sec,
    )


def _make_groq() -> Optional[BaseChatModel]:
    if not settings.groq_api_key:
        logger.info("GROQ_API_KEY not set; Groq disabled.")
        return None
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        logger.warning("langchain_groq not installed; Groq disabled.")
        return None
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=settings.groq_temperature,
        max_tokens=settings.max_answer_tokens,
        timeout=settings.groq_timeout_sec,
    )


def _make_mistral() -> Optional[BaseChatModel]:
    if not settings.mistral_api_key:
        logger.info("MISTRAL_API_KEY not set; Mistral disabled.")
        return None
    try:
        from langchain_mistralai import ChatMistralAI
    except ImportError:
        logger.warning("langchain_mistralai not installed; Mistral disabled.")
        return None
    return ChatMistralAI(
        api_key=settings.mistral_api_key,
        model=settings.mistral_model,
        temperature=settings.mistral_temperature,
        max_tokens=settings.max_answer_tokens,
        timeout=settings.mistral_timeout_sec,
    )


def _check_openrouter_model_id(model_id: str) -> None:
    """
    Hard guard so a typo or copy-paste from a paid model ID cannot
    silently spend the professor's OpenRouter credits. Even with a $0
    credit cap on the key (which we also recommend), this gives an
    obvious local error instead of a remote 402.
    """
    if settings.openrouter_free_only and not model_id.endswith(":free"):
        raise RuntimeError(
            f"OpenRouter safety guard: model_id={model_id!r} does not end in "
            "':free' but settings.openrouter_free_only=True. Either fix the "
            "model ID or pass --allow-paid (sets OPENROUTER_FREE_ONLY=false)."
        )


def _make_openrouter(model_id: Optional[str] = None) -> Optional[BaseChatModel]:
    """
    Build a ChatOpenAI-compatible client pointed at OpenRouter.

    Because OpenRouter speaks the OpenAI API, we don't need a new SDK —
    langchain_openai's ChatOpenAI handles it via base_url override.
    """
    if not settings.openrouter_api_key:
        logger.info("OPENROUTER_API_KEY not set; OpenRouter disabled.")
        return None
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        logger.warning("langchain_openai not installed; OpenRouter disabled.")
        return None

    chosen = model_id or settings.openrouter_model
    _check_openrouter_model_id(chosen)

    return ChatOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=chosen,
        temperature=settings.openrouter_temperature,
        max_tokens=settings.max_answer_tokens,
        timeout=settings.openrouter_timeout_sec,
        # OpenRouter-recommended headers (optional, used for leaderboards).
        default_headers={
            "HTTP-Referer": settings.openrouter_referer,
            "X-Title": settings.openrouter_app_title,
        },
    )


_PROVIDER_BUILDERS: dict[str, Callable[[], Optional[BaseChatModel]]] = {
    "ollama": _make_ollama,
    "openai": _make_openai,
    "anthropic": _make_anthropic,
    "google": _make_google,
    "groq": _make_groq,
    "mistral": _make_mistral,
    "openrouter": _make_openrouter,
    "none": lambda: None,
}


def provider_display_name(provider: str) -> str:
    if provider == "ollama":
        return f"Ollama {settings.ollama_model}"
    if provider == "openai":
        return f"OpenAI {settings.openai_model}"
    if provider == "anthropic":
        return f"Anthropic {settings.anthropic_model}"
    if provider == "google":
        return f"Google {settings.google_model}"
    if provider == "groq":
        return f"Groq {settings.groq_model}"
    if provider == "mistral":
        return f"Mistral {settings.mistral_model}"
    if provider == "openrouter":
        return f"OpenRouter {settings.openrouter_model}"
    return provider


def available_providers() -> list[str]:
    """Names of the providers whose builder returns a non-None model right now."""
    out: list[str] = []
    for name, builder in _PROVIDER_BUILDERS.items():
        if name == "none":
            continue
        try:
            if builder() is not None:
                out.append(name)
        except Exception as e:  # noqa: BLE001
            logger.debug("Provider %s unavailable: %s", name, e)
    return out


class LLMRouter:
    """
    Routes a prompt to a primary LLM, with a fallback on exceptions.

    Use `.invoke(prompt)` exactly like a plain LangChain LLM.
    """

    def __init__(self, primary_name: str, fallback_name: str):
        self.primary_name = primary_name
        self.fallback_name = fallback_name
        self.primary = _PROVIDER_BUILDERS.get(primary_name, lambda: None)()
        self.fallback = _PROVIDER_BUILDERS.get(fallback_name, lambda: None)()

        if self.primary is None and self.fallback is None:
            logger.warning(
                "No LLM available (primary=%s, fallback=%s). "
                "RAG will run in retrieval-only mode.",
                primary_name,
                fallback_name,
            )

    @property
    def has_llm(self) -> bool:
        return self.primary is not None or self.fallback is not None

    @property
    def active_provider(self) -> str:
        if self.primary is not None:
            return self.primary_name
        if self.fallback is not None:
            return self.fallback_name
        return "none"

    def invoke(self, prompt: Any) -> BaseMessage:
        """
        Try primary, then fallback. Raises the *primary* exception if
        both fail, because that's the one the operator most likely
        wants to debug.
        """
        primary_err: Optional[Exception] = None

        if self.primary is not None:
            try:
                return self.primary.invoke(prompt)
            except Exception as e:  
                primary_err = e
                logger.warning("Primary LLM (%s) failed: %s", self.primary_name, e)

        if self.fallback is not None:
            try:
                logger.info("Falling back to %s.", self.fallback_name)
                return self.fallback.invoke(prompt)
            except Exception as e:  
                logger.error("Fallback LLM (%s) also failed: %s", self.fallback_name, e)

        if primary_err:
            raise primary_err
        raise RuntimeError("No LLM providers configured.")

    def with_structured_output(self, schema):
        """
        Return a router whose .invoke returns instances of `schema`
        (a Pydantic model). All supported providers implement
        `with_structured_output` natively in LangChain.

        Provider quirks
        ---------------
        Google Gemini's default `with_structured_output` uses a
        `response_schema` mode that silently returns None when the
        safety filter trips or when the schema has Pydantic
        constraints (Field ge/le, etc.) that Gemini's JSON validator
        doesn't fully support. We force `method="json_mode"` for
        Google, which is more permissive and asks the model to emit
        JSON via the prompt rather than via a strict schema. We then
        parse it ourselves into the Pydantic model.
        """

        def _bind(llm, name):
            if llm is None:
                return None
            if name in ("google", "openrouter"):
                import json
                import re
                from langchain_core.runnables import RunnableLambda

                _JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
                provider_label = name

                def _extract_and_validate(msg):
                    text = getattr(msg, "content", msg)
                    if isinstance(text, list):
                        text = " ".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in text
                        )
                    if not isinstance(text, str):
                        text = str(text)

                    text = text.strip()
                    if text.startswith("```"):
                        text = re.sub(r"^```(?:json)?\s*", "", text)
                        text = re.sub(r"\s*```$", "", text)

                    match = _JSON_RE.search(text)
                    if not match:
                        logger.warning(
                            "%s response had no JSON object; first 200 chars: %r",
                            provider_label, text[:200],
                        )
                        return None

                    try:
                        data = json.loads(match.group(0))
                    except json.JSONDecodeError as e:
                        logger.warning("%s JSON decode failed: %s", provider_label, e)
                        return None

                    if isinstance(data, dict):
                        if "confidence" in data:
                            try:
                                data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
                            except (TypeError, ValueError):
                                data["confidence"] = 0.5
                        if "urgency" in data and isinstance(data["urgency"], str):
                            data["urgency"] = data["urgency"].strip().lower()

                    try:
                        return schema(**data)
                    except Exception as e:
                        logger.warning(
                            "%s schema validation failed: %s; data=%r",
                            provider_label, e, data,
                        )
                        return None

                return llm | RunnableLambda(_extract_and_validate)
            return llm.with_structured_output(schema)

        primary = _bind(self.primary, self.primary_name)
        fallback = _bind(self.fallback, self.fallback_name)

        parent = self

        class _Structured:
            def invoke(self, prompt: Any):
                primary_err = None
                if primary is not None:
                    try:
                        return primary.invoke(prompt)
                    except Exception as e:  
                        primary_err = e
                        logger.warning(
                            "Structured primary LLM (%s) failed: %s",
                            parent.primary_name,
                            e,
                        )
                if fallback is not None:
                    try:
                        logger.info(
                            "Structured fallback to %s.", parent.fallback_name
                        )
                        return fallback.invoke(prompt)
                    except Exception as e:  
                        logger.error(
                            "Structured fallback (%s) failed: %s",
                            parent.fallback_name,
                            e,
                        )
                if primary_err:
                    raise primary_err
                raise RuntimeError("No LLM providers configured.")

        return _Structured()


_router_cache: Optional[LLMRouter] = None


def get_llm() -> LLMRouter:
    """Singleton router driven by settings.llm_provider / settings.llm_fallback."""
    global _router_cache
    if _router_cache is None:
        _router_cache = LLMRouter(settings.llm_provider, settings.llm_fallback)
        logger.info(
            "LLMRouter ready: primary=%s, fallback=%s, active=%s",
            _router_cache.primary_name,
            _router_cache.fallback_name,
            _router_cache.active_provider,
        )
    return _router_cache


def build_llm(provider_name: str, fallback_name: str = "none") -> LLMRouter:
    """
    Build a fresh router for one specific provider, bypassing the singleton.

    Used by `compare_models.py` to benchmark each model in isolation —
    we want each row of the comparison table to reflect exactly one
    provider's behavior, with no silent fallback masking failures.
    """
    return LLMRouter(provider_name, fallback_name)


def openrouter_display_name(model_id: str) -> str:
    """e.g. 'meta-llama/llama-3.3-70b-instruct:free' -> 'OpenRouter Llama 3.3 70B Instruct (free)'."""
    base = model_id.split("/", 1)[-1]
    is_free = base.endswith(":free")
    base = base[: -len(":free")] if is_free else base
    pretty = base.replace("-", " ").title()
    suffix = " (free)" if is_free else ""
    return f"OpenRouter {pretty}{suffix}"


def build_openrouter_llm(model_id: str) -> "LLMRouter":
    """
    Build a fresh, no-fallback LLMRouter that calls a specific OpenRouter
    model ID. The :free safety guard runs inside _make_openrouter, so a
    paid model ID is rejected before any network call is made.

    Used by compare_models.py.
    """
    router = LLMRouter.__new__(LLMRouter)  
    router.primary_name = "openrouter"
    router.fallback_name = "none"
    router.primary = _make_openrouter(model_id)
    router.fallback = None
    if router.primary is None:
        logger.warning(
            "build_openrouter_llm(%s) returned a router with no LLM. "
            "Check OPENROUTER_API_KEY.",
            model_id,
        )
    return router


def list_configured_openrouter_models() -> list[str]:
    """Parse settings.openrouter_models into a clean list of IDs."""
    return [
        m.strip()
        for m in settings.openrouter_models.split(",")
        if m.strip()
    ]