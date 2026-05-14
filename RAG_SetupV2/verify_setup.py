from __future__ import annotations

import os
import sys
import time
from typing import Optional

from config import settings


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def main() -> int:
    print("=== Be Safe on the Road — OpenRouter setup verification ===\n")

    # 1. .env is loaded?
    print("[1/6] Checking .env...")
    if not settings.openrouter_api_key:
        _fail(
            "OPENROUTER_API_KEY is empty. "
            "Did you `cp .env.example .env` and fill it in?"
        )
        return 1
    if not settings.openrouter_api_key.startswith("sk-or-"):
        _warn(
            f"OPENROUTER_API_KEY does not start with 'sk-or-' "
            f"(starts with {settings.openrouter_api_key[:6]!r}). "
            f"Is this really an OpenRouter key?"
        )
    else:
        _ok(f"OPENROUTER_API_KEY loaded ({settings.openrouter_api_key[:10]}...)")

    # 2. Safety guard active?
    print("\n[2/6] Checking :free safety guard...")
    if not settings.openrouter_free_only:
        _warn(
            "OPENROUTER_FREE_ONLY=false — paid models can be invoked. "
            "This is fine if you have explicit budget approval; otherwise "
            "set OPENROUTER_FREE_ONLY=true in .env."
        )
    else:
        _ok("OPENROUTER_FREE_ONLY=true (paid model IDs will be rejected)")

    # 3. Configured model list looks sane?
    print("\n[3/6] Checking configured model list...")
    from llm import list_configured_openrouter_models, _check_openrouter_model_id
    models = list_configured_openrouter_models()
    if not models:
        _fail("OPENROUTER_MODELS is empty.")
        return 1
    _ok(f"{len(models)} models configured:")
    for m in models:
        try:
            _check_openrouter_model_id(m)
            print(f"      ✓ {m}")
        except RuntimeError as e:
            _fail(f"   {m} REJECTED: {e}")
            return 1

    # 4. langchain_openai importable?
    print("\n[4/6] Checking langchain_openai...")
    try:
        from langchain_openai import ChatOpenAI  # noqa: F401
        _ok("langchain_openai imports OK")
    except ImportError as e:
        _fail(f"langchain_openai not installed: {e}")
        _warn("Run:  pip install -r requirements.txt")
        return 1

    # 5. Live call against the cheapest free model — 1 token answer.
    print("\n[5/6] Making 1 live API call to confirm auth + routing...")
    print(f"      Model: {models[0]}")
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            model=models[0],
            temperature=0.0,
            max_tokens=10,
            timeout=30,
            default_headers={
                "HTTP-Referer": settings.openrouter_referer,
                "X-Title": settings.openrouter_app_title,
            },
        )
        t0 = time.time()
        resp = llm.invoke("Reply with exactly: OK")
        dt = (time.time() - t0) * 1000.0
        text = (getattr(resp, "content", "") or "").strip()
        _ok(f"Response in {dt:.0f}ms: {text!r}")
        if not text:
            _warn(
                "Empty response body. Some free models occasionally return "
                "empty under heavy load — try again or pick a different model."
            )
    except Exception as e:
        _fail(f"Live call failed: {e}")
        msg = str(e).lower()
        if "401" in msg or "unauthorized" in msg or "invalid api key" in msg:
            _warn("→ This looks like an auth error. Check OPENROUTER_API_KEY.")
        elif "402" in msg or "payment" in msg or "credit" in msg:
            _warn(
                "→ Payment-required error. Either the key has no credits "
                "AND the model isn't free, or the :free suffix is wrong. "
                "Check the model ID at https://openrouter.ai/models."
            )
        elif "404" in msg or "not found" in msg:
            _warn(
                "→ Model not found. Free models churn — check the live "
                "list at https://openrouter.ai/models?q=free."
            )
        elif "429" in msg or "rate limit" in msg:
            _warn(
                "→ Rate limited. Free tier is 20 req/min, 200 req/day. "
                "Wait a minute and re-run."
            )
        return 1

    # 6. Judge model reachable?
    print("\n[6/6] Verifying judge model...")
    try:
        from langchain_openai import ChatOpenAI
        judge_llm = ChatOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            model=settings.openrouter_judge_model,
            temperature=0.0,
            max_tokens=10,
            timeout=30,
        )
        t0 = time.time()
        resp = judge_llm.invoke("Reply with exactly: JUDGE_OK")
        dt = (time.time() - t0) * 1000.0
        text = (getattr(resp, "content", "") or "").strip()
        _ok(f"Judge response in {dt:.0f}ms: {text!r}")
    except Exception as e:
        _fail(f"Judge model failed: {e}")
        _warn(
            "→ The judge model may have been removed from the free tier. "
            "Pick a different OPENROUTER_JUDGE_MODEL in .env."
        )
        return 1

    print(f"\n{GREEN}All checks passed.{RESET} Safe to run:")
    print("  python compare_models.py --openrouter --trials 1")
    print("(Use --trials 3 for the final paper numbers, but only after a")
    print(" successful --trials 1 run confirms everything works.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
