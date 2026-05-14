from __future__ import annotations

import logging
import time
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import settings


logger = logging.getLogger(__name__)

class JudgeVerdict(BaseModel):
    """Strict 0/1 verdicts plus a one-sentence rationale per dimension."""

    correctness: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 if the answer is factually correct given the context "
        "and reference notes; 0 otherwise.",
    )
    correctness_reason: str = Field(
        ..., description="One short sentence justifying the correctness verdict."
    )

    groundedness: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 if every factual claim in the answer is supported by "
        "the retrieved context; 0 if any claim is unsupported.",
    )
    groundedness_reason: str = Field(
        ..., description="One short sentence justifying the groundedness verdict."
    )

    instruction_following: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 if the answer obeys all of the driver-assistant "
        "format rules; 0 if any rule is violated.",
    )
    instruction_following_reason: str = Field(
        ..., description="One short sentence justifying the instruction-following verdict."
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = """\
You are a strict, impartial judge evaluating answers from an in-car
driver-safety assistant. The assistant is supposed to answer the
driver's question using only the retrieved knowledge-base context
provided to it, in 1-2 short sentences, plus a single imperative
`action` and an `urgency` from {{low, medium, high, critical}}.

You must return three independent binary verdicts (0 or 1):

1. CORRECTNESS
   Score 1 if and only if the assistant's `answer` is factually
   correct given the retrieved context and the reference notes
   below. Minor paraphrasing is fine. Outright factual errors,
   wrong numeric thresholds, or contradictions of the context are
   score 0.

2. GROUNDEDNESS
   Score 1 if and only if every factual claim in the `answer` is
   directly supported by the RETRIEVED CONTEXT. If the answer adds
   plausible-but-unsupported claims, score 0. Generic safety advice
   (e.g. "drive carefully") that the context does not support is
   score 0.

3. INSTRUCTION-FOLLOWING
   Score 1 if and only if ALL of the following format rules hold:
   - `answer` is at most two sentences and is calm/concrete (no
     moralizing, no "please drive safely" filler).
   - `action` is a single short imperative (e.g. "slow down").
   - `urgency` is exactly one of: low, medium, high, critical.
   - The answer does NOT invent numeric thresholds that are absent
     from the retrieved context.
   Any violation -> score 0.

Be strict. Half-credit is not allowed. When in doubt, score 0 and
explain why. Return JSON only."""

JUDGE_USER = """\
Driver question:
{question}

Retrieved context (the assistant was supposed to ground its answer in this):
{context}

Reference notes (curator-provided ground-truth points for this query):
{reference}

Assistant answer:
{answer}

Assistant action: {action}
Assistant urgency: {urgency}

Return a JSON object with fields:
  correctness, correctness_reason,
  groundedness, groundedness_reason,
  instruction_following, instruction_following_reason."""

_judge_prompt = ChatPromptTemplate.from_messages(
    [("system", JUDGE_SYSTEM), ("human", JUDGE_USER)]
)


# ---------------------------------------------------------------------------
# Reference notes
# ---------------------------------------------------------------------------
REFERENCE_NOTES: dict[str, str] = {
    "Why did you warn me? I'm at risk level 3.":
        "Risk level 3 = high. The driver should slow down and put away distractions.",
    "What does critical risk mean?":
        "Risk level 4 = critical / collision-prone. Immediate action required.",
    "Is this safe? I see level 0 on the screen.":
        "Risk level 0 = low. Driving conditions are nominal; keep alert.",
    "What should I do right now?":
        "At risk level 3, slow down and increase following distance immediately.",
    "Why did the risk jump from low to moderate?":
        "Transitions from low to moderate usually mean a behavior change "
        "(distraction) or worsening conditions. Increase attention and gap.",
    "Is texting really that bad?":
        "Texting greatly increases crash risk; should never be done while moving.",
    "Can I take a phone call while driving?":
        "Hand-held calls are unsafe; even hands-free degrades attention.",
    "I'm feeling sleepy behind the wheel.":
        "Drowsiness is dangerous. Pull over safely and rest.",
    "The driver in front keeps tailgating me.":
        "Don't engage. Maintain speed, change lanes safely if possible.",
    "What's the best way to drive in heavy rain?":
        "Reduce speed below the limit, increase following distance to ~5 sec, "
        "use low beams, avoid hard braking.",
    "My car is skidding on the rain.":
        "Hydroplaning. Ease off the accelerator, do not brake hard, steer straight.",
    "How should I handle fog?":
        "Slow down, use low-beam headlights or fog lights, increase following distance.",
    "Is black ice real?":
        "Yes. It is nearly invisible and very slippery; reduce speed and avoid braking.",
    "How to drive in snow safely?":
        "Reduce speed, increase following distance to 5-6 seconds, smooth inputs.",
    "Sun is blinding me.":
        "Use the sun visor, slow down, increase following distance.",
    "What is the three-second rule?":
        "Maintain at least 3 seconds of following distance behind the vehicle ahead.",
    "Can I turn right on red?":
        "In most jurisdictions yes, after a full stop and only when safe, "
        "unless a sign prohibits it.",
    "Should I speed up on a yellow light?":
        "No. A yellow means stop if it is safe to do so.",
    "What happens if a school bus stops with red lights?":
        "All traffic must stop in both directions until the lights stop flashing.",
    "An ambulance is behind me.":
        "Pull over safely to the right and stop until it has passed.",
    "How do I merge onto the highway?":
        "Match the speed of traffic on the entrance ramp and merge into a gap.",
    "There's a cyclist next to me in traffic.":
        "Leave at least one lane width / 1m+ when passing; do not cut in.",
    "How do I share the road with trucks?":
        "Avoid blind spots, leave extra following distance, do not cut in front.",
    "A deer ran in front of me.":
        "Brake firmly in a straight line. Do not swerve into oncoming traffic.",
    "Driving through a school zone.":
        "Reduce to the posted school-zone speed, watch for pedestrians.",
    "The temperature gauge is going up.":
        "Engine is overheating. Pull over safely and turn off the engine.",
    "My tire just blew out.":
        "Hold the wheel firmly, ease off the accelerator, do not brake hard, "
        "steer straight, and pull over.",
    "The check engine light is flashing.":
        "Flashing CEL = serious. Reduce load, pull over safely, get it checked.",
    "Brake pedal feels soft.":
        "Brake failure risk. Pump the brakes, downshift, use the parking brake "
        "gradually, pull over.",
    "How does the car measure distance to other vehicles?":
        "Stereo vision / depth estimation from forward-facing cameras.",
    "How did you detect I was texting?":
        "A driver-facing CNN classifier detected the texting behavior.",
    "Does this system work without internet?":
        "Yes. Detection runs on-device; the assistant chat needs an LLM "
        "(local Ollama works offline).",
    "Can I trust these warnings?":
        "Warnings are advisory. They have a non-zero false positive rate "
        "and do not replace the driver's judgment.",
    "Why did you warn me at low speed?":
        "Risk depends on context, not just speed; e.g. distraction in a "
        "school zone is risky even at 25 km/h.",
}


def _reference_for(query: str) -> str:
    return REFERENCE_NOTES.get(
        query,
        "(no curated reference; judge correctness based on the retrieved context alone)",
    )


# ---------------------------------------------------------------------------
# Judge backbone
# ---------------------------------------------------------------------------
def _make_judge_llm() -> Optional[BaseChatModel]:
    """
    Build a judge model. We deliberately bypass LLMRouter so the judge
    is a single explicit model — no silent fallback masking which model
    actually graded each row.
    """
    p = settings.judge_provider
    if p == "openai":
        if not settings.openai_api_key:
            return None
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            return None
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.judge_model_openai,
            temperature=settings.judge_temperature,
            timeout=settings.judge_timeout_sec,
        )
    if p == "anthropic":
        if not settings.anthropic_api_key:
            return None
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            return None
        return ChatAnthropic(
            api_key=settings.anthropic_api_key,
            model=settings.judge_model_anthropic,
            temperature=settings.judge_temperature,
            timeout=settings.judge_timeout_sec,
        )
    if p == "google":
        if not settings.google_api_key:
            return None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            return None
        return ChatGoogleGenerativeAI(
            google_api_key=settings.google_api_key,
            model=settings.judge_model_google,
            temperature=settings.judge_temperature,
            timeout=settings.judge_timeout_sec,
        )
    if p == "openrouter":
        if not settings.openrouter_api_key:
            return None
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            return None
        from llm import _check_openrouter_model_id
        _check_openrouter_model_id(settings.openrouter_judge_model)
        return ChatOpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            model=settings.openrouter_judge_model,
            temperature=settings.judge_temperature,
            timeout=settings.judge_timeout_sec,
            default_headers={
                "HTTP-Referer": settings.openrouter_referer,
                "X-Title": settings.openrouter_app_title,
            },
        )
    logger.warning("Unsupported judge_provider=%s; judge disabled.", p)
    return None


class Judge:
    """Wraps the judge LLM with a structured-output adapter."""

    def __init__(self):
        llm = _make_judge_llm()
        if llm is None:
            raise RuntimeError(
                f"Judge LLM ({settings.judge_provider}) could not be constructed. "
                "Set the appropriate API key or change settings.judge_provider."
            )
        self._llm = llm
        if settings.judge_provider in ("openrouter", "google"):
            import json
            import re
            from langchain_core.runnables import RunnableLambda

            _JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
            provider_label = settings.judge_provider

            def _extract(msg):
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
                m = _JSON_RE.search(text)
                if not m:
                    logger.warning(
                        "Judge (%s) returned no JSON object; first 200 chars: %r",
                        provider_label, text[:200],
                    )
                    return None
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError as e:
                    logger.warning("Judge (%s) JSON decode failed: %s", provider_label, e)
                    return None
                if isinstance(data, dict):
                    for k in ("correctness", "groundedness", "instruction_following"):
                        if k in data:
                            v = data[k]
                            if isinstance(v, bool):
                                data[k] = 1 if v else 0
                            elif isinstance(v, str):
                                data[k] = 1 if v.strip().lower() in ("1", "true", "yes") else 0
                            else:
                                try:
                                    data[k] = 1 if int(v) else 0
                                except (TypeError, ValueError):
                                    data[k] = 0
                try:
                    return JudgeVerdict(**data)
                except Exception as e:
                    logger.warning(
                        "Judge (%s) schema validation failed: %s; data=%r",
                        provider_label, e, data,
                    )
                    return None

            self._structured = llm | RunnableLambda(_extract)
        else:
            self._structured = llm.with_structured_output(JudgeVerdict)

    def score(
        self,
        question: str,
        context: str,
        answer: str,
        action: str,
        urgency: str,
    ) -> JudgeVerdict:
        prompt = _judge_prompt.invoke(
            {
                "question": question,
                "context": context,
                "reference": _reference_for(question),
                "answer": answer,
                "action": action,
                "urgency": urgency,
            }
        )
        try:
            return self._structured.invoke(prompt)
        except Exception as e:
            logger.warning("Judge call failed (%s); retrying once after 1s.", e)
            time.sleep(1.0)
            return self._structured.invoke(prompt)


_judge_singleton: Optional[Judge] = None


def get_judge() -> Judge:
    global _judge_singleton
    if _judge_singleton is None:
        _judge_singleton = Judge()
    return _judge_singleton


def aggregate(verdicts: list[JudgeVerdict]) -> dict[str, float]:
    """Convert per-query verdicts into per-model summary scores in [0, 1]."""
    if not verdicts:
        return {"correctness": 0.0, "groundedness": 0.0, "instruction_following": 0.0}
    n = len(verdicts)
    return {
        "correctness": round(sum(v.correctness for v in verdicts) / n, 4),
        "groundedness": round(sum(v.groundedness for v in verdicts) / n, 4),
        "instruction_following": round(
            sum(v.instruction_following for v in verdicts) / n, 4
        ),
    }


def context_str(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks for the judge — same shape the model saw."""
    if not chunks:
        return "(no retrieved context)"
    lines: list[str] = []
    for i, c in enumerate(chunks, 1):
        cat = c.get("category", "misc")
        text = c.get("text", "")
        lines.append(f"[{i}] ({cat}) {text}")
    return "\n".join(lines)