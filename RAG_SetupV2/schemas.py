from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SafetyAnswer(BaseModel):
    """
    The LLM must fill every field. A terse 1–2 sentence `answer` is
    what gets read aloud in the car; the other fields drive UI color
    coding and post-trip analysis.
    """

    answer: str = Field(
        ...,
        description="One or two short sentences, eyes-free-safe, "
        "directly responsive to the driver's question.",
    )
    action: str = Field(
        ...,
        description="A single imperative action for the driver "
        "(e.g. 'slow down', 'increase following distance', "
        "'no action needed').",
    )
    urgency: str = Field(
        ...,
        description="One of: low, medium, high, critical.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Subjective confidence in the answer given the "
        "retrieved context, 0..1.",
    )

class DrivingContext(BaseModel):
    risk_level: Optional[int] = None
    depth_m: Optional[float] = None
    speed_kmh: Optional[float] = None
    speed_mph: Optional[float] = None
    behavior: Optional[str] = None
    headway_s: Optional[float] = None

    def as_prompt_lines(self) -> str:
        parts: list[str] = []
        if self.risk_level is not None:
            parts.append(f"risk_level={self.risk_level}")
        if self.depth_m is not None:
            parts.append(f"depth_m={self.depth_m}")
        if self.speed_kmh is not None:
            parts.append(f"speed_kmh={self.speed_kmh}")
        elif self.speed_mph is not None:
            parts.append(f"speed_mph={self.speed_mph}")
        if self.headway_s is not None:
            parts.append(f"headway_s={self.headway_s}")
        if self.behavior:
            parts.append(f"behavior={self.behavior}")
        return ", ".join(parts) if parts else "no telemetry provided"

class RetrievedChunk(BaseModel):
    text: str
    score: float
    id: str
    category: str
    risk_level: int


class AskResponse(BaseModel):
    answer: str
    chunks: list[str]            
    chunks_detailed: list[RetrievedChunk]
    action: str
    urgency: str
    confidence: float
    latency_sec: float
    used_llm: bool
    llm_provider: str
    success: bool
    timestamp_query: int
    timestamp_resp: int
    metadata: dict[str, Any] = Field(default_factory=dict)
