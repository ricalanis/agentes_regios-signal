"""Recommendation Protocol + concrete Intervention + WS frame envelopes."""
from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable
from pydantic import BaseModel, Field, ConfigDict


def unix_seconds_to_ms(t: float) -> int:
    """Boundary 3 helper: AffectSample/Intervention ts is unix seconds (float);
    MoodFrame/AgentActionFrame ts is unix ms (int). One conversion site.

    Matches the prior inline behavior (`int(x * 1000)` truncation),
    so existing wire output is byte-identical."""
    return int(t * 1000)


@runtime_checkable
class Recommendation(Protocol):
    schema_version: int
    decision_id: str
    arm_id: str
    ts: float


class Intervention(BaseModel):
    schema_version: Literal[1] = 1
    decision_id: str
    arm_id: str
    ts: float
    action_type: str
    title: str
    body: str
    duration_s: float
    intensity: Literal["low", "med", "high"]
    rationale: str


class AgentInfo(BaseModel):
    id: str
    persona: str
    model: str
    parent_id: str | None = None


class MoodFrame(BaseModel):
    """Spec A frame to ws://127.0.0.1:7777."""
    model_config = ConfigDict(extra="forbid")
    v: Literal[1] = 1
    type: Literal["mood"] = "mood"
    vector: dict[str, float]
    ts: int  # unix ms (Date.now()-style)


class AgentActionFrame(BaseModel):
    """Spec B-shaped action frame; same WS connection."""
    model_config = ConfigDict(extra="forbid")
    v: Literal[1] = 1
    type: Literal["agent_action"] = "agent_action"
    ts: int
    decision_id: str
    agent: AgentInfo
    intervention: Intervention
    signals_at_decision: dict[str, float]
