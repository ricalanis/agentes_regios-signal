"""SQLModel tables for sensable-portfolio."""
from __future__ import annotations

from sqlmodel import Field, SQLModel


class Decision(SQLModel, table=True):
    id: str = Field(primary_key=True)
    ts: float = Field(index=True)
    arm_id: str = Field(index=True)
    target_id: str = "default"
    context_json: str
    intervention_json: str
    run_id: str | None = None


class Reward(SQLModel, table=True):
    decision_id: str = Field(primary_key=True, foreign_key="decision.id")
    components_json: str
    user_score: float | None = None
    reward: float
    computed_at: float


class Feedback(SQLModel, table=True):
    decision_id: str = Field(primary_key=True, foreign_key="decision.id")
    score: float
    comment: str | None = None
    ts: float


class Arm(SQLModel, table=True):
    id: str = Field(primary_key=True)
    persona: str
    prompt_id: str
    model: str
    parent_id: str | None = None
    created_at: float
    retired_at: float | None = None


class PolicySnapshot(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    ts: float
    algo: str
    blob: bytes


class SnapshotLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    ts: float = Field(index=True)
    kind: str = Field(index=True)
    value: float
