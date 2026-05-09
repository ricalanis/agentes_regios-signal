"""Arm catalog: YAML-backed; new rows can be added at runtime (evolver)."""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from importlib import resources

import yaml


@dataclass(frozen=True)
class ArmRow:
    id: str
    persona: str
    prompt_id: str
    model: str
    parent_id: str | None
    created_at: float
    retired_at: float | None = None
    system: str = ""
    human: str = ""


class ArmRegistry:
    def __init__(self, rows: list[ArmRow]):
        self._rows: dict[str, ArmRow] = {r.id: r for r in rows}

    @classmethod
    def from_default_pkg(cls) -> "ArmRegistry":
        pkg = resources.files("sensable_portfolio.arms.prompts")
        rows: list[ArmRow] = []
        for path in pkg.iterdir():
            if path.suffix not in (".yaml", ".yml"):
                continue
            data = yaml.safe_load(path.read_text())
            rows.append(ArmRow(
                id=data["id"], persona=data["persona"], prompt_id=data["id"],
                model=data["model"], parent_id=None, created_at=time.time(),
                system=data.get("system", ""), human=data.get("human", ""),
            ))
        return cls(rows)

    def active_arms(self) -> list[ArmRow]:
        return [r for r in self._rows.values() if r.retired_at is None]

    def all_arms(self) -> list[ArmRow]:
        return list(self._rows.values())

    def add(self, row: ArmRow) -> None:
        if row.id in self._rows:
            raise ValueError(f"arm {row.id} already exists")
        self._rows[row.id] = row

    def retire(self, arm_id: str, at: float) -> None:
        if arm_id not in self._rows:
            return
        self._rows[arm_id] = replace(self._rows[arm_id], retired_at=at)

    def get(self, arm_id: str) -> ArmRow:
        return self._rows[arm_id]
