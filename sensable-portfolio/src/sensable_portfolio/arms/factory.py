"""Build a LangChain Runnable for an Arm.

The Runnable takes a dict input (decision_id, ts, context_features, signals_at_decision)
and emits an Intervention. The LLM call is abstracted via `llm_factory(model_name)`
so tests can inject a deterministic fake."""
from __future__ import annotations

from typing import Any, Callable

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from sensable_portfolio.arms.registry import ArmRow
from sensable_portfolio.contracts import Intervention


def _wrap_to_intervention(arm: ArmRow) -> Callable[[Any], Intervention]:
    def _coerce(out: Any) -> Intervention:
        if isinstance(out, Intervention):
            return out.model_copy(update={"arm_id": arm.id})
        if isinstance(out, dict):
            return Intervention(**out)
        raise TypeError(f"Arm {arm.id} returned unexpected type: {type(out)}")
    return _coerce


def build_arm_runnable(
    arm: ArmRow,
    llm_factory: Callable[[str], Runnable],
) -> Runnable:
    prompt = ChatPromptTemplate.from_messages([
        ("system", arm.system or f"You are the {arm.persona} arm."),
        ("human",  arm.human  or "Propose one Intervention. Inputs: {context_features} {signals_at_decision}"),
    ])
    base_llm = llm_factory(arm.model)
    coerce = RunnableLambda(_wrap_to_intervention(arm))
    return prompt | base_llm | coerce
