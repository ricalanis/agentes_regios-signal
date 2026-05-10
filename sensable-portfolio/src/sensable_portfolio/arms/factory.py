"""Build a LangChain Runnable for an Arm.

The Runnable takes a dict input (decision_id, ts, context_features, signals_at_decision)
and emits an Intervention. The LLM call is abstracted via `llm_factory(model_name)`
so tests can inject a deterministic fake."""
from __future__ import annotations

import re
from typing import Any, Callable

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from sensable_portfolio.arms.registry import ArmRow
from sensable_portfolio.contracts import Intervention

# Known template variables used in arm prompts.
_KNOWN_VARS = frozenset({"decision_id", "ts", "context_features", "signals_at_decision"})


def _escape_literal_braces(text: str) -> str:
    """Escape curly-brace groups that are not known template variables.

    LangChain's f-string template raises KeyError when the prompt contains
    literal text like ``{low, med, high}`` that looks like a variable
    placeholder but is not supplied as input.  We double the braces so
    LangChain treats them as literal output characters.
    """
    def _replace(m: re.Match) -> str:
        inner = m.group(1)
        if inner in _KNOWN_VARS:
            return m.group(0)   # keep original – it's a real variable
        return "{{" + inner + "}}"

    return re.sub(r"\{([^}]+)\}", _replace, text)


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
    system_text = arm.system or f"You are the {arm.persona} arm."
    human_text = arm.human or "Propose one Intervention. Inputs: {context_features} {signals_at_decision}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", _escape_literal_braces(system_text)),
        ("human",  _escape_literal_braces(human_text)),
    ])
    base_llm = llm_factory(arm.model)
    coerce = RunnableLambda(_wrap_to_intervention(arm))
    return prompt | base_llm | coerce
