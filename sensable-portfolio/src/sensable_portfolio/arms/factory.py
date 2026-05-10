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
from sensable_portfolio.contracts import Intervention, InterventionDraft

# Common field-name aliases open models occasionally emit for our schema.
# When an LLM returns a raw dict instead of an InterventionDraft (e.g. JSON-mode
# without strict schema enforcement), we remap these before constructing.
_DICT_ALIASES = {
    "intervention_type": "action_type",
    "name": "title",
    "instruction": "body",
    "type": "action_type",
    "pattern": "action_type",
}

# Prompt-level safety net for models/endpoints that don't strictly enforce
# the JSON schema bound via `with_structured_output(...)`. Ollama Cloud
# routing has been observed to ignore the `format` parameter on some
# models, so we ALSO instruct the model in plain English. Use double
# braces because LangChain's f-string template interprets single braces.
_JSON_INSTRUCTIONS = (
    "Respond with ONLY a single valid JSON object matching this schema. "
    "No prose, no markdown, no code fences. Required keys exactly:\n"
    '{{ "action_type": "<short kind, e.g. breath, micro_break, reframe>", '
    '"title": "<short title, ≤ 60 chars>", '
    '"body": "<the actionable instruction>", '
    '"duration_s": <number, integer or float>, '
    '"intensity": "low" | "med" | "high", '
    '"rationale": "<one short sentence>" }}'
)

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


def _promote_to_intervention(
    out: Any, *, arm: ArmRow, decision_id: str, ts: float,
) -> Intervention:
    """Stamp server-owned IDs onto whatever the LLM step produced.

    Accepts a fully-formed `Intervention` (legacy fakes), an
    `InterventionDraft` (the canonical bound output), or a raw `dict`
    (open-model JSON-mode falling outside strict schema). Always returns
    a complete `Intervention` with `decision_id`, `arm_id`, and `ts`
    overridden by the values we control."""
    overrides = {"arm_id": arm.id, "decision_id": decision_id, "ts": ts}
    if isinstance(out, Intervention):
        return out.model_copy(update=overrides)
    if isinstance(out, InterventionDraft):
        return Intervention(**out.model_dump(), **overrides)
    if isinstance(out, dict):
        d = {_DICT_ALIASES.get(k, k): v for k, v in out.items()}
        d.pop("schema_version", None)
        d.update(overrides)
        return Intervention(**d)
    raise TypeError(f"Arm {arm.id} returned unexpected type: {type(out)}")


def build_arm_runnable(
    arm: ArmRow,
    llm_factory: Callable[[str], Runnable],
) -> Runnable:
    system_text = arm.system or f"You are the {arm.persona} arm."
    human_text = arm.human or "Propose one Intervention. Inputs: {context_features} {signals_at_decision}"
    # Append JSON instructions to the system prompt (already-escaped literal text)
    system_with_schema = _escape_literal_braces(system_text) + "\n\n" + _JSON_INSTRUCTIONS
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_with_schema),
        ("human",  _escape_literal_braces(human_text)),
    ])
    base_llm = llm_factory(arm.model)
    llm_chain = prompt | base_llm  # produces InterventionDraft (or compat shape)

    async def _ainvoke(inputs: dict[str, Any]) -> Intervention:
        out = await llm_chain.ainvoke(inputs)
        return _promote_to_intervention(
            out, arm=arm,
            decision_id=str(inputs.get("decision_id", "")),
            ts=float(inputs.get("ts", 0.0)),
        )

    return RunnableLambda(_ainvoke)
