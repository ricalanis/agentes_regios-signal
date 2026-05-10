"""LLM factories for arm Runnables.

Each factory takes a model-name string (from the arm's YAML `model:` field)
and returns a LangChain Runnable that, given the arm's input dict, emits
a structured `Intervention`.

Sentinel: `"default"` and `"fake"` map to the env-configured default model
so seed YAMLs don't need to hard-code a model name."""
from __future__ import annotations

from typing import Any, Callable

from langchain_core.runnables import Runnable

from sensable_portfolio.contracts import Intervention, InterventionDraft

_SENTINEL_NAMES = {"default", "fake", ""}


def ollama_llm_factory(
    default_model: str,
    base_url: str = "http://localhost:11434",
    api_key: str | None = None,
) -> Callable[[str], Runnable]:
    """Return a factory: model_name -> ChatOllama Runnable with structured output.

    Two operating modes:
      - Local daemon (default): base_url=http://localhost:11434, api_key=None.
        The local Ollama daemon proxies `:cloud`-suffixed tags after `ollama signin`.
      - Direct Cloud API: api_key set + base_url="https://ollama.com". The factory
        attaches an Authorization: Bearer header to the underlying httpx client and
        bypasses the local daemon entirely. Use bare model tags (e.g. `kimi-k2.6`)
        in this mode."""
    from langchain_ollama import ChatOllama

    client_kwargs: dict[str, Any] | None = None
    if api_key:
        client_kwargs = {"headers": {"Authorization": f"Bearer {api_key}"}}

    def _factory(model_name: str) -> Runnable:
        actual = default_model if model_name in _SENTINEL_NAMES else model_name
        kwargs: dict[str, Any] = {"model": actual, "base_url": base_url}
        if client_kwargs is not None:
            kwargs["client_kwargs"] = client_kwargs
        chat = ChatOllama(**kwargs)
        return chat.with_structured_output(InterventionDraft, method="json_schema")

    return _factory


def anthropic_llm_factory(
    default_model: str,
    api_key: str | None = None,
) -> Callable[[str], Runnable]:
    """Return a factory: model_name -> ChatAnthropic Runnable with structured output.

    `api_key=None` means ChatAnthropic reads ANTHROPIC_API_KEY from env."""
    from langchain_anthropic import ChatAnthropic

    def _factory(model_name: str) -> Runnable:
        actual = default_model if model_name in _SENTINEL_NAMES else model_name
        kwargs: dict[str, Any] = {"model": actual}
        if api_key:
            kwargs["api_key"] = api_key
        chat = ChatAnthropic(**kwargs)
        return chat.with_structured_output(InterventionDraft)

    return _factory
