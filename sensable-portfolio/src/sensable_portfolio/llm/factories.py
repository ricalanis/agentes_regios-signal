"""LLM factories for arm Runnables.

Each factory takes a model-name string (from the arm's YAML `model:` field)
and returns a LangChain Runnable that, given the arm's input dict, emits
a structured `Intervention`.

Sentinel: `"default"` and `"fake"` map to the env-configured default model
so seed YAMLs don't need to hard-code a model name."""
from __future__ import annotations

from typing import Any, Callable

from langchain_core.runnables import Runnable

from sensable_portfolio.contracts import Intervention

_SENTINEL_NAMES = {"default", "fake", ""}


def ollama_llm_factory(
    default_model: str,
    base_url: str = "http://localhost:11434",
) -> Callable[[str], Runnable]:
    """Return a factory: model_name -> ChatOllama Runnable with structured output.

    Works for both local Ollama and Ollama Cloud-routed `:cloud` model tags
    (e.g. `kimi-k2:1t-cloud`). The local daemon proxies cloud tags transparently
    after `ollama signin`."""
    from langchain_ollama import ChatOllama

    def _factory(model_name: str) -> Runnable:
        actual = default_model if model_name in _SENTINEL_NAMES else model_name
        chat = ChatOllama(model=actual, base_url=base_url)
        return chat.with_structured_output(Intervention, method="json_schema")

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
        return chat.with_structured_output(Intervention)

    return _factory
