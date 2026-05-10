"""Unit tests for the Ollama + Anthropic LLM factories.

We avoid real network calls by patching the chat classes at module level."""
from __future__ import annotations

import pytest


def test_ollama_factory_uses_default_for_sentinels(monkeypatch):
    captured = {}

    class FakeChat:
        def __init__(self, model, base_url):
            captured["model"] = model
            captured["base_url"] = base_url

        def with_structured_output(self, _schema, method=None):
            captured["method"] = method
            return self

    import langchain_ollama
    monkeypatch.setattr(langchain_ollama, "ChatOllama", FakeChat, raising=True)

    from sensable_portfolio.llm.factories import ollama_llm_factory

    f = ollama_llm_factory("kimi-k2:1t-cloud", base_url="http://example:1")
    out = f("default")
    assert captured["model"] == "kimi-k2:1t-cloud"
    assert captured["base_url"] == "http://example:1"
    assert captured["method"] == "json_schema"
    assert out is not None


def test_ollama_factory_honors_explicit_model_name(monkeypatch):
    captured = {}

    class FakeChat:
        def __init__(self, model, base_url):
            captured["model"] = model

        def with_structured_output(self, _schema, method=None):
            return self

    import langchain_ollama
    monkeypatch.setattr(langchain_ollama, "ChatOllama", FakeChat, raising=True)

    from sensable_portfolio.llm.factories import ollama_llm_factory
    f = ollama_llm_factory("kimi-k2:1t-cloud")
    f("qwen2.5:14b-instruct")
    assert captured["model"] == "qwen2.5:14b-instruct"


def test_ollama_factory_attaches_bearer_header_when_api_key_set(monkeypatch):
    """Direct Cloud API mode: api_key should attach Authorization: Bearer
    via client_kwargs.headers so requests reach https://ollama.com auth-ed."""
    captured = {}

    class FakeChat:
        def __init__(self, model, base_url, client_kwargs=None):
            captured["model"] = model
            captured["base_url"] = base_url
            captured["client_kwargs"] = client_kwargs

        def with_structured_output(self, _schema, method=None):
            return self

    import langchain_ollama
    monkeypatch.setattr(langchain_ollama, "ChatOllama", FakeChat, raising=True)

    from sensable_portfolio.llm.factories import ollama_llm_factory

    f = ollama_llm_factory("kimi-k2.6", base_url="https://ollama.com", api_key="abc.xyz")
    f("default")
    assert captured["base_url"] == "https://ollama.com"
    assert captured["client_kwargs"] == {"headers": {"Authorization": "Bearer abc.xyz"}}


def test_ollama_factory_omits_client_kwargs_when_no_api_key(monkeypatch):
    """Local-daemon mode: no api_key → don't pass client_kwargs at all,
    preserving prior call shape for downstream tests + behavior."""
    captured = {}

    class FakeChat:
        def __init__(self, model, base_url):
            # If client_kwargs were forwarded here, this signature would mismatch
            # and the test would fail with TypeError. That's the contract.
            captured["model"] = model

        def with_structured_output(self, _schema, method=None):
            return self

    import langchain_ollama
    monkeypatch.setattr(langchain_ollama, "ChatOllama", FakeChat, raising=True)

    from sensable_portfolio.llm.factories import ollama_llm_factory
    f = ollama_llm_factory("kimi-k2:1t-cloud")  # no api_key
    f("default")
    assert captured["model"] == "kimi-k2:1t-cloud"


def test_anthropic_factory_uses_default_for_sentinels(monkeypatch):
    captured = {}

    class FakeChat:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs

        def with_structured_output(self, _schema):
            return self

    import langchain_anthropic
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", FakeChat, raising=True)

    from sensable_portfolio.llm.factories import anthropic_llm_factory
    f = anthropic_llm_factory("claude-haiku-4-5-20251001", api_key="k")
    out = f("default")
    assert captured["model"] == "claude-haiku-4-5-20251001"
    assert captured["kwargs"].get("api_key") == "k"
    assert out is not None


def test_anthropic_factory_honors_explicit_model_name(monkeypatch):
    captured = {}

    class FakeChat:
        def __init__(self, model, **kwargs):
            captured["model"] = model

        def with_structured_output(self, _schema):
            return self

    import langchain_anthropic
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", FakeChat, raising=True)

    from sensable_portfolio.llm.factories import anthropic_llm_factory
    f = anthropic_llm_factory("claude-haiku-4-5-20251001")
    f("claude-sonnet-4-6")
    assert captured["model"] == "claude-sonnet-4-6"
