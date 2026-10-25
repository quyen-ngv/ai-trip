import pytest

from app import providers

_TIER_VARS = [f"{tier}_{field}" for tier in ("CHEAP_LLM", "QUALITY_LLM", "ASSISTANT_LLM")
              for field in ("PROVIDER", "BASE_URL", "API_KEY", "MODEL")]


@pytest.fixture
def switch(monkeypatch):
    for name in _TIER_VARS + ["AI_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_CHAT_MODEL"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant")
    monkeypatch.setenv("CLAUDE_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("CLAUDE_CHEAP_MODEL", "claude-haiku-5-5")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-6-luna")
    monkeypatch.setenv("OPENAI_CHEAP_MODEL", "gpt-6-luna-mini")
    return monkeypatch


def _view(endpoint):
    return endpoint.base_url, endpoint.api_key, endpoint.model


def test_one_switch_moves_every_tier(switch):
    switch.setenv("AI_PROVIDER", "claude")
    assert _view(providers.from_tiers("QUALITY_LLM", "CHEAP_LLM")) == (
        "https://api.anthropic.com", "sk-ant", "claude-sonnet-5-5")
    assert _view(providers.from_tiers("CHEAP_LLM")) == ("https://api.anthropic.com", "sk-ant", "claude-haiku-5-5")
    assert providers.from_tiers("ASSISTANT_LLM", "QUALITY_LLM", "CHEAP_LLM").is_claude

    switch.setenv("AI_PROVIDER", "openai")
    assert _view(providers.from_tiers("QUALITY_LLM")) == ("https://api.openai.com/v1", "sk-openai", "gpt-6-luna")
    assert _view(providers.from_tiers("CHEAP_LLM")) == ("https://api.openai.com/v1", "sk-openai", "gpt-6-luna-mini")


def test_a_value_pinned_on_a_tier_wins_over_the_switch(switch):
    switch.setenv("AI_PROVIDER", "claude")
    switch.setenv("QUALITY_LLM_MODEL", "claude-opus-5-5")
    switch.setenv("CHEAP_LLM_PROVIDER", "openai")
    assert providers.from_tiers("QUALITY_LLM").model == "claude-opus-5-5"
    assert _view(providers.from_tiers("CHEAP_LLM")) == ("https://api.openai.com/v1", "sk-openai", "gpt-6-luna-mini")


def test_social_jobs_follow_the_switch_unless_they_name_a_provider(switch):
    switch.setenv("AI_PROVIDER", "claude")
    extraction = providers.from_provider(None)
    prefilter = providers.from_provider(None, cheap=True)
    assert (extraction.name, extraction.model) == ("ANTHROPIC", "claude-sonnet-5-5")
    assert prefilter.model == "claude-haiku-5-5"
    named = providers.from_provider("OPENAI", "gpt-x")
    assert (named.base_url, named.api_key, named.model) == ("https://api.openai.com/v1", "sk-openai", "gpt-x")


def test_claude_still_reads_the_older_key_name(switch):
    switch.delenv("CLAUDE_API_KEY")
    switch.setenv("ANTHROPIC_API_KEY", "sk-ant-old")
    assert providers.provider_defaults("claude").api_key == "sk-ant-old"


def test_no_switch_and_no_tier_leaves_the_tier_unconfigured(switch):
    assert not providers.from_tiers("QUALITY_LLM").configured
