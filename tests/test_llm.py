"""TradingSignal model + model-registry scoping tests."""

import pytest

from kimi_quant.config import config
from kimi_quant.llm import (
    TradingSignal,
    _build_model_registry,
    create_llm,
)


@pytest.fixture
def both_providers(monkeypatch):
    monkeypatch.setattr(config, "moonshot_api_key", "test-kimi-key")
    monkeypatch.setattr(config, "deepseek_api_key", "test-ds-key")
    monkeypatch.setattr(config, "kimi_model", "kimi-k3")
    monkeypatch.setattr(config, "deepseek_model", "deepseek-v4-pro")


# ─── TradingSignal ────────────────────────────────────────────────────────


def test_actions_reconcile_into_action():
    sig = TradingSignal(action="HOLD", actions=["CLOSE", "SHORT"],
                        confidence=0.8, reasoning="flip")
    assert sig.action == "CLOSE/SHORT"
    assert sig.get_actions() == ["CLOSE", "SHORT"]


def test_get_actions_falls_back_to_single_action():
    sig = TradingSignal(action="LONG", confidence=0.8, reasoning="r")
    assert sig.get_actions() == ["LONG"]


# ─── Model override scoping ───────────────────────────────────────────────


def test_model_override_scoped_to_provider(both_providers):
    registry = _build_model_registry(
        0.1, 2048, model="kimi-k3-thinking", model_provider="kimi",
    )
    assert registry["kimi"].model_name == "kimi-k3-thinking"
    # The OTHER provider must keep its own default — model names are
    # provider-specific and would 400 on the wrong endpoint.
    assert registry["deepseek"].model_name == "deepseek-v4-pro"


def test_model_override_without_provider_is_ignored(both_providers):
    registry = _build_model_registry(0.1, 2048, model="kimi-k3-thinking")
    assert registry["kimi"].model_name == "kimi-k3"
    assert registry["deepseek"].model_name == "deepseek-v4-pro"


def test_registry_skips_providers_without_keys(monkeypatch):
    monkeypatch.setattr(config, "moonshot_api_key", "")
    monkeypatch.setattr(config, "deepseek_api_key", "test-ds-key")
    registry = _build_model_registry(0.1, 2048)
    assert set(registry.keys()) == {"deepseek"}


def test_create_llm_fallback_chain(both_providers):
    llm = create_llm(primary="kimi")
    assert llm is not None
