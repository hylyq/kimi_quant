"""Shared test fixtures.

Tests must not depend on the user's real .env — the Config singleton loads
it at import time, so every test that touches executor/network paths
monkeypatches the relevant config attributes explicitly.
"""

import pytest

from kimi_quant.config import config


@pytest.fixture(autouse=True)
def _offline_defaults(monkeypatch):
    """Force offline-safe, deterministic config for every test.

    The Config singleton loads the developer's real .env at import time —
    values like MAX_POSITION_SIZE would otherwise leak into assertions.
    """
    monkeypatch.setattr(config, "dry_run", True)
    monkeypatch.setattr(config, "hl_testnet", True)
    monkeypatch.setattr(config, "hl_private_key", "")
    monkeypatch.setattr(config, "trading_pair", "BTC")
    monkeypatch.setattr(config, "max_position_size", 0.01)
    monkeypatch.setattr(config, "min_confidence", 0.7)
    monkeypatch.setattr(config, "max_leverage", 3)
