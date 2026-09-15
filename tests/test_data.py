"""_SnapshotCache (TTL/single-flight/atomic write) + cycle-diff tests."""

import kimi_quant.data as data_mod
from kimi_quant.data import _SnapshotCache


def use_tmp_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(data_mod, "_CACHE_DIR", tmp_path)


def test_fresh_ttl_dedupes_concurrent_fetches(monkeypatch, tmp_path):
    """Two fetches within the TTL window share one upstream call."""
    use_tmp_cache(monkeypatch, tmp_path)
    cache = _SnapshotCache("l2", fresh_ttl=60.0)
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return {"levels": [[], []]}

    assert cache.fetch(fetch) == {"levels": [[], []]}
    assert cache.fetch(fetch) == {"levels": [[], []]}
    assert calls["n"] == 1


def test_fetch_after_ttl_expires(monkeypatch, tmp_path):
    use_tmp_cache(monkeypatch, tmp_path)
    cache = _SnapshotCache("l2", fresh_ttl=0.0)  # no dedupe
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return calls["n"]

    assert cache.fetch(fetch) == 1
    assert cache.fetch(fetch) == 2


def test_stale_fallback_on_error(monkeypatch, tmp_path):
    use_tmp_cache(monkeypatch, tmp_path)
    cache = _SnapshotCache("l2")

    cache.fetch(lambda: {"ok": True})  # seed the cache

    def failing():
        raise ConnectionError("network down")

    assert cache.fetch(failing) == {"ok": True}  # stale copy served


def test_error_propagates_without_cache(monkeypatch, tmp_path):
    use_tmp_cache(monkeypatch, tmp_path)
    cache = _SnapshotCache("l2")

    def failing():
        raise ConnectionError("network down")

    import pytest
    with pytest.raises(ConnectionError):
        cache.fetch(failing)


def test_expired_stale_cache_not_served(monkeypatch, tmp_path):
    use_tmp_cache(monkeypatch, tmp_path)
    cache = _SnapshotCache("l2", max_age=0.0)
    cache.fetch(lambda: {"ok": True})

    def failing():
        raise ConnectionError("down")

    import pytest
    with pytest.raises(ConnectionError):
        cache.fetch(failing)


def test_disk_write_is_valid_json(monkeypatch, tmp_path):
    use_tmp_cache(monkeypatch, tmp_path)
    cache = _SnapshotCache("l2")
    cache.fetch(lambda: {"a": 1})
    import json
    payload = json.loads((tmp_path / "l2.json").read_text())
    assert payload["data"] == {"a": 1}
    assert payload["ts"] > 0


def test_no_tmp_files_left_behind(monkeypatch, tmp_path):
    use_tmp_cache(monkeypatch, tmp_path)
    cache = _SnapshotCache("l2", fresh_ttl=0.0)
    cache.fetch(lambda: {"a": 1})
    cache.fetch(lambda: {"a": 2})
    leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


# ─── Entry discipline (chase / counter-trend context) ─────────────────────


from kimi_quant.data import TimeframeSummary as TS, _build_entry_discipline


def _tf(interval, trend, chg, atr_pct, n=60):
    per_candle_h = {"5m": 5 / 60, "15m": 0.25, "1h": 1.0, "4h": 4.0}[interval]
    return TS(interval=interval, num_candles=n, duration_hours=n * per_candle_h,
              trend=trend, change_pct=chg, current_close=80000, period_open=80000 - chg * 800,
              period_high=80200, period_low=78900, current_range_pct=0.3,
              total_volume=900, avg_volume=950, volume_trend="steady",
              atr=80000 * atr_pct / 100, atr_pct=atr_pct)


def test_steady_grind_is_not_extended():
    tfs = [_tf("15m", "up", 0.35, 0.15), _tf("1h", "up", 1.4, 0.48), _tf("4h", "up", 4.5, 0.88)]
    # 0.35% / (0.15% × √60) = 0.30× — a calm grind must NOT warn
    ctx = _build_entry_discipline(tfs)
    assert "EXTENDED" not in ctx
    assert "Trend 1h+4h: UP" in ctx


def test_spike_is_flagged_extended():
    # +2.5% over the 15m window vs 0.15% ATR → 2.5/(0.15×√60) = 2.2×
    tfs = [_tf("15m", "up", 2.5, 0.15), _tf("1h", "up", 2.8, 0.48), _tf("4h", "up", 3.0, 0.88)]
    ctx = _build_entry_discipline(tfs)
    assert "EXTENDED" in ctx
    assert "2.2×" in ctx


def test_no_short_timeframes_no_crash():
    assert _build_entry_discipline([_tf("1h", "up", 3.0, 0.5), _tf("4h", "up", 5.0, 0.9)]) == ""
    assert _build_entry_discipline([]) == ""
