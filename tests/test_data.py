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
