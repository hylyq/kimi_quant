"""Debate helpers — placeholder detection (skip-msg compatibility)."""

import asyncio

from kimi_quant.debate import _const, _is_placeholder, _skip_msg


def test_failure_placeholder_detected():
    assert _is_placeholder("[Bull failed to respond: timeout]")
    assert _is_placeholder("[Bear TIMEOUT after 60s — slow]")
    assert _is_placeholder("[BullRebut rebuttal failed: x]")
    assert _is_placeholder("[HoldRebut rebuttal TIMEOUT after 60s — x]")


def test_skip_msg_is_placeholder():
    """Skip placeholders must be recognized so later rounds can skip too."""
    assert _is_placeholder(_skip_msg("BullRebut"))
    assert _is_placeholder(_skip_msg("BearRebut"))
    assert _is_placeholder(_skip_msg("HoldRebut"))


def test_real_arguments_not_placeholders():
    assert not _is_placeholder(
        "Bids are stacking at 99k with 40 BTC — strong support."
    )
    assert not _is_placeholder("[some other bracketed text]")
    # Real argument that merely starts with a bracketed citation
    assert not _is_placeholder("[source] Funding is negative.")


def test_const_coroutine():
    assert asyncio.run(_const("hello")) == "hello"


# ─── History rotation (disk-side containment) ─────────────────────────────


import json as _json

import pytest

import kimi_quant.debate as debate_mod
from kimi_quant.debate import _append_history, _read_history


def _entry(i: int, size: int = 300) -> dict:
    return {
        "cycle_id": f"2026-09-16T00:{i:02d}:00+00:00",
        "bull_argument": "A" * size,
        "bear_argument": "B" * size,
        "final_signal_json": "{}",
    }


def test_history_rotation_trims_old_entries(tmp_path, monkeypatch):
    path = str(tmp_path / "debate.jsonl")
    monkeypatch.setattr(debate_mod, "_HISTORY_MAX_BYTES", 1000)   # ~2 entries
    monkeypatch.setattr(debate_mod, "_HISTORY_KEEP_ENTRIES", 3)

    for i in range(10):
        _append_history(path, _entry(i))

    entries = _read_history(path)
    # Append-then-rotate: the newest cycle is always in the file before the
    # trim runs, so the result is exactly the newest KEEP entries.
    assert len(entries) == 3
    assert entries[0]["cycle_id"].endswith("T00:07:00+00:00")
    assert entries[-1]["cycle_id"].endswith("T00:09:00+00:00")
    # Every remaining line is valid JSON (atomic rewrite, no partials)
    for e in entries:
        assert "bull_argument" in e


def test_history_rotation_not_triggered_below_cap(tmp_path, monkeypatch):
    path = str(tmp_path / "debate.jsonl")
    monkeypatch.setattr(debate_mod, "_HISTORY_MAX_BYTES", 10 * 1024 * 1024)

    for i in range(10):
        _append_history(path, _entry(i))

    assert len(_read_history(path)) == 10          # untouched


def test_rotation_never_drops_below_keep(tmp_path, monkeypatch):
    """Fewer entries than KEEP → rotation must not fire even if size is over."""
    path = str(tmp_path / "debate.jsonl")
    monkeypatch.setattr(debate_mod, "_HISTORY_MAX_BYTES", 500)
    monkeypatch.setattr(debate_mod, "_HISTORY_KEEP_ENTRIES", 1000)

    for i in range(3):
        _append_history(path, _entry(i, size=1000))

    assert len(_read_history(path)) == 3


def test_no_inram_checkpointer_regression():
    """MemorySaver on a constant thread leaked ~176KB/cycle and froze a 2GB
    production server. Guard against accidental re-introduction."""
    import inspect

    import kimi_quant.debate as d

    src = inspect.getsource(d)
    assert "from langgraph.checkpoint" not in src      # no checkpointer import
    assert "checkpointer=" not in src                  # never compiled with one
