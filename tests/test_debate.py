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
