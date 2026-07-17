"""Notification via larky's WeChatClient (proven method) or LarkBot (Feishu).

Uses the same WeChatClient.notify() path that cryptoguard uses. A fresh
WeChatClient is created per-send to avoid cross-thread event-loop issues
with redis.asyncio connection pools.

Usage:
    from kimi_quant.notify import notify
    notify.send("Trade opened: LONG 0.01 BTC @ $87000")
"""

import asyncio
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ─── Detection ────────────────────────────────────────────────────────────

_channel: str | None = None  # "larky" | "lark" | None

# 1) Try larky WeChatClient (same approach as cryptoguard — proven working)
try:
    from larky import WeChatClient as _WeChatClient  # noqa: F401
    _channel = "larky"
    logger.info("Notification: larky WeChatClient available (WeChat via Redis)")
except ImportError:
    logger.info("Notification: larky not available")

# 2) Fallback: Feishu via LarkBot
if _channel is None:
    try:
        from larky import LarkBot  # type: ignore[import-untyped]

        if os.getenv("APP_ID") and os.getenv("APP_SECRET"):
            _channel = "lark"
            logger.info("Notification: Feishu detected")
    except ImportError:
        pass


# ─── Feishu Background ────────────────────────────────────────────────────
# (kept for LarkBot fallback — uses its own single-purpose thread)
import queue as _queue_mod

_queue: _queue_mod.Queue[str | None] = _queue_mod.Queue()
_lark_started = False
_lark_thread: threading.Thread | None = None


def _run_lark_loop() -> None:
    import asyncio as _asyncio

    async def _worker(bot: Any) -> None:
        try:
            await bot.start()
            logger.info("Feishu notification bot started")
            while True:
                text = _queue.get()
                if text is None:
                    break
                try:
                    await bot.send_text(text)
                except Exception as e:
                    logger.error("Feishu send failed: %s", e)
        except Exception as e:
            logger.error("Feishu bot error: %s", e)
        finally:
            try:
                await bot.close()
            except Exception:
                pass
            logger.info("Feishu notification bot stopped")

    loop = _asyncio.new_event_loop()
    _asyncio.set_event_loop(loop)
    from larky import LarkBot as _LarkBot
    bot = _LarkBot.from_env()
    loop.run_until_complete(_worker(bot))
    loop.close()


# ─── Public API ───────────────────────────────────────────────────────────


class Notifier:
    """Singleton notifier. Auto-detects larky WeChatClient > Feishu > silent."""

    def send(self, text: str, priority: str = "high") -> None:
        """Send a notification. No-op if no channel available.

        Args:
            text: Plain text message.
            priority: "high" (queued if offline) or "normal" (fire-and-forget).
        """
        if _channel == "larky":
            self._send_larky(text, priority)
        elif _channel == "lark":
            self._send_lark(text)

    def _send_larky(self, text: str, priority: str) -> None:
        """Send via larky WeChatClient.notify() — same path as cryptoguard.

        Creates a fresh WeChatClient per send. This avoids cross-thread
        event-loop issues with redis.asyncio connection pools. The overhead
        (one new Redis connection per notification) is negligible."""
        try:
            wc = _WeChatClient(
                source="kimi-quant",
                redis_host=os.getenv("REDIS_HOST", "localhost"),
                redis_port=int(os.getenv("REDIS_PORT", "6379")),
                redis_db=int(os.getenv("REDIS_DB", "0")),
            )
            asyncio.run(wc.notify(text, priority=priority))
            logger.debug("larky notify sent: %s", text[:60])
        except Exception as e:
            logger.error("larky notify failed: %s", e)

    def _send_lark(self, text: str) -> None:
        """Enqueue message for the Feishu background bot thread."""
        global _lark_started, _lark_thread
        if not _lark_started:
            _lark_thread = threading.Thread(
                target=_run_lark_loop, name="notify-lark", daemon=True
            )
            _lark_thread.start()
            _lark_started = True

        try:
            _queue.put_nowait(text)
        except _queue_mod.Full:
            logger.warning("Notification queue full — message dropped")

    def is_available(self) -> bool:
        return _channel is not None

    @property
    def channel(self) -> str | None:
        return _channel

    def shutdown(self) -> None:
        global _event_loop
        if _channel == "lark" and _lark_thread and _lark_thread.is_alive():
            _queue.put(None)
            _lark_thread.join(timeout=10)
        if _event_loop and _event_loop.is_running():
            _event_loop.call_soon_threadsafe(_event_loop.stop)
            if _loop_thread and _loop_thread.is_alive():
                _loop_thread.join(timeout=5)


notify = Notifier()
