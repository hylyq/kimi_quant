"""Optional notification integration with larky (Feishu/Lark bot).

Auto-detects larky availability at startup. If larky is installed and
configured (APP_ID + APP_SECRET env vars), notifications are sent via
a background asyncio thread. Otherwise, all calls are silent no-ops.

Usage (anywhere in kimi_quant):
    from kimi_quant.notify import notify
    notify.send("Trade opened: LONG 0.01 BTC @ $87000")
    notify.send("Circuit breaker activated: 4 consecutive losses")
"""

import logging
import queue
import threading

logger = logging.getLogger(__name__)

# ─── Detection ────────────────────────────────────────────────────────────

_larky_available = False
_bot_thread: threading.Thread | None = None
_queue: queue.Queue[str | None] = queue.Queue()
_started = False

try:
    import os

    from larky import LarkBot  # type: ignore[import-untyped]

    if os.getenv("APP_ID") and os.getenv("APP_SECRET"):
        _larky_available = True
except ImportError:
    pass


# ─── Background Bot Runner ────────────────────────────────────────────────


def _run_bot_loop() -> None:
    """Run the LarkBot in a dedicated asyncio event loop (background thread).

    Reads messages from the thread-safe queue and sends them via Lark API.
    """
    import asyncio

    async def _bot_worker(bot: LarkBot) -> None:
        """Continuously pull messages from the queue and send them."""
        try:
            await bot.start()
            logger.info("Notification bot started")
            while True:
                text = _queue.get()
                if text is None:  # shutdown signal
                    break
                try:
                    await bot.send_text(text)
                except Exception as e:
                    logger.error("Failed to send notification: %s", e)
        except Exception as e:
            logger.error("Notification bot failed to start: %s", e)
        finally:
            try:
                await bot.close()
            except Exception:
                pass
            logger.info("Notification bot stopped")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    bot = LarkBot.from_env()
    loop.run_until_complete(_bot_worker(bot))
    loop.close()


# ─── Public API ───────────────────────────────────────────────────────────


class Notifier:
    """Singleton notifier — call send() to push a message.

    Automatically starts the background bot thread on first use.
    """

    def send(self, text: str) -> None:
        """Send a notification. No-op if larky is unavailable.

        Args:
            text: Plain text message to send.
        """
        if not _larky_available:
            return

        global _started, _bot_thread
        if not _started:
            _bot_thread = threading.Thread(
                target=_run_bot_loop, name="larky-notify", daemon=True
            )
            _bot_thread.start()
            _started = True
            logger.info("Notification system enabled (larky detected)")

        try:
            _queue.put_nowait(text)
        except queue.Full:
            logger.warning("Notification queue full — message dropped")

    def is_available(self) -> bool:
        """Check whether notifications are active."""
        return _larky_available

    def shutdown(self) -> None:
        """Gracefully stop the background bot thread."""
        if _larky_available and _bot_thread and _bot_thread.is_alive():
            _queue.put(None)  # shutdown signal
            _bot_thread.join(timeout=10)


# Module-level singleton
notify = Notifier()
