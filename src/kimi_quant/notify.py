"""Notification via larky's Redis Pub/Sub (WeChat) or LarkBot (Feishu).

Auto-detection at import time:
  1. Redis + larky WeChatService → publish to wechat:outgoing channel
  2. larky + Feishu APP_ID/APP_SECRET → LarkBot background thread
  3. None → silent no-op

Usage:
    from kimi_quant.notify import notify
    notify.send("Trade opened: LONG 0.01 BTC @ $87000")
"""

import json
import logging
import os
import queue
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Redis channel used by larky's WeChatService
REDIS_CHANNEL = "wechat:outgoing"

# ─── Detection ────────────────────────────────────────────────────────────

_redis_client = None
_channel: str | None = None  # "redis" | "lark" | None
_bot_thread: threading.Thread | None = None
_queue: queue.Queue[str | None] = queue.Queue()
_started = False

# 1) Try Redis (WeChat via larky WeChatService)
try:
    import redis

    _r = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        socket_connect_timeout=2,
    )
    _r.ping()
    _redis_client = _r
    _channel = "redis"
    logger.info("Notification: Redis detected (WeChat via larky)")
except Exception:
    pass

# 2) Fallback: Feishu via LarkBot
if _channel is None:
    try:
        from larky import LarkBot  # type: ignore[import-untyped]

        if os.getenv("APP_ID") and os.getenv("APP_SECRET"):
            _channel = "lark"
            logger.info("Notification: Feishu detected")
    except ImportError:
        pass


# ─── Lark (Feishu) Background Thread ─────────────────────────────────────


def _run_lark_loop() -> None:
    import asyncio

    async def _worker(bot: LarkBot) -> None:
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

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    bot = LarkBot.from_env()
    loop.run_until_complete(_worker(bot))
    loop.close()


# ─── Public API ───────────────────────────────────────────────────────────


class Notifier:
    """Singleton notifier. Auto-detects Redis > Feishu > silent."""

    def send(self, text: str, priority: str = "high") -> None:
        """Send a notification. No-op if no channel available.

        Args:
            text: Plain text message.
            priority: "high" (queued if offline) or "normal" (fire-and-forget).
                      Only meaningful for Redis/WeChat channel.
        """
        if _channel == "redis":
            self._send_redis(text, priority)
        elif _channel == "lark":
            self._send_lark(text)

    def _send_redis(self, text: str, priority: str) -> None:
        """Publish directly to larky's Redis channel.

        Uses the same payload format as larky's WeChatClient.notify().
        WeChatService picks it up and delivers via WeChat.
        """
        try:
            payload = json.dumps(
                {
                    "text": text,
                    "source": "kimi-quant",
                    "priority": priority,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
            )
            _redis_client.publish(REDIS_CHANNEL, payload)
        except Exception as e:
            logger.error("Redis publish failed: %s", e)

    def _send_lark(self, text: str) -> None:
        """Enqueue message for the Feishu background bot thread."""
        global _started, _bot_thread
        if not _started:
            _bot_thread = threading.Thread(
                target=_run_lark_loop, name="notify-lark", daemon=True
            )
            _bot_thread.start()
            _started = True

        try:
            _queue.put_nowait(text)
        except queue.Full:
            logger.warning("Notification queue full — message dropped")

    def is_available(self) -> bool:
        return _channel is not None

    @property
    def channel(self) -> str | None:
        return _channel

    def shutdown(self) -> None:
        if _channel == "lark" and _bot_thread and _bot_thread.is_alive():
            _queue.put(None)
            _bot_thread.join(timeout=10)
        # Redis: nothing to shut down (connection pooled)


notify = Notifier()
