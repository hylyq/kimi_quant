"""Optional notification integration via larky (WeChat or Feishu).

Auto-detects available channels at startup. Preference order:
  1. WeChat (if saved account exists on disk)
  2. Feishu/Lark (if APP_ID + APP_SECRET env vars set)
  3. Silent no-op if nothing is available

Usage (anywhere in kimi_quant):
    from kimi_quant.notify import notify
    notify.send("Trade opened: LONG 0.01 BTC @ $87000")
"""

import json
import logging
import os
import queue
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Detection ────────────────────────────────────────────────────────────

_channel: str | None = None  # "wechat" | "lark" | None
_bot_thread: threading.Thread | None = None
_queue: queue.Queue[str | None] = queue.Queue()
_started = False

try:
    from larky import LarkBot, WeChatBot  # type: ignore[import-untyped]

    # Check WeChat first (user preference): saved account on disk
    _wechat_state = os.path.expanduser("~/.openclaw/openclaw-weixin")
    _wechat_index = Path(_wechat_state) / "accounts.json"
    if _wechat_index.exists():
        _channel = "wechat"
        logger.info("Notification: WeChat detected")
    elif os.getenv("APP_ID") and os.getenv("APP_SECRET"):
        _channel = "lark"
        logger.info("Notification: Feishu detected")
except ImportError:
    pass


# ─── Background Bot Runners ───────────────────────────────────────────────


def _run_wechat_loop() -> None:
    """Run WeChatBot in a dedicated asyncio event loop (background thread)."""
    import asyncio

    async def _worker(bot: WeChatBot) -> None:
        try:
            # Load saved account and init session (no polling needed)
            account_ids = bot._list_account_ids()
            if not account_ids:
                logger.error("WeChat: no saved accounts found")
                return
            bot._account = bot._load_account(account_ids[0])
            if not bot._account:
                logger.error("WeChat: failed to load account")
                return
            await bot._init_session()
            logger.info("WeChat notification bot started")
            while True:
                text = _queue.get()
                if text is None:
                    break
                try:
                    await bot.notify(text)
                except Exception as e:
                    logger.error("WeChat send failed: %s", e)
        except Exception as e:
            logger.error("WeChat bot error: %s", e)
        finally:
            try:
                await bot.close()
            except Exception:
                pass
            logger.info("WeChat notification bot stopped")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    bot = WeChatBot.from_env()
    loop.run_until_complete(_worker(bot))
    loop.close()


def _run_lark_loop() -> None:
    """Run LarkBot in a dedicated asyncio event loop (background thread)."""
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
    """Singleton notifier. Auto-detects WeChat > Feishu > silent."""

    def send(self, text: str) -> None:
        """Send a notification. No-op if no channel available."""
        if not _channel:
            return

        global _started, _bot_thread
        if not _started:
            runner = _run_wechat_loop if _channel == "wechat" else _run_lark_loop
            _bot_thread = threading.Thread(
                target=runner, name="notify-bot", daemon=True
            )
            _bot_thread.start()
            _started = True
            logger.info("Notification system enabled (%s)", _channel)

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
        if _channel and _bot_thread and _bot_thread.is_alive():
            _queue.put(None)
            _bot_thread.join(timeout=10)


notify = Notifier()
