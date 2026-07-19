"""Shared curl_cffi TLS fingerprint patching for Hyperliquid SDK.

Firefox TLS fingerprints are scrutinized far less than Chrome by
anti-bot/anti-scraping services on cloud egress gateways (e.g. Alibaba Cloud).
Firefox has different cipher suite ordering, extension signaling, and TLS
behavior patterns that are less likely to trigger rate-based blocking.

Import this module BEFORE constructing any Hyperliquid Info/Exchange objects,
since their __init__ methods make API calls immediately.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_cf_requests: Any = None


def _patch_hyperliquid_sdk() -> None:
    """Monkey-patch Hyperliquid SDK's requests module with curl_cffi + Firefox."""
    global _cf_requests
    if _cf_requests is not None:
        return  # already patched

    try:
        from curl_cffi import requests as _cf  # noqa: F811
        _cf_requests = _cf
    except ImportError:
        logger.info("curl_cffi not available — TLS fingerprinting may occur")
        return

    # Override Session() to always impersonate Firefox's TLS fingerprint.
    # The SDK calls requests.Session() with no args — we inject impersonation.
    _OriginalSession = _cf_requests.Session

    def _make_session(**kw):
        return _OriginalSession(impersonate="firefox147", timeout=30, **kw)

    _cf_requests.Session = _make_session  # type: ignore[assignment]

    import hyperliquid.api as _hl_api
    import hyperliquid.exchange as _hl_exchange
    import hyperliquid.info as _hl_info
    _hl_api.requests = _cf_requests
    _hl_exchange.requests = _cf_requests
    _hl_info.requests = _cf_requests

    logger.info("Hyperliquid SDK patched with curl_cffi (Firefox TLS fingerprint)")


# Patch on import
_patch_hyperliquid_sdk()
