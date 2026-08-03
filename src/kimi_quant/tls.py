"""Shared curl_cffi TLS fingerprint + multi-IP failover patching for Hyperliquid SDK.

Firefox TLS fingerprints are scrutinized far less than Chrome by
anti-bot/anti-scraping services on cloud egress gateways (e.g. Alibaba Cloud).
Firefox has different cipher suite ordering, extension signaling, and TLS
behavior patterns that are less likely to trigger rate-based blocking.

Multi-IP failover:
CloudFront Geo DNS intermittently resolves api.hyperliquid.xyz to edge IPs
that are unreachable from the server's network path (TCP RST during the TLS
handshake — curl error 35). To be immune to DNS routing, each request is
pinned to a working edge IP via CURLOPT_RESOLVE. The candidate pool is seeded
with the last-known-good IPs (persisted to ~/.kimi_quant/hl_ips.json) plus
fresh DNS answers; on a network-level failure the request is retried against
the next candidate, and successful IPs are promoted to the front of the pool.
DNS answers are re-merged periodically so newly-valid edge IPs can join.

Import this module BEFORE constructing any Hyperliquid Info/Exchange objects,
since their __init__ methods make API calls immediately.
"""

import json
import logging
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_cf_requests: Any = None

# ─── Multi-IP Failover ────────────────────────────────────────────────────

# Connection-level errors mean "the path to this IP is broken — try another".
# HTTP errors (4xx/5xx) are real server answers and must NOT trigger failover;
# the Hyperliquid SDK raises those after session.post returns.
_NETWORK_ERRORS: tuple = ()

_DNS_REFRESH_SECS = 600      # re-resolve DNS every 10 minutes
_BAN_SECS = 30               # IP banned for this long after 2 consecutive failures
_PERSIST_MIN_INTERVAL = 60   # don't rewrite the seed file more than once/min
_PERSIST_PATH = Path.home() / ".kimi_quant" / "hl_ips.json"


def _is_ipv4(s: str) -> bool:
    parts = s.split(".")
    return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


class _IPPool:
    """Candidate edge-IP pool for one host, ordered best-first.

    Thread-safe: requests may come from concurrent threads (data.py parallel
    fetch), but each Session's curl object is thread-local in curl_cffi, so
    only the pool itself needs locking.
    """

    def __init__(self, host: str, port: int = 443):
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._ips: list[str] = []
        self._fail_count: dict[str, int] = {}
        self._ban_until: dict[str, float] = {}
        self._last_dns = 0.0
        self._last_persist = 0.0
        self._load_seed()

    def has_candidates(self) -> bool:
        with self._lock:
            return bool(self._ips)

    def pick(self, exclude: set[str]) -> str | None:
        """Return the best not-excluded, not-banned candidate, or None."""
        with self._lock:
            now = time.time()
            if now - self._last_dns >= _DNS_REFRESH_SECS:
                self._refresh_dns_locked(now)
            for ip in self._ips:
                if ip in exclude:
                    continue
                if self._ban_until.get(ip, 0.0) > now:
                    continue
                return ip
            return None

    def mark_success(self, ip: str) -> None:
        with self._lock:
            self._fail_count[ip] = 0
            self._ban_until.pop(ip, None)
            if self._ips and self._ips[0] != ip:
                # Promote the working IP to the front of the pool
                self._ips = [ip] + [x for x in self._ips if x != ip]
            self._persist_locked()

    def mark_failure(self, ip: str) -> None:
        with self._lock:
            n = self._fail_count.get(ip, 0) + 1
            self._fail_count[ip] = n
            if n >= 2:
                self._ban_until[ip] = time.time() + _BAN_SECS

    # ── internals ──

    def _load_seed(self) -> None:
        """Load persisted last-known-good IPs, then merge fresh DNS answers."""
        try:
            data = json.loads(_PERSIST_PATH.read_text())
            self._ips = [ip for ip in data.get(self.host, []) if _is_ipv4(ip)]
        except FileNotFoundError:
            self._ips = []
        except Exception as e:
            logger.debug("IP seed load failed: %s", e)
            self._ips = []
        self._refresh_dns_locked(time.time())

    def _refresh_dns_locked(self, now: float) -> None:
        self._last_dns = now
        try:
            infos = socket.getaddrinfo(
                self.host, self.port, socket.AF_INET, socket.SOCK_STREAM
            )
        except OSError as e:
            logger.debug("DNS refresh for %s failed: %s", self.host, e)
            return
        fresh: list[str] = []
        for info in infos:
            ip = info[4][0]
            if ip not in fresh:
                fresh.append(ip)
        # Keep the existing (best-first, failure-adapted) order, append new DNS
        # answers at the end. Known-good seeds that DNS no longer returns stay
        # in the pool — they may still be reachable and are the whole point of
        # the seed mechanism.
        merged = list(self._ips)
        merged += [ip for ip in fresh if ip not in merged]
        if len(merged) > 10:
            logger.debug("IP pool for %s trimmed: %s", self.host, merged)
        self._ips = merged[:10]
        self._persist_locked()

    def _persist_locked(self) -> None:
        now = time.time()
        if now - self._last_persist < _PERSIST_MIN_INTERVAL:
            return
        self._last_persist = now
        try:
            _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = (
                json.loads(_PERSIST_PATH.read_text())
                if _PERSIST_PATH.exists()
                else {}
            )
            data[self.host] = self._ips[:5]
            _PERSIST_PATH.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug("IP seed persist failed: %s", e)


_pools: dict[str, _IPPool] = {}
_pools_lock = threading.Lock()

_HL_HOSTS = ("api.hyperliquid.xyz", "api.hyperliquid-testnet.xyz")


def _get_pool(host: str) -> _IPPool | None:
    if host not in _HL_HOSTS:
        return None
    with _pools_lock:
        pool = _pools.get(host)
        if pool is None:
            pool = _IPPool(host)
            _pools[host] = pool
        return pool


def _wrap_session(session: Any) -> Any:
    """Pin requests to working edge IPs via CURLOPT_RESOLVE with failover.

    Wraps Session.post/get (the Hyperliquid SDK only calls post, but get is
    wrapped for safety). Only hosts in _HL_HOSTS are managed; all other
    requests pass through untouched.
    """
    try:
        from curl_cffi.curl import CurlOpt
    except ImportError:
        return session

    orig_post = session.post
    orig_get = session.get

    def _failover_call(orig: Callable, url: str, *args: Any, **kwargs: Any) -> Any:
        host = urlparse(url).hostname
        if not host:
            return orig(url, *args, **kwargs)
        pool = _get_pool(host)
        if pool is None or not pool.has_candidates():
            return orig(url, *args, **kwargs)

        seen: set[str] = set()
        last_exc: Exception | None = None
        while True:
            ip = pool.pick(exclude=seen)
            if ip is None:
                break
            seen.add(ip)
            try:
                session.curl.setopt(CurlOpt.RESOLVE, [f"{host}:443:{ip}"])
            except Exception:
                # Can't pin — fall back to system resolution
                return orig(url, *args, **kwargs)
            try:
                resp = orig(url, *args, **kwargs)
                pool.mark_success(ip)
                return resp
            except _NETWORK_ERRORS as e:
                last_exc = e
                pool.mark_failure(ip)
                logger.warning(
                    "HL failover: %s -> %s failed (%s), trying next IP",
                    host, ip, type(e).__name__,
                )
        if last_exc is not None:
            raise last_exc
        return orig(url, *args, **kwargs)

    session.post = lambda url, *a, **kw: _failover_call(orig_post, url, *a, **kw)
    session.get = lambda url, *a, **kw: _failover_call(orig_get, url, *a, **kw)
    return session


def _patch_hyperliquid_sdk() -> None:
    """Monkey-patch Hyperliquid SDK's requests module with curl_cffi + Firefox."""
    global _cf_requests, _NETWORK_ERRORS
    if _cf_requests is not None:
        return  # already patched

    try:
        from curl_cffi import requests as _cf  # noqa: F811
        from curl_cffi.requests.exceptions import (
            SSLError,
            ConnectionError as _CurlConnectionError,
            Timeout as _CurlTimeout,
        )
        _cf_requests = _cf
        _NETWORK_ERRORS = (SSLError, _CurlConnectionError, _CurlTimeout)
    except ImportError:
        logger.info("curl_cffi not available — TLS fingerprinting may occur")
        return

    # Override Session() to always impersonate Firefox's TLS fingerprint
    # and add the multi-IP failover wrapper.
    _OriginalSession = _cf_requests.Session

    def _make_session(**kw):
        session = _OriginalSession(impersonate="firefox147", timeout=30, **kw)
        return _wrap_session(session)

    _cf_requests.Session = _make_session  # type: ignore[assignment]

    import hyperliquid.api as _hl_api
    import hyperliquid.exchange as _hl_exchange
    import hyperliquid.info as _hl_info
    _hl_api.requests = _cf_requests
    _hl_exchange.requests = _cf_requests
    _hl_info.requests = _cf_requests

    logger.info("Hyperliquid SDK patched with curl_cffi (Firefox TLS fingerprint + IP failover)")


# Patch on import
_patch_hyperliquid_sdk()
