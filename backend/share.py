"""
Sharing foundation for LazySD UI — Gradio's ``share=True`` idea, without Gradio.

Three layers, each usable on its own:

1. **Access control** (`ShareAccessMiddleware`): anything that is not plain
   loopback traffic is *remote*. Remote visitors need the share token and only
   get a small "guest" API surface (generate / browse), never settings, engine
   installs, model registration, file import, etc. (unless `share_allow_admin`).

   Tunnels (cloudflared, ngrok, ...) connect to us from 127.0.0.1, so the
   socket peer alone says nothing. A request that carries proxy headers
   (X-Forwarded-For, CF-Connecting-IP, ...) is therefore treated as remote.

2. **Tunnel providers** (`TunnelProvider` + `PROVIDERS`): pluggable. Only
   cloudflared quick-tunnels (no account needed) are implemented; adding another
   provider = subclass + register in `PROVIDERS`.

3. **Manager + routes** (`ShareManager`, `register_share_routes`): start/stop
   the tunnel, token generation, LAN URL discovery, status for the UI.
"""

from __future__ import annotations

import atexit
import collections
import hmac
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, urlencode

from .settings import IS_WINDOWS, get_settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COOKIE_NAME = "lazysd_token"
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}  # 'testclient' = Starlette TestClient

# Headers that only a reverse proxy / tunnel adds. Their presence => not a direct local user.
PROXY_HEADERS = (
    b"forwarded", b"x-forwarded-for", b"x-real-ip", b"cf-connecting-ip", b"cf-ray",
    b"x-forwarded-host", b"x-forwarded-proto", b"true-client-ip",
)


# ============================================================
# 1. Access control
# ============================================================
# Guest allow-list: (method, compiled regex on path). Everything else is admin-only.
_GUEST_ALLOW: list[tuple[str, re.Pattern]] = [
    ("GET", re.compile(r"^/(index\.html)?$")),
    ("GET", re.compile(r"^/(css|js|locale|outputs)/")),
    ("GET", re.compile(r"^/api/(config|models|presets|loras|history|system/stats|system/open-folder|open-folder)$")),
    ("GET", re.compile(r"^/api/models/[^/]+/state$")),
    ("GET", re.compile(r"^/api/jobs/[^/]+$")),
    ("POST", re.compile(r"^/api/(generate|upload|system/open-folder|open-folder)$")),
    ("POST", re.compile(r"^/api/history/[^/]+/share$")),
    ("POST", re.compile(r"^/api/jobs/[^/]+/cancel$")),
]


def is_guest_allowed(method: str, path: str) -> bool:
    method = method.upper()
    if method == "HEAD":
        method = "GET"
    return any(m == method and rx.search(path) for m, rx in _GUEST_ALLOW)


def is_remote_scope(scope) -> bool:
    client = scope.get("client")
    host = client[0] if client else ""
    if host not in LOOPBACK_HOSTS:
        return True
    for name, _ in scope.get("headers", []):
        if name.lower() in PROXY_HEADERS:
            return True
    return False


def _headers(scope) -> dict[bytes, bytes]:
    return {k.lower(): v for k, v in scope.get("headers", [])}


def extract_token(scope) -> tuple[Optional[str], str]:
    """Return (token, source) where source in {'header','cookie','query',''}."""
    h = _headers(scope)
    auth = h.get(b"authorization", b"").decode("latin-1")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip(), "header"
    x = h.get(b"x-access-token", b"").decode("latin-1").strip()
    if x:
        return x, "header"
    ck = h.get(b"cookie", b"").decode("latin-1")
    if ck:
        try:
            c = SimpleCookie()
            c.load(ck)
            if COOKIE_NAME in c:
                return c[COOKIE_NAME].value, "cookie"
        except Exception:
            pass
    qs = parse_qs(scope.get("query_string", b"").decode("latin-1"))
    if qs.get("token"):
        return qs["token"][0], "query"
    return None, ""


def token_ok(provided: Optional[str], expected: Optional[str]) -> bool:
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided.encode(), expected.encode())


_LOGIN_HTML = """<!doctype html><meta charset=utf-8><title>LazySD UI</title>
<body style="font-family:system-ui;background:#0f0f12;color:#e4e4e7;display:grid;place-items:center;height:100vh;margin:0">
<form onsubmit="location.href='/?token='+encodeURIComponent(t.value);return false"
 style="background:#1f1f23;padding:28px 32px;border-radius:14px;border:1px solid #333;width:320px">
<h3 style="margin:0 0 6px">LazySD UI</h3>
<p style="margin:0 0 14px;color:#a1a1aa;font-size:13px">Remote access token required</p>
<input id=t type=password autofocus placeholder="Access token"
 style="width:100%;box-sizing:border-box;padding:10px;border-radius:8px;border:1px solid #444;background:#131316;color:#fff">
<button style="margin-top:12px;width:100%;padding:10px;border:0;border-radius:8px;background:#7c3aed;color:#fff;font-weight:600">Enter</button>
</form></body>"""


async def _respond(send, status: int, body: bytes, content_type: str, extra_headers: list | None = None):
    headers = [(b"content-type", content_type.encode()), (b"content-length", str(len(body)).encode()),
               (b"cache-control", b"no-store")]
    headers += extra_headers or []
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class ShareAccessMiddleware:
    """Pure-ASGI middleware (no BaseHTTPMiddleware => no streaming/StaticFiles quirks)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or not is_remote_scope(scope):
            scope.setdefault("state", {})["account"] = {
                "role": "host",
                "id": "host",
                "username": "Хост",
                "is_guest": False,
            }
            return await self.app(scope, receive, send)

        if scope["type"] == "websocket":
            # No websockets are used by the UI; refuse for remote peers.
            await send({"type": "websocket.close", "code": 1008})
            return

        settings = get_settings()
        expected = settings.share_token
        method = scope.get("method", "GET")
        path = scope.get("path", "/")
        wants_html = b"text/html" in _headers(scope).get(b"accept", b"")

        if not expected:
            return await _respond(
                send, 403,
                json.dumps({"detail": "Remote access is disabled. Enable sharing on the host machine."}).encode(),
                "application/json",
            )

        from .accounts import get_accounts_manager
        acc_mgr = get_accounts_manager()
        provided, source = extract_token(scope)

        matched_guest = acc_mgr.find_by_token(provided) if provided else None
        is_master = token_ok(provided, expected)

        if not is_master and not matched_guest:
            if method == "GET" and wants_html and path in ("/", "/index.html"):
                return await _respond(send, 401, _LOGIN_HTML.encode(), "text/html; charset=utf-8")
            return await _respond(send, 401, json.dumps({"detail": "Invalid or missing access token"}).encode(),
                                  "application/json")

        # Valid token passed via ?token= -> set cookie, strip it from the URL.
        if source == "query" and method == "GET":
            qs = parse_qs(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
            qs.pop("token", None)
            location = path + (("?" + urlencode(qs, doseq=True)) if qs else "")
            https = _headers(scope).get(b"x-forwarded-proto", b"").lower() == b"https"
            cookie = f"{COOKIE_NAME}={provided}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000"
            if https:
                cookie += "; Secure"
            return await _respond(send, 302, b"", "text/plain",
                                  [(b"location", location.encode()), (b"set-cookie", cookie.encode())])

        # Resolve caller role
        if is_master and settings.share_allow_admin:
            caller = {
                "role": "host",
                "id": "host_remote",
                "username": "Хост (Удалённый)",
                "is_guest": False,
            }
        elif matched_guest:
            caller = {
                "role": "guest",
                "id": matched_guest["id"],
                "username": matched_guest.get("username", "Гость"),
                "is_guest": True,
            }
        else:
            caller = {
                "role": "guest",
                "id": "guest_public",
                "username": "Гость",
                "is_guest": True,
            }

        scope.setdefault("state", {})["account"] = caller
        scope["state"]["is_guest"] = caller["is_guest"]

        if caller["is_guest"] and not is_guest_allowed(method, path):
            return await _respond(
                send, 403,
                json.dumps({"detail": "This action is only available on the host machine (guest access)."}).encode(),
                "application/json",
            )

        return await self.app(scope, receive, send)


# ============================================================
# 2. Tunnel providers
# ============================================================
def _bin_dir() -> Path:
    d = PROJECT_ROOT / "bin" / ("windows" if IS_WINDOWS else "linux")
    d.mkdir(parents=True, exist_ok=True)
    return d


class TunnelProvider:
    """Base class. Subclass, set `id`/`label`, implement the 4 methods, add to PROVIDERS."""

    id = "base"
    label = "Base"

    def executable(self) -> Optional[str]:
        raise NotImplementedError

    def install(self, progress: Optional[Callable[[str], None]] = None) -> str:
        raise NotImplementedError

    def command(self, exe: str, port: int) -> list[str]:
        raise NotImplementedError

    def parse_url(self, line: str) -> Optional[str]:
        raise NotImplementedError


class CloudflaredProvider(TunnelProvider):
    """Cloudflare quick tunnel: free, no account, random *.trycloudflare.com URL."""

    id = "cloudflared"
    label = "Cloudflare Quick Tunnel"
    _URL_RE = re.compile(r"https://[a-z0-9\-]+\.trycloudflare\.com")
    _ASSET = {
        "win32": "cloudflared-windows-amd64.exe",
        "linux": "cloudflared-linux-amd64",
    }

    def _local_path(self) -> Path:
        return _bin_dir() / ("cloudflared.exe" if IS_WINDOWS else "cloudflared")

    def executable(self) -> Optional[str]:
        local = self._local_path()
        if local.is_file():
            return str(local)
        return shutil.which("cloudflared")

    def install(self, progress=None) -> str:
        import requests

        asset = self._ASSET.get("win32" if IS_WINDOWS else sys.platform if sys.platform.startswith("linux") else "")
        if not asset:
            raise RuntimeError("Automatic cloudflared install supports Windows and Linux only. "
                               "Install it manually and make sure `cloudflared` is on PATH.")
        url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/{asset}"
        dest = self._local_path()
        tmp = dest.with_suffix(dest.suffix + ".part")
        if progress:
            progress(f"Downloading {url}")
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1024 * 256):
                        f.write(chunk)
            if tmp.stat().st_size < 5 * 1024 * 1024:
                raise RuntimeError("Downloaded cloudflared looks truncated")
            tmp.replace(dest)
            if not IS_WINDOWS:
                dest.chmod(0o755)
        finally:
            tmp.unlink(missing_ok=True)
        return str(dest)

    def command(self, exe: str, port: int) -> list[str]:
        return [exe, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"]

    def parse_url(self, line: str) -> Optional[str]:
        m = self._URL_RE.search(line)
        return m.group(0) if m else None


PROVIDERS: dict[str, type[TunnelProvider]] = {
    CloudflaredProvider.id: CloudflaredProvider,
    # Future: "ngrok": NgrokProvider, "localtunnel": ..., "ssh": ...
}


# ============================================================
# 3. Manager
# ============================================================
_BIND = {"host": "127.0.0.1", "port": 7860}


def set_bind(host: str, port: int) -> None:
    _BIND.update(host=host, port=int(port))


def is_lan_exposed() -> bool:
    return _BIND["host"] not in ("127.0.0.1", "localhost", "::1")


def lan_ips() -> list[str]:
    ips: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is actually sent
            ips.append(s.getsockname()[0])
    except Exception:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    return [ip for ip in ips if not ip.startswith("127.")]


def ensure_token(regenerate: bool = False) -> str:
    settings = get_settings()
    if regenerate or not settings.share_token:
        settings.update(share_token=secrets.token_urlsafe(24))
    return settings.share_token


class ShareManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._state = "stopped"          # stopped | starting | running | error
        self._url: Optional[str] = None
        self._error: Optional[str] = None
        self._provider_id: Optional[str] = None
        self._log: collections.deque = collections.deque(maxlen=60)
        self._started_at: Optional[float] = None
        atexit.register(self.stop)

    # ---- provider helpers
    def _provider(self, provider_id: Optional[str] = None) -> TunnelProvider:
        pid = provider_id or get_settings().share_tunnel_provider
        cls = PROVIDERS.get(pid)
        if not cls:
            raise ValueError(f"Unknown tunnel provider: {pid}. Available: {', '.join(PROVIDERS)}")
        return cls()

    # ---- lifecycle
    def start(self, port: Optional[int] = None) -> dict:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                return self.status()
            provider = self._provider()
            exe = provider.executable()
            if not exe:
                try:
                    self._log.append(f"Auto-downloading {provider.label} binary from official release...")
                    exe = provider.install(progress=lambda msg: self._log.append(msg))
                except Exception as e:
                    raise RuntimeError(f"Не удалось автоматически загрузить {provider.label}: {e}")
            ensure_token()
            port = int(port or _BIND["port"])
            self._state, self._url, self._error = "starting", None, None
            self._provider_id = provider.id
            self._log.clear()
            self._started_at = time.time()
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if IS_WINDOWS else 0
            self._proc = subprocess.Popen(
                provider.command(exe, port),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=flags,
            )
            threading.Thread(target=self._reader, args=(self._proc, provider), daemon=True).start()
        return self.status()

    def _reader(self, proc: subprocess.Popen, provider: TunnelProvider) -> None:
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                line = line.rstrip()
                if not line:
                    continue
                self._log.append(line)
                url = provider.parse_url(line)
                if url and self._state != "running":
                    with self._lock:
                        self._url, self._state = url, "running"
        finally:
            code = proc.wait()
            with self._lock:
                if self._proc is proc:
                    if self._state != "stopped":
                        self._state = "error" if self._state == "starting" or code not in (0, None) else "stopped"
                        if self._state == "error":
                            self._error = f"Tunnel process exited (code {code}). " + (self._log[-1] if self._log else "")
                    self._url = None
                    self._proc = None

    def stop(self) -> dict:
        with self._lock:
            proc, self._proc = self._proc, None
            self._state, self._url, self._error = "stopped", None, None
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=4)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        return self.status()

    # ---- status (admin only: contains the token)
    def status(self) -> dict:
        settings = get_settings()
        token = settings.share_token
        provider = self._provider()
        exe = provider.executable()
        port = _BIND["port"]

        def with_token(base: str) -> str:
            return f"{base}/?token={token}" if token else base

        lan = []
        if is_lan_exposed():
            host = _BIND["host"]
            ips = lan_ips() if host in ("0.0.0.0", "::") else [host]
            lan = [with_token(f"http://{ip}:{port}") for ip in ips]
        share_url = with_token(self._url) if self._url else None
        return {
            "running": self._state == "running",
            "public_url": self._url,
            "share_url": share_url,
            "bind": {"host": _BIND["host"], "port": port, "lan_exposed": is_lan_exposed(), "lan_urls": lan},
            "token_set": bool(token),
            "token": token,
            "allow_admin": bool(settings.share_allow_admin),
            "auto_start": bool(settings.share_auto_start),
            "providers": [{"id": k, "label": v.label} for k, v in PROVIDERS.items()],
            "tunnel": {
                "provider": self._provider_id or provider.id,
                "provider_label": provider.label,
                "installed": bool(exe),
                "state": self._state,
                "url": self._url,
                "share_url": share_url,
                "error": self._error,
                "log": list(self._log)[-8:],
                "uptime_sec": int(time.time() - self._started_at) if self._started_at and self._state == "running" else 0,
            },
        }


_MANAGER: Optional[ShareManager] = None


def get_share_manager() -> ShareManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = ShareManager()
    return _MANAGER


# ============================================================
# Routes
# ============================================================
def register_share_routes(app) -> None:
    import asyncio

    from fastapi import HTTPException

    mgr = get_share_manager()

    @app.get("/api/share/status")
    async def share_status():
        return mgr.status()

    @app.post("/api/share/token")
    async def share_token(body: dict):
        """Body: {"action": "generate"|"set"|"clear", "token": "..."}"""
        action = (body or {}).get("action", "generate")
        settings = get_settings()
        if action == "generate":
            ensure_token(regenerate=True)
        elif action == "set":
            tok = str((body or {}).get("token", "")).strip()
            if len(tok) < 8 or not re.fullmatch(r"[A-Za-z0-9_\-.~]+", tok):
                raise HTTPException(400, "Token must be >= 8 chars of [A-Za-z0-9_-.~]")
            settings.update(share_token=tok)
        elif action == "clear":
            mgr.stop()
            settings.update(share_token=None)
        else:
            raise HTTPException(400, "action must be generate | set | clear")
        return mgr.status()

    @app.post("/api/share/options")
    async def share_options(body: dict):
        allowed = {"share_auto_start", "share_allow_admin", "share_tunnel_provider"}
        upd = {k: v for k, v in (body or {}).items() if k in allowed}
        if "share_tunnel_provider" in upd and upd["share_tunnel_provider"] not in PROVIDERS:
            raise HTTPException(400, "Unknown tunnel provider")
        if "share_auto_start" in upd:
            upd["share_auto_start"] = bool(upd["share_auto_start"])
        if "share_allow_admin" in upd:
            upd["share_allow_admin"] = bool(upd["share_allow_admin"])
        if upd:
            get_settings().update(**upd)
        return mgr.status()

    @app.post("/api/share/install")
    async def share_install():
        try:
            await asyncio.to_thread(mgr._provider().install)
        except Exception as e:
            raise HTTPException(500, f"{type(e).__name__}: {e}")
        return mgr.status()

    @app.post("/api/share/start")
    async def share_start():
        try:
            return await asyncio.to_thread(mgr.start)
        except Exception as e:
            raise HTTPException(400, f"{type(e).__name__}: {e}")

    @app.post("/api/share/stop")
    async def share_stop():
        return await asyncio.to_thread(mgr.stop)
