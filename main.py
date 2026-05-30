"""OpenAI-compatible proxy for OpenRouter free models.

Features
--------
* Periodically (every 5 minutes) fetches GET https://openrouter.ai/api/v1/models
  and keeps free models in memory.
* Exposes ``/v1/models`` and ``/v1/chat/completions`` (+ ``/v1/completions``)
  in the OpenAI API shape, so any OpenAI SDK can talk to this server.
* Reads API keys from ``key.txt`` (one per line). If the file is missing it
  is created on startup. Keys are used in round-robin fashion; upstream HTTP
  429 responses are returned to the caller without retrying another key.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, AsyncIterator, Iterable

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()


def _parse_api_keys(value: str) -> set[str]:
    return {item.strip() for item in value.replace("\n", ",").split(",") if item.strip()}


OPENROUTER_BASE = os.environ.get("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
KEY_FILE = Path(os.environ.get("OPENROUTER_KEY_FILE", "key.txt"))
REFRESH_INTERVAL = int(os.environ.get("MODEL_REFRESH_INTERVAL", "300"))  # 5 min
REQUEST_TIMEOUT = float(os.environ.get("UPSTREAM_TIMEOUT", "300"))
HTTP_REFERER = os.environ.get("OPENROUTER_REFERER", "https://github.com/XxxXteam/openrouter-auto")
X_TITLE = os.environ.get("OPENROUTER_TITLE", "openrouter-auto")
CLIENT_API_KEYS = (
    _parse_api_keys(os.environ.get("OPENAI_API_KEY", ""))
    | _parse_api_keys(os.environ.get("OPENAI_API_KEYS", ""))
)
FREE_MODEL_IDS = {"openrouter/auto"}


def _is_free_model(model: dict[str, Any]) -> bool:
    model_id = str(model.get("id", ""))
    return model_id.endswith(":free") or model_id in FREE_MODEL_IDS


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("openrouter-auto")


def require_client_api_key(authorization: str | None = Header(default=None)) -> None:
    """Validate the caller's OpenAI-compatible Bearer token."""
    if not CLIENT_API_KEYS:
        return

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not any(secrets.compare_digest(token, key) for key in CLIENT_API_KEYS):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Key manager
# ---------------------------------------------------------------------------
class KeyManager:
    """Round-robin pool of OpenRouter API keys."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()
        self._keys: list[str] = []
        self._cycle: itertools.cycle | None = None
        self.reload()

    # -- file management -------------------------------------------------
    def reload(self) -> None:
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")
            log.warning("key file %s not found, created an empty one", self.path)
        keys = [
            line.strip()
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        with self._lock:
            self._keys = keys
            self._cycle = itertools.cycle(keys) if keys else None
        log.info("loaded %d API key(s) from %s", len(keys), self.path)

    # -- selection -------------------------------------------------------
    def acquire(self) -> str | None:
        """Return the next key in round-robin order."""
        with self._lock:
            if not self._keys:
                return None
            return next(self._cycle)  # type: ignore[arg-type]

    @property
    def has_keys(self) -> bool:
        with self._lock:
            return bool(self._keys)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._keys)


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
class ModelCache:
    """Stores the latest list of free OpenRouter models."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._raw: list[dict[str, Any]] = []
        self._updated_at: float = 0.0

    async def set(self, raw_models: list[dict[str, Any]]) -> None:
        free = [m for m in raw_models if _is_free_model(m)]
        async with self._lock:
            self._raw = free
            self._updated_at = time.time()
        log.info("model cache refreshed: %d free models (of %d total)",
                 len(free), len(raw_models))

    async def get_raw(self) -> list[dict[str, Any]]:
        async with self._lock:
            return list(self._raw)

    async def to_openai(self) -> dict[str, Any]:
        async with self._lock:
            data = [
                {
                    "id": m["id"],
                    "object": "model",
                    "created": m.get("created") or int(self._updated_at),
                    "owned_by": m["id"].split("/", 1)[0] if "/" in m["id"] else "openrouter",
                    # extra info – clients that follow OpenAI's spec ignore unknown fields
                    "name": m.get("name"),
                    "description": m.get("description"),
                    "context_length": m.get("context_length"),
                    "pricing": m.get("pricing"),
                    "architecture": m.get("architecture"),
                    "supported_parameters": m.get("supported_parameters"),
                }
                for m in self._raw
            ]
        return {"object": "list", "data": data}


model_cache = ModelCache()
key_manager = KeyManager(KEY_FILE)


# ---------------------------------------------------------------------------
# Background refresher
# ---------------------------------------------------------------------------
async def fetch_models_once(client: httpx.AsyncClient) -> None:
    try:
        resp = await client.get(f"{OPENROUTER_BASE}/models", timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        await model_cache.set(payload.get("data", []))
    except Exception as exc:  # noqa: BLE001
        log.error("failed to refresh models: %s", exc)


async def model_refresher() -> None:
    async with httpx.AsyncClient() as client:
        while True:
            await fetch_models_once(client)
            await asyncio.sleep(REFRESH_INTERVAL)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(model_refresher())
    log.info("startup: model refresher started, key pool=%d", key_manager.size)
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="openrouter-auto", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers for upstream forwarding
# ---------------------------------------------------------------------------
def _build_upstream_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": HTTP_REFERER,
        "X-Title": X_TITLE,
    }


async def _forward_non_stream(path: str, body: dict[str, Any]) -> JSONResponse:
    if not key_manager.has_keys:
        raise HTTPException(503, "No API keys configured. Add keys to key.txt and reload.")

    tried: set[str] = set()
    last_error: tuple[int, str] | None = None
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        # try each key at most once
        for _ in range(max(1, key_manager.size)):
            key = key_manager.acquire()
            if key is None or key in tried:
                break
            tried.add(key)
            try:
                resp = await client.post(
                    f"{OPENROUTER_BASE}{path}",
                    headers=_build_upstream_headers(key),
                    json=body,
                )
            except httpx.HTTPError as exc:
                last_error = (502, f"upstream connection error: {exc}")
                continue

            return JSONResponse(
                status_code=resp.status_code,
                content=_safe_json(resp),
            )

    code, msg = last_error or (502, "no response from upstream")
    return JSONResponse(status_code=code, content={"error": {"message": msg, "type": "upstream_error"}})


def _safe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return {"error": {"message": resp.text, "type": "upstream_error"}}


async def _forward_stream(path: str, body: dict[str, Any]) -> StreamingResponse:
    if not key_manager.has_keys:
        raise HTTPException(503, "No API keys configured. Add keys to key.txt and reload.")

    async def event_source() -> AsyncIterator[bytes]:
        tried: set[str] = set()
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            for _ in range(max(1, key_manager.size)):
                key = key_manager.acquire()
                if key is None or key in tried:
                    break
                tried.add(key)
                try:
                    async with client.stream(
                        "POST",
                        f"{OPENROUTER_BASE}{path}",
                        headers=_build_upstream_headers(key),
                        json=body,
                    ) as resp:
                        if resp.status_code >= 400:
                            text = (await resp.aread()).decode("utf-8", errors="replace")
                            yield _sse_error(resp.status_code, text)
                            return
                        async for chunk in resp.aiter_bytes():
                            if chunk:
                                yield chunk
                        return
                except httpx.HTTPError as exc:
                    yield _sse_error(502, f"upstream connection error: {exc}")
                    return
            yield _sse_error(502, "no response from upstream")

    return StreamingResponse(event_source(), media_type="text/event-stream")


def _sse_error(status: int, message: str) -> bytes:
    payload = {"error": {"message": message, "type": "upstream_error", "code": status}}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", dependencies=[Depends(require_client_api_key)])
async def root() -> dict[str, Any]:
    return {
        "service": "openrouter-auto",
        "models_cached": len(await model_cache.get_raw()),
        "keys_loaded": key_manager.size,
        "refresh_interval_sec": REFRESH_INTERVAL,
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/admin/reload-keys", dependencies=[Depends(require_client_api_key)])
async def reload_keys() -> dict[str, Any]:
    key_manager.reload()
    return {"keys_loaded": key_manager.size}


@app.post("/admin/refresh-models", dependencies=[Depends(require_client_api_key)])
async def refresh_models() -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        await fetch_models_once(client)
    return {"models_cached": len(await model_cache.get_raw())}


@app.get("/v1/models", dependencies=[Depends(require_client_api_key)])
async def list_models() -> dict[str, Any]:
    return await model_cache.to_openai()


@app.get("/v1/models/{model_id:path}", dependencies=[Depends(require_client_api_key)])
async def retrieve_model(model_id: str) -> dict[str, Any]:
    payload = await model_cache.to_openai()
    for m in payload["data"]:
        if m["id"] == model_id:
            return m
    raise HTTPException(404, f"model '{model_id}' not found in free model cache")


async def _proxy_openai(request: Request, upstream_path: str):
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"invalid JSON body: {exc}")

    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be a JSON object")

    if body.get("stream"):
        return await _forward_stream(upstream_path, body)
    return await _forward_non_stream(upstream_path, body)


@app.post("/v1/chat/completions", dependencies=[Depends(require_client_api_key)])
async def chat_completions(request: Request):
    return await _proxy_openai(request, "/chat/completions")


@app.post("/v1/completions", dependencies=[Depends(require_client_api_key)])
async def completions(request: Request):
    return await _proxy_openai(request, "/completions")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
