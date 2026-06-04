"""
`admin` is providing the ``/gate/*`` namespace for InferenceGate's own control
endpoints.  This namespace is reserved: requests under ``/gate/`` are never
routed upstream and never recorded as cassettes.  It is the canonical way for
clients (Glue, pytest-xdist controllers, operators) to introspect and reconfigure
a running Gate without restarting it.

Endpoints:

- ``GET  /gate/health``        — liveness + current mode/fuzzy settings.
- ``GET  /gate/config``        — full redacted config snapshot.
- ``POST /gate/config``        — apply runtime overrides to mode/fuzzy/limits.
- ``POST /gate/index/reload``  — re-scan the cassette directory and rebuild index.
- ``GET  /gate/stats``         — counters: cassettes, replies, hits/misses by tier.

Sensitive values (api_key, Authorization header, URL userinfo) are masked via
:func:`redact` before being included in any response.

Key functions: :func:`register_admin_routes`, :func:`redact`.
"""

import asyncio
import hashlib
import json
import logging
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from urllib.parse import urlsplit, urlunsplit

from aiohttp import web

from inference_gate.modes import Mode
from inference_gate.outflow.model_router import EndpointConfig, ModelRoute, OutflowRouter
from inference_gate.router.router import Router

log = logging.getLogger("InflowServer.admin")


class ConfigGate:
    """
    Coordinates atomic ``POST /gate/config`` updates with in-flight proxy requests.

    Two invariants are enforced:

    1. **Serialised updates** — at most one config update runs at a time.  Multiple
       concurrent POSTs (e.g. from N xdist workers spinning up in parallel) queue
       behind ``_update_lock``.
    2. **Quiescent rebuild** — while an update is performing the
       validate→start-new→swap→stop-old dance on the router's outflow component,
       no proxy request is allowed to enter ``Router.route_request``.  In-flight
       requests that started before the update drain to zero before the swap
       proceeds.  New proxy requests block on ``_open_event`` until the update
       releases.

    A separate ``_admission_lock`` guards the open-flag + in-flight counter
    transitions so proxy entries cannot interleave with the updater's "close
    gate" step.

    ``last_payload_hash`` is the SHA-256 (truncated) of the canonicalised body of
    the last successfully-applied POST.  Repeated identical POSTs short-circuit
    without taking the update lock or quiescing requests.
    """

    def __init__(self) -> None:
        # Serialise concurrent POST /gate/config calls.
        self._update_lock = asyncio.Lock()
        # Guard transitions of ``_open`` + ``_inflight`` so proxy admission and
        # the updater's close-gate step cannot interleave.
        self._admission_lock = asyncio.Lock()
        self._open = True
        self._open_event = asyncio.Event()
        self._open_event.set()
        self._inflight = 0
        self._inflight_zero = asyncio.Event()
        self._inflight_zero.set()
        self.last_payload_hash: str | None = None

    @asynccontextmanager
    async def request_slot(self) -> AsyncIterator[None]:
        """
        Async context manager held by every proxy request for the duration of
        :meth:`Router.route_request`.

        Blocks until the gate is open, then increments the in-flight counter
        atomically.  On exit, decrements the counter and (if it reaches zero)
        signals the drain Event so a waiting updater can proceed.
        """
        # Loop in case a config update closes the gate between the wait and the
        # admission-lock acquisition.
        while True:
            async with self._admission_lock:
                if self._open:
                    self._inflight += 1
                    self._inflight_zero.clear()
                    break
            # Gate was closed under us; wait for it to reopen and retry.
            await self._open_event.wait()
        try:
            yield
        finally:
            async with self._admission_lock:
                self._inflight -= 1
                if self._inflight == 0:
                    self._inflight_zero.set()

    @asynccontextmanager
    async def exclusive_update(self) -> AsyncIterator[None]:
        """
        Async context manager held by ``POST /gate/config`` while applying a
        non-idempotent change to router state.

        Acquires the update lock (serialising concurrent updates), closes the
        admission gate so no new proxy request can enter ``route_request``,
        waits for currently in-flight requests to drain, then yields control
        to the caller.  On exit the gate is reopened so blocked proxy requests
        can proceed.
        """
        async with self._update_lock:
            async with self._admission_lock:
                self._open = False
                self._open_event.clear()
            try:
                # Wait for in-flight proxy requests to finish before mutating
                # router.outflow / starting/stopping the underlying client.
                await self._inflight_zero.wait()
                yield
            finally:
                async with self._admission_lock:
                    self._open = True
                    self._open_event.set()


def _hash_payload(body: dict[str, Any]) -> str:
    """
    Compute a stable SHA-256 fingerprint (16 hex chars) of a JSON-serialisable
    config payload.  Dict keys are sorted recursively so logically-identical
    payloads always produce the same hash regardless of input ordering.
    """
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

REDACTION_PLACEHOLDER = "***"

_FUZZY_SAMPLING_VALUES = ("off", "soft", "aggressive")


def _redact_url(url: str) -> str:
    """
    Return ``url`` with any embedded ``user:password@`` userinfo replaced by ``***@``.
    Non-URL strings are returned unchanged.
    """
    if not url or "://" not in url:
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.username and not parts.password:
        return url
    netloc = parts.hostname or ""
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    netloc = f"{REDACTION_PLACEHOLDER}@{netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _redact_endpoint(name: str, cfg: EndpointConfig) -> dict[str, Any]:
    """
    Render an :class:`EndpointConfig` as a JSON-safe dict with secrets removed.
    """
    return {
        "name": name,
        "url": _redact_url(cfg.url),
        "api_key": REDACTION_PLACEHOLDER if cfg.api_key else None,
        "timeout": cfg.timeout,
        "proxy": _redact_url(cfg.proxy) if cfg.proxy else None,
        "verify_ssl": cfg.verify_ssl,
    }


def _redact_outflow(outflow: OutflowRouter | None) -> dict[str, Any] | None:
    """
    Render the router's outflow component (``OutflowRouter``) as a JSON-safe
    dict with secrets redacted.

    Returns ``None`` when ``outflow is None`` (replay-only mode).  The returned
    dict has shape ``{"type": "router", "endpoints": {...}, "routes": [...]}``.
    """
    if outflow is None:
        return None
    return {
        "type": "router",
        "endpoints": {name: _redact_endpoint(name, cfg) for name, cfg in outflow.endpoints.items()},
        "routes": [{"pattern": route.pattern, "endpoint": route.endpoint_name} for route in outflow.routes],
    }


_HEADER_REDACT_PATTERN = re.compile(r"^(authorization|proxy-authorization|x-api-key)$", re.IGNORECASE)


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """
    Return a copy of ``headers`` with auth-bearing fields masked.

    Useful for any future endpoint that echoes back request metadata.
    """
    out: dict[str, str] = {}
    for name, value in headers.items():
        if _HEADER_REDACT_PATTERN.match(name):
            out[name] = REDACTION_PLACEHOLDER
        else:
            out[name] = value
    return out


def redact(value: Any) -> Any:
    """
    Best-effort recursive redaction for arbitrary JSON-serialisable structures.

    Strings whose key suggests an API key, password, or token are masked; all
    other scalars pass through.  This is the public helper for endpoints that
    need to dump partially-known payloads.
    """
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, sub in value.items():
            if isinstance(key, str) and re.search(r"(api[_-]?key|password|secret|token|authorization)", key, re.IGNORECASE):
                out[key] = REDACTION_PLACEHOLDER if sub else sub
            else:
                out[key] = redact(sub)
        return out
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def _config_snapshot(router: Router) -> dict[str, Any]:
    """
    Build the JSON payload for ``GET /gate/config`` from the live :class:`Router`.
    """
    storage = router.storage
    return {
        "mode": router.mode.value,
        "fuzzy_model": router.fuzzy_model,
        "fuzzy_sampling": router.fuzzy_sampling,
        "max_non_greedy_replies": router.max_non_greedy_replies,
        "non_streaming_models": list(router.non_streaming_models),
        "cache_dir": str(storage.cache_dir),
        "outflow": _redact_outflow(router.outflow),
        "stats": {
            "cassettes": len(storage.index.by_content_hash),
        },
    }


async def _handle_health(request: web.Request) -> web.Response:
    """
    Return a small liveness payload with the current mode and fuzzy settings.

    Mirrors ``GET /health`` (kept as alias for backward compatibility) but
    located in the reserved ``/gate/`` namespace so future probes can rely on
    the namespace contract.
    """
    router: Router = request.app["router"]
    return web.json_response({
        "status": "healthy",
        "mode": router.mode.value,
        "fuzzy_model": router.fuzzy_model,
        "fuzzy_sampling": router.fuzzy_sampling,
    })


async def _handle_get_config(request: web.Request) -> web.Response:
    """
    Return a full, redacted snapshot of the running Gate's config.
    """
    router: Router = request.app["router"]
    return web.json_response(_config_snapshot(router))


def _extract_current_outflow_state(outflow: OutflowRouter | None) -> tuple[dict[str, EndpointConfig], list[ModelRoute]]:
    """
    Return the live ``(endpoints, routes)`` tuple from the running outflow,
    or empty containers when ``outflow`` is ``None`` (replay-only mode).

    Used by :func:`_handle_post_config` to merge a partial payload (e.g. a
    ``mode`` flip without ``endpoints``/``models`` fields) with the current
    routing state.
    """
    if outflow is None:
        return {}, []
    return outflow.endpoints, outflow.routes


def _validate_endpoints_payload(value: Any) -> str | None:
    """
    Validate a JSON ``endpoints`` payload.  Returns ``None`` when valid or an
    error message string when invalid.
    """
    if not isinstance(value, dict):
        return "endpoints must be a JSON object mapping name → endpoint config"
    for name, cfg in value.items():
        if not isinstance(name, str) or not name:
            return "endpoints keys must be non-empty strings"
        if not isinstance(cfg, dict):
            return f"endpoints[{name!r}] must be a JSON object"
        if "url" not in cfg or not isinstance(cfg["url"], str) or not cfg["url"]:
            return f"endpoints[{name!r}] must contain a non-empty string 'url'"
        for opt_key, expected in (("api_key", str), ("proxy", str), ("timeout", (int, float)), ("verify_ssl", bool)):
            if opt_key in cfg and cfg[opt_key] is not None and not isinstance(cfg[opt_key], expected):
                return f"endpoints[{name!r}].{opt_key} must be of type {expected}"
    return None


def _validate_models_payload(value: Any, endpoint_names: set[str]) -> str | None:
    """
    Validate a JSON ``models`` payload.  Returns ``None`` when valid or an
    error message string when invalid.

    Each entry must be a JSON object with string ``pattern`` and either string
    ``endpoint`` (matching one of ``endpoint_names``) or ``endpoint=null``
    (offline sentinel).
    """
    if not isinstance(value, list):
        return "models must be a JSON array of routing rules"
    for idx, route in enumerate(value):
        if not isinstance(route, dict):
            return f"models[{idx}] must be a JSON object"
        pattern = route.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            return f"models[{idx}].pattern must be a non-empty string"
        endpoint = route.get("endpoint")
        if endpoint is not None and not isinstance(endpoint, str):
            return f"models[{idx}].endpoint must be a string or null"
        if isinstance(endpoint, str) and endpoint not in endpoint_names:
            return (f"models[{idx}].endpoint={endpoint!r} does not match any configured endpoint; "
                    f"known endpoints: {sorted(endpoint_names)}")
    return None


def _build_endpoints(payload: dict[str, Any], default_timeout: float) -> dict[str, EndpointConfig]:
    """
    Convert a validated ``endpoints`` payload dict into ``{name: EndpointConfig}``.

    Missing per-endpoint ``timeout`` falls back to ``default_timeout`` (the
    server's record_timeout default).
    """
    out: dict[str, EndpointConfig] = {}
    for name, cfg in payload.items():
        out[name] = EndpointConfig(
            url=cfg["url"],
            api_key=cfg.get("api_key"),
            timeout=float(cfg.get("timeout") if cfg.get("timeout") is not None else default_timeout),
            proxy=cfg.get("proxy"),
            verify_ssl=cfg.get("verify_ssl", True),
        )
    return out


def _build_routes(payload: list[dict[str, Any]]) -> list[ModelRoute]:
    """
    Convert a validated ``models`` payload list into an ordered list of
    :class:`ModelRoute`.
    """
    return [ModelRoute(pattern=route["pattern"], endpoint_name=route.get("endpoint"), order=idx) for idx, route in enumerate(payload)]


def _build_outflow(body: dict[str, Any], current: OutflowRouter | None, mode: Mode, default_timeout: float) -> OutflowRouter | None:
    """
    Construct a fresh :class:`OutflowRouter` reflecting the merged config.

    Merges fields from ``body`` (POST /gate/config payload) over the values
    currently held by ``current``.  When ``mode`` is not RECORD_AND_REPLAY no
    outflow is needed and ``None`` is returned.
    """
    if mode != Mode.RECORD_AND_REPLAY:
        return None
    cur_endpoints, cur_routes = _extract_current_outflow_state(current)
    if "endpoints" in body:
        endpoints = _build_endpoints(body["endpoints"], default_timeout)
    else:
        endpoints = dict(cur_endpoints)
    if "models" in body:
        routes = _build_routes(body["models"])
    else:
        routes = list(cur_routes)
    return OutflowRouter(endpoints=endpoints, routes=routes)


async def _handle_post_config(request: web.Request) -> web.Response:
    """
    Apply a partial runtime override of router/outflow config.

    Accepted JSON body keys:

    - Router-level: ``mode``, ``fuzzy_model``, ``fuzzy_sampling``,
      ``max_non_greedy_replies``, ``non_streaming_models``.
    - Outflow-level: ``endpoints`` (mapping name → ``{url, api_key?, proxy?, timeout?, verify_ssl?}``),
      ``models`` (ordered list of ``{pattern, endpoint}`` rules; ``endpoint``
      may be ``null`` to register an offline sentinel route).

    Outflow keys (and ``mode`` transitions) trigger an atomic rebuild of the
    underlying :class:`OutflowRouter`: validation runs first; the new component
    is started, swapped into the router, and the old one is stopped.

    Concurrent updates are serialised, and proxy requests are quiesced during
    the swap (see :class:`ConfigGate`).  Posting a payload byte-identical to
    the last successfully-applied one short-circuits as a no-op — important
    when multiple xdist workers race to push the same routing table at session
    start.

    Unknown keys cause a 400 response so callers can rely on strict validation.
    Returns the new redacted snapshot.
    """
    router: Router = request.app["router"]
    config_gate: ConfigGate = request.app["config_gate"]
    try:
        body = await request.json()
    except (ValueError, Exception):  # pylint: disable=broad-except
        return web.json_response(status=400, data={"error": {"message": "Invalid JSON body", "type": "invalid_request"}})
    if not isinstance(body, dict):
        return web.json_response(status=400, data={"error": {"message": "Body must be a JSON object", "type": "invalid_request"}})

    # Idempotency short-circuit: if the canonicalised body matches the last
    # successfully-applied payload, return the current snapshot without taking
    # the update lock or quiescing proxy traffic.  Validation is skipped because
    # the previous-identical body already passed it.
    payload_hash = _hash_payload(body)
    if payload_hash == config_gate.last_payload_hash:
        log.debug("POST /gate/config no-op: payload hash %s matches last applied", payload_hash)
        return web.json_response(_config_snapshot(router))

    router_keys = {"mode", "fuzzy_model", "fuzzy_sampling", "max_non_greedy_replies", "non_streaming_models"}
    outflow_keys = {"endpoints", "models"}
    accepted = router_keys | outflow_keys
    unknown = set(body) - accepted
    if unknown:
        return web.json_response(status=400, data={"error": {
            "message": f"Unknown config keys: {sorted(unknown)}. Accepted: {sorted(accepted)}",
            "type": "invalid_request",
        }})

    # --- Validate scalar router-level fields up-front ---
    if "mode" in body:
        try:
            new_mode = Mode(body["mode"])
        except ValueError:
            return web.json_response(status=400, data={"error": {
                "message": f"Invalid mode: {body['mode']!r}. Expected one of {[m.value for m in Mode]}",
                "type": "invalid_request",
            }})
    else:
        new_mode = router.mode

    if "fuzzy_model" in body and not isinstance(body["fuzzy_model"], bool):
        return web.json_response(status=400, data={"error": {
            "message": "fuzzy_model must be a boolean",
            "type": "invalid_request",
        }})
    if "fuzzy_sampling" in body and body["fuzzy_sampling"] not in _FUZZY_SAMPLING_VALUES:
        return web.json_response(status=400, data={"error": {
            "message": f"fuzzy_sampling must be one of {list(_FUZZY_SAMPLING_VALUES)}",
            "type": "invalid_request",
        }})
    if "max_non_greedy_replies" in body:
        value = body["max_non_greedy_replies"]
        if not isinstance(value, int) or value < 1:
            return web.json_response(status=400, data={"error": {
                "message": "max_non_greedy_replies must be a positive integer",
                "type": "invalid_request",
            }})
    if "non_streaming_models" in body:
        value = body["non_streaming_models"]
        if not isinstance(value, list) or not all(isinstance(m, str) for m in value):
            return web.json_response(status=400, data={"error": {
                "message": "non_streaming_models must be a list of strings",
                "type": "invalid_request",
            }})

    # --- Validate outflow-level fields ---
    if "endpoints" in body:
        err = _validate_endpoints_payload(body["endpoints"])
        if err:
            return web.json_response(status=400, data={"error": {"message": err, "type": "invalid_request"}})
    if "models" in body:
        # Determine the endpoint-name set against which to validate models:
        # body's endpoints (if present) take precedence over the live ones.
        if "endpoints" in body:
            endpoint_names = set(body["endpoints"].keys())
        else:
            cur_endpoints, _ = _extract_current_outflow_state(router.outflow)
            endpoint_names = set(cur_endpoints.keys())
        err = _validate_models_payload(body["models"], endpoint_names)
        if err:
            return web.json_response(status=400, data={"error": {"message": err, "type": "invalid_request"}})

    # --- Apply scalar router-level changes (cannot fail past this point) ---
    # Acquire the config gate: serialises concurrent updates and waits for
    # any in-flight proxy requests to drain before mutating router state.
    async with config_gate.exclusive_update():
        router.mode = new_mode
        if "fuzzy_model" in body:
            router.fuzzy_model = body["fuzzy_model"]
        if "fuzzy_sampling" in body:
            router.fuzzy_sampling = body["fuzzy_sampling"]
        if "max_non_greedy_replies" in body:
            router.max_non_greedy_replies = body["max_non_greedy_replies"]
        if "non_streaming_models" in body:
            router.non_streaming_models = list(body["non_streaming_models"])

        # --- Atomically rebuild outflow when mode or any outflow key changed ---
        needs_rebuild = "mode" in body or bool(outflow_keys & set(body))
        if needs_rebuild:
            # Use the gate's record_timeout if available on the app, else 600.0 default.
            default_timeout = float(request.app.get("record_timeout", 600.0))
            new_outflow = _build_outflow(body, router.outflow, new_mode, default_timeout)
            if new_outflow is not None:
                await new_outflow.start()
            old_outflow = router.outflow
            router.outflow = new_outflow
            if old_outflow is not None:
                try:
                    await old_outflow.stop()
                except Exception:  # pylint: disable=broad-except
                    log.warning("Old outflow.stop() raised during /gate/config rebuild", exc_info=True)

        # Record the successfully-applied hash so repeated identical pushes
        # short-circuit on the idempotency check above.
        config_gate.last_payload_hash = payload_hash

    log.info("Runtime config updated via /gate/config: %s", sorted(body))
    return web.json_response(_config_snapshot(router))


async def _handle_index_reload(request: web.Request) -> web.Response:
    """
    Force a rebuild of the on-disk cassette index.

    Useful when external tools have added cassettes to the cache directory
    while the server was running.
    """
    router: Router = request.app["router"]
    router.storage.index.rebuild(router.storage.requests_dir)
    return web.json_response({
        "status": "reloaded",
        "cassettes": len(router.storage.index.by_content_hash),
    })


async def _handle_stats(request: web.Request) -> web.Response:
    """
    Return per-tier counters describing the current cache state.
    """
    router: Router = request.app["router"]
    storage = router.storage
    return web.json_response({
        "cassettes": len(storage.index.by_content_hash),
        "by_prompt_model_hash": len(storage.index.by_prompt_model_hash),
        "by_prompt_hash": len(storage.index.by_prompt_hash),
        "mode": router.mode.value,
    })


async def _handle_gate_not_found(request: web.Request) -> web.Response:
    """
    Catch-all for unknown ``/gate/*`` paths so they do not fall through to the
    proxy handler.  Returns 404 with a structured JSON body.
    """
    return web.json_response(status=404, data={
        "error": {
            "message": f"Unknown admin endpoint: {request.path}",
            "type": "not_found",
        }
    })


def register_admin_routes(app: web.Application, router: Router, record_timeout: float = 600.0) -> None:
    """
    Register the ``/gate/*`` admin namespace on ``app`` and stash ``router``
    under ``app['router']`` for the handlers.

    ``record_timeout`` is the default upstream HTTP timeout (in seconds) used
    when a ``POST /gate/config`` payload omits per-endpoint ``timeout`` fields.

    Routes are added before any catch-all proxy route so they take precedence.

    Also installs a :class:`ConfigGate` under ``app['config_gate']`` so the
    proxy handler can quiesce in-flight requests during config rebuilds.
    """
    app["router"] = router
    app["record_timeout"] = record_timeout
    app["config_gate"] = ConfigGate()
    app.router.add_route("GET", "/gate/health", _handle_health)
    app.router.add_route("GET", "/gate/config", _handle_get_config)
    app.router.add_route("POST", "/gate/config", _handle_post_config)
    app.router.add_route("POST", "/gate/index/reload", _handle_index_reload)
    app.router.add_route("GET", "/gate/stats", _handle_stats)
    app.router.add_route("*", "/gate/{path:.*}", _handle_gate_not_found)
