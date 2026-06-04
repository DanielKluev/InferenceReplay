"""
Model-based upstream routing for multi-endpoint recording.

The ``OutflowRouter`` inspects each request's ``model`` field and dispatches
to the correct upstream endpoint based on the configured routing table.
This allows a single InferenceGate proxy to record cassettes for models
served by different backends (e.g. vLLM on one GPU server, llama.cpp on
another), and to register models that are *known* to the contract but
have no live endpoint on the current developer machine (offline sentinel).

Routing semantics:

1. Exact model-name match wins outright.
2. Otherwise the most-specific glob pattern wins, where specificity is the
   number of non-wildcard characters in the pattern.  Ties are broken by
   declaration order (earlier wins).
3. The literal ``"*"`` catch-all is treated as the lowest-specificity glob.
4. If no pattern matches the model, ``forward_request`` returns a structured
   HTTP 422 ``unrouted_model`` error response.
5. If a pattern matches but the resolved route's endpoint is ``None``
   (offline sentinel), ``forward_request`` returns a structured HTTP 503
   ``model_offline`` error response.

Unique endpoints are deduplicated by ``(url, api_key, proxy)`` tuple so two
routes pointing at the same physical endpoint share a single
``OutflowClient`` instance.

Key classes: ``EndpointConfig``, ``ModelRoute``, ``OutflowRouter``
"""

import fnmatch
import logging
from dataclasses import dataclass

from inference_gate.outflow.client import OutflowClient
from inference_gate.recording.storage import CachedRequest, CachedResponse


@dataclass(frozen=True)
class EndpointConfig:
    """
    Connection details for a single named upstream endpoint.

    Endpoints are referenced by name from ``ModelRoute`` entries.  Two
    endpoints with identical ``(url, api_key, proxy)`` triples share a single
    underlying ``OutflowClient`` (deduplicated by the router).

    Attributes:
        url: Base URL of the upstream API (e.g. ``"http://127.0.0.1:8125"``).
        api_key: Optional Bearer token for authentication.
        timeout: Request timeout in seconds.
        proxy: Optional HTTP proxy URL for routing upstream requests.
    """

    url: str
    api_key: str | None = None
    timeout: float = 120.0
    proxy: str | None = None
    verify_ssl: bool = True

    def dedup_key(self) -> tuple[str, str | None, str | None, bool]:
        """
        Return the tuple used to dedupe equivalent endpoints to a single client.
        """
        return (self.url.rstrip("/"), self.api_key, self.proxy, self.verify_ssl)


@dataclass
class ModelRoute:
    """
    A single entry in the routing table.

    Attributes:
        pattern: Exact model name or fnmatch glob (e.g. ``"Gemma4:*"``, ``"*"``).
        endpoint_name: Name of the endpoint to forward to, or ``None`` to mark
            the route as the *offline sentinel* (route exists but no live
            upstream is available — yields HTTP 503 ``model_offline``).
        order: Declaration order, used as a tie-breaker between equally
            specific glob patterns.
    """

    pattern: str
    endpoint_name: str | None
    order: int = 0


def _pattern_specificity(pattern: str) -> int:
    """
    Return the number of non-wildcard characters in a glob pattern.

    Higher is more specific.  Used to choose the best glob match when several
    patterns match the same model name.
    """
    return sum(1 for ch in pattern if ch not in ("*", "?"))


def _is_glob(pattern: str) -> bool:
    """
    Return True when ``pattern`` contains any fnmatch wildcard character.
    """
    return any(ch in pattern for ch in ("*", "?", "[", "]"))


class OutflowRouter:
    """
    Routes outgoing requests to different upstream endpoints based on the
    request's ``model`` field.

    Implements the same interface as :class:`OutflowClient` (``start``,
    ``stop``, ``forward_request``) so it can be used as a drop-in replacement
    in the :class:`Router` component.

    Unlike a single ``OutflowClient``, this class returns structured error
    responses (422 ``unrouted_model`` / 503 ``model_offline``) for requests
    whose model has no usable live endpoint, instead of raising.
    """

    def __init__(self, endpoints: dict[str, EndpointConfig], routes: list[ModelRoute]) -> None:
        """
        Initialize the outflow router.

        ``endpoints`` maps endpoint name → :class:`EndpointConfig`.  Each
        unique ``(url, api_key, proxy)`` triple becomes a single pooled
        :class:`OutflowClient`.

        ``routes`` is the ordered routing table.  ``ModelRoute.endpoint_name``
        must be either a key of ``endpoints`` or ``None`` (offline sentinel).
        Validation is performed eagerly.
        """
        self.log = logging.getLogger("OutflowRouter")

        # Validate every non-None endpoint reference up-front.
        for route in routes:
            if route.endpoint_name is not None and route.endpoint_name not in endpoints:
                raise ValueError(
                    f"ModelRoute(pattern={route.pattern!r}) references unknown endpoint {route.endpoint_name!r}; "
                    f"known endpoints: {sorted(endpoints)}")

        self._endpoints: dict[str, EndpointConfig] = dict(endpoints)
        self._routes: list[ModelRoute] = list(routes)

        # Split routes into exact-name and glob lookups.  Preserve declaration
        # order for tie-breaking on equally-specific globs.
        self._exact_routes: dict[str, ModelRoute] = {}
        self._glob_routes: list[ModelRoute] = []
        for idx, route in enumerate(self._routes):
            route.order = idx
            if _is_glob(route.pattern):
                self._glob_routes.append(route)
            else:
                self._exact_routes[route.pattern] = route

        # Sort globs by descending specificity, then ascending declaration order
        # so the first match in this list is always the winner.
        self._glob_routes.sort(key=lambda r: (-_pattern_specificity(r.pattern), r.order))

        # Build the deduplicated client pool keyed by EndpointConfig.dedup_key().
        # endpoint_name → (dedup_key, OutflowClient) so we can resolve quickly.
        self._client_by_key: dict[tuple[str, str | None, str | None, bool], OutflowClient] = {}
        self._client_by_endpoint: dict[str, OutflowClient] = {}
        for name, cfg in self._endpoints.items():
            key = cfg.dedup_key()
            if key not in self._client_by_key:
                self._client_by_key[key] = OutflowClient(upstream_base_url=cfg.url, api_key=cfg.api_key, timeout=cfg.timeout,
                                                         proxy=cfg.proxy, verify_ssl=cfg.verify_ssl)
            self._client_by_endpoint[name] = self._client_by_key[key]

    @property
    def endpoints(self) -> dict[str, EndpointConfig]:
        """
        Read-only view of the configured endpoint table.
        """
        return dict(self._endpoints)

    @property
    def routes(self) -> list[ModelRoute]:
        """
        Read-only view of the routing table in declaration order.
        """
        return list(self._routes)

    def _resolve_route(self, model_name: str | None) -> ModelRoute | None:
        """
        Find the matching :class:`ModelRoute` for ``model_name`` or return
        ``None`` when the model is unrouted.

        Resolution order: exact match → most-specific glob → no match.
        """
        if model_name:
            exact = self._exact_routes.get(model_name)
            if exact is not None:
                self.log.debug("Exact route match for model '%s'", model_name)
                return exact
            for route in self._glob_routes:
                if fnmatch.fnmatch(model_name, route.pattern):
                    self.log.debug("Glob route match '%s' for model '%s'", route.pattern, model_name)
                    return route
            return None
        # No model name — only the bare ``"*"`` catch-all (or empty-string exact)
        # can match.  Honour the same precedence rules.
        empty_exact = self._exact_routes.get("")
        if empty_exact is not None:
            return empty_exact
        for route in self._glob_routes:
            if fnmatch.fnmatch("", route.pattern):
                return route
        return None

    async def start(self) -> None:
        """
        Start every pooled ``OutflowClient`` instance.
        """
        for client in self._client_by_key.values():
            await client.start()
        self.log.info("OutflowRouter started with %d unique endpoint(s); %d route(s) configured", len(self._client_by_key),
                      len(self._routes))

    async def stop(self) -> None:
        """
        Stop every pooled ``OutflowClient`` instance.
        """
        for client in self._client_by_key.values():
            await client.stop()
        self.log.info("OutflowRouter stopped")

    async def forward_request(self, request: CachedRequest) -> CachedResponse:
        """
        Forward a request to the upstream selected by the request's ``model``
        field, or return a structured error response when no live route is
        available.

        Returns a 422 ``unrouted_model`` response when no pattern matches the
        model, and a 503 ``model_offline`` response when the matched route
        has ``endpoint_name=None`` (offline sentinel).  Otherwise delegates
        to the pooled :class:`OutflowClient`.
        """
        model_name: str | None = None
        if request.body and isinstance(request.body, dict):
            model_name = request.body.get("model")

        route = self._resolve_route(model_name)
        if route is None:
            self.log.warning("Unrouted model %r (%s %s)", model_name, request.method, request.path)
            return CachedResponse(
                status_code=422,
                headers={"Content-Type": "application/json"},
                body={
                    "error": {
                        "message":
                            f"Model {model_name!r} is not registered in the InferenceGate routing table; "
                            "add it to the configured ``models`` map or use a glob/catch-all pattern.",
                        "type": "unrouted_model",
                        "code": "unrouted_model",
                        "model": model_name,
                    }
                },
            )

        if route.endpoint_name is None:
            self.log.warning("Model %r matched offline route %r (no live endpoint on this machine)", model_name, route.pattern)
            return CachedResponse(
                status_code=503,
                headers={"Content-Type": "application/json"},
                body={
                    "error": {
                        "message":
                            f"Model {model_name!r} is registered (matched route {route.pattern!r}) but no live "
                            "endpoint is configured on this machine; cassette replay is required.",
                        "type": "model_offline",
                        "code": "model_offline",
                        "model": model_name,
                        "matched_pattern": route.pattern,
                    }
                },
            )

        client = self._client_by_endpoint[route.endpoint_name]
        return await client.forward_request(request)
