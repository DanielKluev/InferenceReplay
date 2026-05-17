"""
Core class that glues all other components together and serves as the entrypoint.

Manages the lifecycle of the application: creates, starts, and stops all
components (Inflow, Router, Outflow, Recording).

Key classes: `InferenceGate`
"""

import asyncio
import logging

from inference_gate.inflow.server import InflowServer
from inference_gate.modes import Mode
from inference_gate.outflow.model_router import EndpointConfig, ModelRoute, OutflowRouter
from inference_gate.recording.storage import CacheStorage
from inference_gate.router.router import Router


class InferenceGate:
    """
    Central orchestrator for the InferenceGate proxy system.

    Creates and manages all components: InflowServer, Router, OutflowRouter,
    and CacheStorage. Handles startup and shutdown lifecycle.

    All upstream connection details live inside the ``endpoints`` map
    (Glue-style schema).  The ``models`` list is the routing table that
    binds a model name (or glob pattern) to an endpoint, or to ``None``
    (offline sentinel — model is registered but no live upstream is
    available, replay-only).  When ``mode == RECORD_AND_REPLAY`` the
    constructor builds a single :class:`OutflowRouter` from these inputs;
    in ``REPLAY_ONLY`` mode no outflow is created.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8080, mode: Mode = Mode.RECORD_AND_REPLAY, cache_dir: str = ".inference_cache",
                 web_ui: bool = False, web_ui_port: int = 8081, non_streaming_models: list[str] | None = None, fuzzy_model: bool = False,
                 fuzzy_sampling: str = "off", max_non_greedy_replies: int = 5, max_live_requests: int | None = None,
                 record_timeout: float = 600.0, endpoints: dict[str, EndpointConfig] | None = None,
                 models: list[ModelRoute] | None = None) -> None:
        """
        Initialize InferenceGate with configuration.

        `host` and `port` configure the local proxy server address.
        `mode` determines behavior: record-and-replay or replay-only.
        `cache_dir` is the directory for storing cached inferences.
        `web_ui` enables the optional web dashboard.
        `web_ui_port` is the port for the web UI server.
        `non_streaming_models` is a list of model names that do not support streaming.
        `fuzzy_model` enables fallback to cache entries with the same prompt
        but a different model when the exact cache key is not found.
        `fuzzy_sampling` controls sampling parameter fuzzy matching: "off", "soft", or "aggressive".
        `max_non_greedy_replies` is the max replies to collect per non-greedy cassette.
        `record_timeout` is the default upstream HTTP timeout (seconds) used
        when an endpoint config does not set its own ``timeout``.  It also
        bounds how long ``forward_request`` may block during recording.
        `endpoints` maps endpoint name → :class:`EndpointConfig`.  Endpoints
        with the same ``(url, api_key, proxy)`` triple share a pooled
        ``OutflowClient``.  Empty/None means no live endpoints are available
        — only ``REPLAY_ONLY`` mode (or a record-mode config pushed via
        ``POST /gate/config``) will work.
        `models` is the ordered routing table.  Each :class:`ModelRoute`
        binds a model name or glob to an endpoint name, or to ``None`` to
        register an *offline* model (matched requests in record mode are
        rejected with HTTP 503 ``model_offline``).
        """
        self.log = logging.getLogger("InferenceGate")
        self.host = host
        self.port = port
        self.mode = mode
        self.cache_dir = cache_dir
        self.web_ui = web_ui
        self.web_ui_port = web_ui_port
        self.non_streaming_models = non_streaming_models or []
        self.fuzzy_model = fuzzy_model
        self.fuzzy_sampling = fuzzy_sampling
        self.max_non_greedy_replies = max_non_greedy_replies
        self.max_live_requests = max_live_requests
        self.record_timeout = record_timeout
        self.endpoints: dict[str, EndpointConfig] = dict(endpoints) if endpoints else {}
        self.models: list[ModelRoute] = list(models) if models else []

        # Components (created during start)
        self._storage: CacheStorage | None = None
        self._outflow: OutflowRouter | None = None
        self._router: Router | None = None
        self._server: InflowServer | None = None
        self._webui_server: "WebUIServer | None" = None

    def _create_components(self) -> None:
        """
        Create all system components based on configuration.

        Creates CacheStorage, OutflowRouter (if needed), Router, InflowServer, and optionally WebUIServer.
        """
        self._storage = CacheStorage(self.cache_dir)

        # OutflowRouter is only needed for RECORD_AND_REPLAY mode.  In replay
        # mode we deliberately leave ``_outflow`` as ``None`` so a stray
        # cache miss surfaces as a 503 from ``Router._handle_replay_only``
        # rather than silently reaching upstream.
        if self.mode == Mode.RECORD_AND_REPLAY:
            self._outflow = OutflowRouter(endpoints=self.endpoints, routes=self.models)
        else:
            self._outflow = None

        self._router = Router(mode=self.mode, storage=self._storage, outflow=self._outflow, non_streaming_models=self.non_streaming_models,
                              fuzzy_model=self.fuzzy_model, fuzzy_sampling=self.fuzzy_sampling,
                              max_non_greedy_replies=self.max_non_greedy_replies, max_live_requests=self.max_live_requests)
        self._server = InflowServer(host=self.host, port=self.port, router=self._router, record_timeout=self.record_timeout)

        # WebUIServer is optional
        if self.web_ui:
            from inference_gate.webui.server import WebUIServer
            # Only expose a non-None upstream_base_url to the WebUI when we actually have upstream access
            # enabled.  With the new multi-endpoint schema we surface the *first* configured endpoint URL
            # (or None when no live endpoints are configured) for display purposes; the WebUI does not need
            # the full routing table.
            webui_upstream_base_url: str | None = None
            if self.mode == Mode.RECORD_AND_REPLAY and self.endpoints:
                webui_upstream_base_url = next(iter(self.endpoints.values())).url
            # For security, always bind WebUI to localhost unless user explicitly configures a different host
            webui_host = "127.0.0.1"
            self._webui_server = WebUIServer(host=webui_host, port=self.web_ui_port, storage=self._storage, mode=self.mode,
                                             cache_dir=self.cache_dir, upstream_base_url=webui_upstream_base_url, proxy_host=self.host,
                                             proxy_port=self.port)

    async def start(self) -> None:
        """
        Start InferenceGate and all its components.

        Creates components, starts the outflow client (if applicable),
        and starts the inflow HTTP server and WebUI server (if enabled).
        """
        self.log.info("Starting InferenceGate in %s mode", self.mode.value)
        self._create_components()

        # Start outflow client if we need upstream access
        if self._outflow is not None:
            await self._outflow.start()

        # Start the inflow HTTP server
        assert self._server is not None
        await self._server.start()

        # Start the WebUI server if enabled
        if self._webui_server is not None:
            await self._webui_server.start()

        self.log.info("InferenceGate ready on http://%s:%d", self.host, self.port)
        if self._webui_server is not None:
            self.log.info("WebUI dashboard available at http://127.0.0.1:%d", self.web_ui_port)

    async def stop(self) -> None:
        """
        Stop InferenceGate and all its components.

        Stops the inflow server, WebUI server (if enabled), and outflow client in order.
        """
        self.log.info("Stopping InferenceGate")

        if self._webui_server is not None:
            await self._webui_server.stop()

        if self._server is not None:
            await self._server.stop()

        if self._outflow is not None:
            await self._outflow.stop()

        self.log.info("InferenceGate stopped")

    async def run_forever(self) -> None:
        """
        Start InferenceGate and run until interrupted.

        Blocks until a KeyboardInterrupt or cancellation occurs,
        then performs clean shutdown.
        """
        await self.start()
        try:
            # Run until interrupted
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await self.stop()

    @property
    def actual_port(self) -> int:
        """
        Get the actual port the inflow server is listening on.

        When started with port 0, the OS assigns an ephemeral port.
        This property returns that actual port after `start()` has been called.
        Falls back to the configured port if the server is not yet started.
        """
        if self._server is not None:
            return self._server.actual_port
        return self.port

    @property
    def base_url(self) -> str:
        """
        Get the base URL of the running InferenceGate proxy.

        Returns a URL like `http://127.0.0.1:54321` using the actual port.
        """
        return f"http://{self.host}:{self.actual_port}"

    @property
    def storage(self) -> CacheStorage | None:
        """
        Get the CacheStorage instance, if created.
        """
        return self._storage

    @property
    def router(self) -> Router | None:
        """
        Get the active `Router` instance, if the gate has been started.

        Exposed to allow test infrastructure (see ``inference_gate.pytest_plugin``
        markers ``inferencegate_strict`` / ``inferencegate``) to mutate
        per-test matching policy (``fuzzy_model`` / ``fuzzy_sampling``)
        without restarting the proxy.
        """
        return self._router
