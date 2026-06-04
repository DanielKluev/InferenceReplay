"""
Outflow client for forwarding requests to the real AI model endpoint.

Uses `aiohttp` for asynchronous HTTP requests to the upstream API.

Key classes: `OutflowClient`
"""

import logging
from typing import Any

import aiohttp

from inference_gate.recording.storage import CachedRequest, CachedResponse


class OutflowClient:
    """
    Forwards requests to the real AI model endpoint and returns responses.

    Uses an `aiohttp.ClientSession` for async HTTP communication.
    The client manages its own session lifecycle and must be properly
    started and stopped via `start()` and `stop()`.
    """

    def __init__(self, upstream_base_url: str, api_key: str | None = None, timeout: float = 120.0, proxy: str | None = None, verify_ssl: bool = True) -> None:
        """
        Initialize the outflow client.

        `upstream_base_url` is the base URL of the upstream AI API (e.g. "https://api.openai.com").
        Optional `api_key` is used for Bearer token authentication.
        `timeout` controls the total request timeout in seconds.
        `proxy` is an optional HTTP proxy URL (e.g. ``"http://127.0.0.1:8888/"``).
        When set, all upstream requests are routed through the specified proxy server.
        """
        self.log = logging.getLogger("OutflowClient")
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.proxy = proxy
        self.verify_ssl = verify_ssl
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        """
        Start the outflow client by creating the HTTP session.
        """
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        self.log.info("OutflowClient started, upstream: %s (proxy: %s)", self.upstream_base_url, self.proxy or "none")

    async def stop(self) -> None:
        """
        Stop the outflow client and close the HTTP session.
        """
        if self._session is not None:
            await self._session.close()
            self._session = None
        self.log.info("OutflowClient stopped")

    def _get_session(self) -> aiohttp.ClientSession:
        """
        Get the active HTTP session.

        Raises RuntimeError if session is not started.
        """
        if self._session is None:
            raise RuntimeError("OutflowClient not started. Call start() first.")
        return self._session

    def _build_upstream_headers(self, request: CachedRequest) -> dict[str, str]:
        """
        Build headers for the upstream request.

        Filters out hop-by-hop headers and injects API key if configured.
        """
        headers: dict[str, str] = {}
        for key, value in request.headers.items():
            key_lower = key.lower()
            # Skip hop-by-hop headers
            if key_lower in ("host", "content-length", "transfer-encoding"):
                continue
            headers[key] = value

        # Inject API key if configured
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Ensure content-type is set for requests with body
        if request.body is not None and "content-type" not in {k.lower() for k in headers}:
            headers["Content-Type"] = "application/json"

        return headers

    async def forward_request(self, request: CachedRequest) -> CachedResponse:
        """
        Forward a request to the upstream API and return the response.

        Handles both streaming and non-streaming responses.
        For streaming responses, all chunks are collected and stored.

        Returns a `CachedResponse` containing the upstream response data.
        """
        session = self._get_session()
        url = f"{self.upstream_base_url}{request.path}"
        headers = self._build_upstream_headers(request)

        # Determine if this is a streaming request
        is_streaming = False
        if request.body and isinstance(request.body, dict):
            is_streaming = request.body.get("stream", False)

        self.log.debug("Forwarding %s %s (streaming=%s)", request.method, url, is_streaming)

        if is_streaming:
            return await self._forward_streaming(session, request.method, url, headers, request.body)
        else:
            return await self._forward_standard(session, request.method, url, headers, request.body)

    async def _forward_standard(self, session: aiohttp.ClientSession, method: str, url: str, headers: dict[str, str],
                                body: dict[str, Any] | None) -> CachedResponse:
        """
        Forward a standard (non-streaming) request to upstream.

        Returns a `CachedResponse` with the parsed JSON body.
        """
        kwargs: dict[str, Any] = {"headers": headers}
        if body is not None:
            kwargs["json"] = body
        if self.proxy:
            kwargs["proxy"] = self.proxy
        if not self.verify_ssl:
            kwargs["ssl"] = False

        async with session.request(method, url, **kwargs) as resp:
            response_headers = {k: v for k, v in resp.headers.items() if k.lower() in ("content-type",)}
            try:
                response_body = await resp.json()
            except Exception:
                # If response is not JSON, store as text wrapped in a dict
                text = await resp.text()
                response_body = {"_raw_text": text}

            self.log.info("Upstream responded %d for %s %s", resp.status, method, url)
            return CachedResponse(status_code=resp.status, headers=response_headers, body=response_body, is_streaming=False)

    async def _forward_streaming(self, session: aiohttp.ClientSession, method: str, url: str, headers: dict[str, str],
                                 body: dict[str, Any] | None) -> CachedResponse:
        """
        Forward a streaming request to upstream and collect all chunks.

        Returns a `CachedResponse` with collected SSE chunks.
        """
        kwargs: dict[str, Any] = {"headers": headers}
        if body is not None:
            kwargs["json"] = body
        if self.proxy:
            kwargs["proxy"] = self.proxy
        if not self.verify_ssl:
            kwargs["ssl"] = False

        chunks: list[str] = []
        async with session.request(method, url, **kwargs) as resp:
            response_headers = {k: v for k, v in resp.headers.items() if k.lower() in ("content-type",)}
            async for chunk_bytes in resp.content.iter_any():
                chunk_str = chunk_bytes.decode("utf-8")
                chunks.append(chunk_str)

            self.log.info("Upstream streaming completed %d for %s %s (%d chunks)", resp.status, method, url, len(chunks))
            return CachedResponse(status_code=resp.status, headers=response_headers, chunks=chunks, is_streaming=True)
