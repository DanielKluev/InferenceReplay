"""
Router component that determines how to handle incoming requests.

Decides whether to replay an inference from local storage or forward
the request to the real AI model endpoint based on the current operating mode.
Supports multi-reply cassettes for non-greedy sampling and tiered fuzzy
matching (exact → sampling fuzzy → model fuzzy).

Key classes: `Router`, `ReplayCounter`
"""

import logging
import random
from typing import Any

from inference_gate.headers import (HeaderValidationError, ParsedHeaders, parse_headers, required_engine_matches,
                                     strip_inferencegate_headers)
from inference_gate.modes import Mode
from inference_gate.outflow.client import OutflowClient
from inference_gate.outflow.model_router import OutflowRouter
from inference_gate.recording.hashing import compute_content_hash, compute_prompt_hash, compute_prompt_model_hash, is_greedy
from inference_gate.recording.models import IndexRow
from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry, CacheStorage

# Paths where the proxy may force ``stream=True`` on upstream requests.
# These *must* have a matching reassembly function in ``recording.reassembly``
# so that a single streaming cassette can serve both streaming and
# non-streaming clients.  All other paths (``/tokenize``, ``/detokenize``,
# ``/v1/models``, ``/v1/completions``, etc.) are forwarded as-is.
_FORCE_STREAMING_PATH_PREFIXES: tuple[str, ...] = (
    "/v1/chat/completions",
    "/v1/responses",
    "/responses",
)


class ReplayCounter:
    """
    Tracks which reply to serve next for multi-reply cassettes.

    Supports round-robin (default), random, and first strategies.
    """

    def __init__(self) -> None:
        # content_hash → next reply index (0-based internally, 1-based in tapes)
        self._counters: dict[str, int] = {}

    def next_reply(self, content_hash: str, total_replies: int, strategy: str = "round-robin") -> int:
        """
        Get the next reply number (1-based) to serve for a cassette.

        `strategy` is one of: "round-robin", "random", "first".
        """
        if total_replies <= 0:
            return 1

        if strategy == "first":
            return 1

        if strategy == "random":
            return random.randint(1, total_replies)

        # round-robin (default)
        idx = self._counters.get(content_hash, 0)
        reply_num = (idx % total_replies) + 1
        self._counters[content_hash] = idx + 1
        return reply_num


class Router:
    """
    Routes incoming requests to either cached storage or upstream API.

    In `RECORD_AND_REPLAY` mode, checks cache first and returns cached response
    if available. On cache miss, forwards to upstream via OutflowClient, records
    the response, and returns it.

    For non-greedy requests (temperature > 0 or unspecified), the router collects
    multiple replies up to `max_non_greedy_replies` before switching to replay
    cycling. For greedy requests (temperature == 0), a single reply is stored
    and replayed immediately on cache hit.

    Supports tiered fuzzy matching:
    1. Exact match (content_hash)
    2. Sampling fuzzy (prompt_model_hash) — when `fuzzy_sampling` is not "off"
    3. Model fuzzy (prompt_hash) — when `fuzzy_model` is True

    In `REPLAY_ONLY` mode, only returns cached responses.
    """

    def __init__(self, mode: Mode, storage: CacheStorage, outflow: OutflowClient | OutflowRouter | None = None,
                 non_streaming_models: list[str] | None = None, fuzzy_model: bool = False, fuzzy_sampling: str = "off",
                 max_non_greedy_replies: int = 5, max_live_requests: int | None = None) -> None:
        """
        Initialize the router.

        `mode` determines the routing behavior.
        `storage` is the cache storage for recorded inferences.
        `outflow` is the client for forwarding to upstream (required for RECORD_AND_REPLAY mode).
            Accepts either a single ``OutflowClient`` or an ``OutflowRouter`` for
            model-based multi-upstream routing.
        `non_streaming_models` is a list of model names that do not support streaming.
        `fuzzy_model` enables fallback to cache entries with the same prompt but a different model.
        `fuzzy_sampling` controls sampling parameter fuzzy matching: "off", "soft", or "aggressive".
        `max_non_greedy_replies` is the max replies to collect per non-greedy cassette.
        """
        self.log = logging.getLogger("Router")
        self.mode = mode
        self.storage = storage
        self.outflow = outflow
        self.non_streaming_models = non_streaming_models or []
        self.fuzzy_model = fuzzy_model
        self.fuzzy_sampling = fuzzy_sampling
        self.max_non_greedy_replies = max_non_greedy_replies
        self.max_live_requests = max_live_requests
        self.live_requests_count = 0
        self.replay_counter = ReplayCounter()

        if mode == Mode.RECORD_AND_REPLAY and outflow is None:
            raise ValueError("OutflowClient is required for RECORD_AND_REPLAY mode")

    async def route_request(self, method: str, path: str, headers: dict[str, str], body: dict[str, Any] | None = None,
                            query_params: dict[str, str] | None = None) -> CachedResponse:
        """
        Route an incoming request based on the current mode.

        Builds a `CachedRequest`, performs tiered cache lookup, and either
        replays or forwards to upstream depending on the mode.

        Parses and strips `X-InferenceGate-*` contract headers (and the legacy
        `X-Gate-Reply-Strategy` alias) before any further processing so they
        never enter cache-hash computations or upstream traffic.  Unknown
        contract headers raise `HeaderValidationError` which the inflow server
        translates into a 400 response.

        Returns a `CachedResponse` with the response data.
        """
        # Parse contract headers first so a malformed header surfaces as 400 before any work.
        parsed = parse_headers(headers)
        # Strip contract headers from the dict that flows into hashing / upstream.
        clean_headers = strip_inferencegate_headers(headers)

        # Resolve per-request overrides for mode, fuzzy_model, fuzzy_sampling, reply strategy.
        effective_mode = self._resolve_mode(parsed)
        effective_fuzzy_model = self._resolve_fuzzy_model(parsed)
        effective_fuzzy_sampling = self._resolve_fuzzy_sampling(parsed)
        strategy = parsed.control.get("reply_strategy", "round-robin")
        if strategy not in ("round-robin", "random", "first"):
            strategy = "round-robin"

        # Build cache request for lookup using the cleaned header set.
        cached_request = self._build_cached_request(method, path, clean_headers, body, query_params)

        # Tiered cache lookup with per-request fuzzy overrides + Require-* engine filters.
        index_row = self._tiered_lookup(method, path, body, fuzzy_model=effective_fuzzy_model,
                                        fuzzy_sampling=effective_fuzzy_sampling, parsed=parsed)

        # Debug mirror: write the verbatim incoming request to ``requests_raw/<hash>.json``
        # regardless of mode or hit/miss.  This is the canonical artifact for "what did
        # the model actually see" debugging — chat-template glitches, prompt assembly
        # bugs, accidental tool-call leakage all surface in this directory.
        try:
            dbg_content_hash = compute_content_hash(method, path, body)
            dbg_pm_hash = compute_prompt_model_hash(method, path, body)
            dbg_p_hash = compute_prompt_hash(body)
            outcome = "hit" if index_row is not None else "miss"
            self.storage.dump_raw_request(method=method, path=path, headers=clean_headers, body=body,
                                          query_params=query_params, content_hash=dbg_content_hash,
                                          prompt_model_hash=dbg_pm_hash, prompt_hash=dbg_p_hash, outcome=outcome,
                                          extra={"mode": effective_mode.value, "fuzzy_model": effective_fuzzy_model,
                                                 "fuzzy_sampling": effective_fuzzy_sampling,
                                                 "matched_content_hash": index_row.content_hash if index_row else None})
        except Exception:  # pylint: disable=broad-except
            self.log.exception("requests_raw dump failed; ignoring")

        if effective_mode == Mode.REPLAY_ONLY:
            return self._handle_replay_only(index_row, cached_request, strategy)

        # RECORD_AND_REPLAY mode
        return await self._handle_record_and_replay(index_row, cached_request, strategy, parsed=parsed)

    def _resolve_mode(self, parsed: ParsedHeaders) -> Mode:
        """
        Resolve effective mode for this request, honouring `Control-Mode` override.

        Falls back to the router's session-default mode when no override is set.
        """
        override = parsed.control.get("mode")
        if override is None:
            return self.mode
        if override == "replay":
            return Mode.REPLAY_ONLY
        if override == "record":
            return Mode.RECORD_AND_REPLAY
        # ``passthrough`` is reserved; treated as record-and-replay for now.
        return Mode.RECORD_AND_REPLAY

    def _resolve_fuzzy_model(self, parsed: ParsedHeaders) -> bool:
        """
        Resolve effective `fuzzy_model` for this request.
        """
        value = parsed.require.get("fuzzy_model")
        if value is None:
            return self.fuzzy_model
        return value.lower() == "on"

    def _resolve_fuzzy_sampling(self, parsed: ParsedHeaders) -> str:
        """
        Resolve effective `fuzzy_sampling` for this request.
        """
        value = parsed.require.get("fuzzy_sampling")
        if value is None:
            return self.fuzzy_sampling
        return value.lower()

    def _build_cached_request(self, method: str, path: str, headers: dict[str, str], body: dict[str, Any] | None,
                              query_params: dict[str, str] | None) -> CachedRequest:
        """
        Build a `CachedRequest` from incoming request data.

        Filters headers to only include relevant ones for cache key computation.
        """
        # Filter headers to only include relevant ones for caching
        relevant_headers: dict[str, str] = {}
        for key, value in headers.items():
            key_lower = key.lower()
            if key_lower in ("content-type", "accept"):
                relevant_headers[key] = value

        return CachedRequest(method=method, path=path, headers=relevant_headers, body=body, query_params=query_params)

    def _tiered_lookup(self, method: str, path: str, body: dict[str, Any] | None, fuzzy_model: bool | None = None,
                       fuzzy_sampling: str | None = None, parsed: ParsedHeaders | None = None) -> IndexRow | None:
        """
        Perform tiered cache lookup: exact → sampling fuzzy → model fuzzy.

        ``fuzzy_model`` and ``fuzzy_sampling`` are per-request overrides; when
        ``None`` the router defaults are used.  ``parsed`` carries the parsed
        contract headers; when present, ``Require-Engine`` and
        ``Require-Engine-Version`` filter candidate cassettes via
        :func:`required_engine_matches` against the row's denormalised
        ``engine`` column (and tape metadata for finer matches).

        Returns the matching `IndexRow` or None.
        """
        effective_fuzzy_model = self.fuzzy_model if fuzzy_model is None else fuzzy_model
        effective_fuzzy_sampling = self.fuzzy_sampling if fuzzy_sampling is None else fuzzy_sampling

        # Tier 1: Exact match (content_hash)
        content_hash = compute_content_hash(method, path, body)
        row = self.storage.index.by_content_hash.get(content_hash)
        if row is not None and self._row_satisfies_requires(row, parsed):
            self.log.debug("Exact match: content_hash=%s", content_hash)
            return row

        # Tier 2: Sampling fuzzy (prompt_model_hash) — if enabled
        if effective_fuzzy_sampling != "off":
            pm_hash = compute_prompt_model_hash(method, path, body)
            candidates = self.storage.index.by_prompt_model_hash.get(pm_hash, [])
            candidates = [c for c in candidates if self._row_satisfies_requires(c, parsed)]
            row = self._filter_sampling_candidates(candidates, body, effective_fuzzy_sampling)
            if row is not None:
                self.log.info("Sampling fuzzy match: prompt_model_hash=%s, cached_model=%s", pm_hash, row.model)
                return row

        # Tier 3: Model fuzzy (prompt_hash) — if enabled
        if effective_fuzzy_model:
            p_hash = compute_prompt_hash(body)
            candidates = self.storage.index.by_prompt_hash.get(p_hash, [])
            candidates = [c for c in candidates if self._row_satisfies_requires(c, parsed)]
            # Apply sampling filter here too if fuzzy_sampling is active
            if effective_fuzzy_sampling != "off":
                row = self._filter_sampling_candidates(candidates, body, effective_fuzzy_sampling)
            elif candidates:
                row = self._pick_best_candidate(candidates)
            else:
                row = None
            if row is not None:
                self.log.info("Model fuzzy match: prompt_hash=%s, cached_model=%s", p_hash, row.model)
                return row

        return None

    def _row_satisfies_requires(self, row: IndexRow, parsed: ParsedHeaders | None) -> bool:
        """
        Apply ``Require-Engine`` / ``Require-Engine-Version`` filter to a candidate row.

        Engine name is denormalised onto the index row for fast filtering.  Engine
        version is not denormalised; when a version constraint is set we conservatively
        load the tape metadata to evaluate it.  When no constraints are set this is a no-op.
        """
        if parsed is None:
            return True
        require_engine = parsed.require.get("engine")
        require_engine_version = parsed.require.get("engine_version")
        if not require_engine and not require_engine_version:
            return True
        candidate_meta: dict[str, str] = {}
        if row.engine:
            candidate_meta["engine"] = row.engine
        if require_engine_version:
            # Need full tape metadata to evaluate version constraint.
            loaded = self.storage.load_tape(row.content_hash)
            if loaded is not None:
                tape_meta, _ = loaded
                candidate_meta = dict(tape_meta.metadata)
        return required_engine_matches(parsed, candidate_meta)

    def _filter_sampling_candidates(self, candidates: list[IndexRow], body: dict[str, Any] | None,
                                    fuzzy_sampling: str | None = None) -> IndexRow | None:
        """
        Filter and pick from sampling-fuzzy candidates based on the fuzzy_sampling level.

        `soft`: only non-greedy cassettes matched against non-greedy request (non-greedy ↔ non-greedy).
        `aggressive`: any cassette matches (greedy ↔ non-greedy allowed).

        ``fuzzy_sampling`` is a per-request override; when ``None`` the router default is used.
        """
        if not candidates:
            return None

        level = self.fuzzy_sampling if fuzzy_sampling is None else fuzzy_sampling

        if level == "soft":
            # Soft: only match non-greedy with non-greedy
            request_is_greedy = is_greedy(body)
            filtered = [r for r in candidates if r.is_greedy == request_is_greedy]
            return self._pick_best_candidate(filtered)

        if level == "aggressive":
            # Aggressive: any match is fine
            return self._pick_best_candidate(candidates)

        return None

    @staticmethod
    def _pick_best_candidate(candidates: list[IndexRow]) -> IndexRow | None:
        """
        Pick the best candidate from a list of index rows.

        Prefers the one with the most replies (most data), then most recent.
        """
        if not candidates:
            return None
        # Sort by replies desc, then recorded desc (most data first, newest first)
        return max(candidates, key=lambda r: (r.replies, r.recorded))

    def _handle_replay_only(self, index_row: IndexRow | None, cached_request: CachedRequest, strategy: str) -> CachedResponse:
        """
        Handle request in REPLAY_ONLY mode.

        Returns cached response if found (cycling through replies for multi-reply cassettes),
        otherwise returns 503 error response.
        """
        if index_row is not None:
            self.log.info("Replaying cached response for %s %s", cached_request.method, cached_request.path)
            return self._serve_reply(index_row, strategy)

        self.log.warning("Cache miss in replay-only mode for %s %s", cached_request.method, cached_request.path)
        return CachedResponse(
            status_code=503,
            headers={"Content-Type": "application/json"},
            body={
                "error": {
                    "message": f"No cached response for {cached_request.method} {cached_request.path}. "
                               "Running in replay-only mode - cannot proxy to upstream.",
                    "type": "cache_miss",
                    "code": "no_cached_response"
                }
            },
            is_streaming=False,
        )

    async def _handle_record_and_replay(self, index_row: IndexRow | None, cached_request: CachedRequest, strategy: str,
                                        parsed: ParsedHeaders | None = None) -> CachedResponse:
        """
        Handle request in RECORD_AND_REPLAY mode.

        For greedy requests: cache hit → replay immediately; cache miss → forward and store.
        For non-greedy requests: cache hit with replies < max → forward and append;
        cache hit with replies >= max → cycle through stored replies.

        ``parsed`` carries the parsed contract headers; ``Metadata-*`` fields are
        forwarded to storage for new cassettes.
        """
        request_is_greedy = is_greedy(cached_request.body)
        extra_metadata = parsed.metadata if parsed is not None else {}

        if index_row is not None:
            # Determine max replies for this cassette
            max_replies = 1 if request_is_greedy else self.max_non_greedy_replies

            if not request_is_greedy and index_row.replies < max_replies:
                # Non-greedy cassette not full yet — record another reply
                self.log.info("Non-greedy cassette %s has %d/%d replies — recording another", index_row.content_hash, index_row.replies,
                              max_replies)
                return await self._forward_and_append(cached_request, index_row, extra_metadata=extra_metadata)

            # Cassette full or greedy — replay
            self.log.info("Cache hit - replaying response for %s %s (replies=%d)", cached_request.method, cached_request.path,
                          index_row.replies)
            return self._serve_reply(index_row, strategy)

        # Cache miss - forward to upstream and create new cassette
        assert self.outflow is not None
        self.log.info("Cache miss - forwarding to upstream for %s %s", cached_request.method, cached_request.path)
        return await self._forward_and_store(cached_request, request_is_greedy, extra_metadata=extra_metadata)

    async def _forward_and_store(self, cached_request: CachedRequest, request_is_greedy: bool,
                                 extra_metadata: dict[str, str] | None = None) -> CachedResponse:
        """
        Forward a request to upstream, store the response as a new cassette, and return the response.

        ``extra_metadata`` carries parsed ``X-InferenceGate-Metadata-*`` fields and is
        passed to :meth:`CacheStorage.put_async`, which records them into the new tape.
        """
        assert self.outflow is not None

        if self.max_live_requests is not None and self.live_requests_count >= self.max_live_requests:
            self.log.warning("Live request limit reached (%d). Rejecting upstream call.", self.max_live_requests)
            return CachedResponse(
                status_code=503,
                headers={"Content-Type": "application/json"},
                body={
                    "error": {
                        "message": f"Global live request limit ({self.max_live_requests}) exceeded. Temporary unavailable.",
                        "type": "live_request_limit_exceeded",
                        "code": "temporary_unavailable"
                    }
                },
                is_streaming=False,
            )

        self.live_requests_count += 1

        # Remember the client's original streaming preference
        original_client_streaming = False
        if cached_request.body and isinstance(cached_request.body, dict):
            original_client_streaming = cached_request.body.get("stream", False)

        # Force streaming on the upstream request unless the model or path is excluded
        self._maybe_force_streaming(cached_request, original_client_streaming)

        response = await self.outflow.forward_request(cached_request)

        # Determine max replies for this cassette
        max_replies = 1 if request_is_greedy else self.max_non_greedy_replies

        # Record the response
        model = cached_request.body.get("model") if cached_request.body else None
        temperature = cached_request.body.get("temperature") if cached_request.body else None
        p_hash = compute_prompt_hash(cached_request.body)

        entry = CacheEntry(
            request=cached_request,
            response=response,
            model=model,
            temperature=temperature,
            prompt_hash=p_hash,
            original_client_streaming=original_client_streaming,
        )
        content_hash = await self.storage.put_async(entry, max_replies=max_replies, extra_metadata=extra_metadata)
        self.log.info("Recorded response with content hash %s", content_hash)

        return response

    async def _forward_and_append(self, cached_request: CachedRequest, index_row: IndexRow,
                                  extra_metadata: dict[str, str] | None = None) -> CachedResponse:
        """
        Forward a request to upstream and append the response to an existing cassette.

        ``extra_metadata`` is forwarded for symmetry; the underlying storage layer keeps
        first-write-wins semantics for cassette-level metadata (appended replies do not
        overwrite the existing ``TapeMetadata.metadata`` mapping).
        """
        assert self.outflow is not None

        if self.max_live_requests is not None and self.live_requests_count >= self.max_live_requests:
            self.log.warning("Live request limit reached (%d). Rejecting upstream call.", self.max_live_requests)
            return CachedResponse(
                status_code=503,
                headers={"Content-Type": "application/json"},
                body={
                    "error": {
                        "message": f"Global live request limit ({self.max_live_requests}) exceeded. Temporary unavailable.",
                        "type": "live_request_limit_exceeded",
                        "code": "temporary_unavailable"
                    }
                },
                is_streaming=False,
            )

        self.live_requests_count += 1

        # Remember the client's original streaming preference
        original_client_streaming = False
        if cached_request.body and isinstance(cached_request.body, dict):
            original_client_streaming = cached_request.body.get("stream", False)

        # Force streaming on upstream
        self._maybe_force_streaming(cached_request, original_client_streaming)

        response = await self.outflow.forward_request(cached_request)

        # Store as an appended reply to existing cassette
        model = cached_request.body.get("model") if cached_request.body else None
        temperature = cached_request.body.get("temperature") if cached_request.body else None
        p_hash = compute_prompt_hash(cached_request.body)

        entry = CacheEntry(
            request=cached_request,
            response=response,
            model=model,
            temperature=temperature,
            prompt_hash=p_hash,
            original_client_streaming=original_client_streaming,
        )
        await self.storage.put_async(entry, max_replies=index_row.max_replies, extra_metadata=extra_metadata)
        self.log.info("Appended reply to cassette %s", index_row.content_hash)

        return response

    def _serve_reply(self, index_row: IndexRow, strategy: str) -> CachedResponse:
        """
        Serve a reply from a cassette, cycling through replies for multi-reply cassettes.
        """
        reply_num = self.replay_counter.next_reply(index_row.content_hash, index_row.replies, strategy)

        # Get the (response_hash, status_code) tuples for this cassette so that
        # recorded error statuses (4xx/5xx) replay with their original HTTP code.
        reply_meta = self.storage.get_reply_metadata(index_row.content_hash)
        if not reply_meta:
            self.log.error("No response hashes found for cassette %s", index_row.content_hash)
            return CachedResponse(
                status_code=500,
                headers={"Content-Type": "application/json"},
                body={"error": {
                    "message": "Internal error: cassette has no replies",
                    "type": "internal_error"
                }},
                is_streaming=False,
            )

        # Clamp reply_num to available replies
        reply_idx = min(reply_num, len(reply_meta)) - 1
        response_hash, status_code = reply_meta[reply_idx]

        # Load response (try streaming first, fall back to non-streaming)
        cached_response = self.storage.load_response(response_hash, streaming=True, status_code=status_code)
        if cached_response is None:
            cached_response = self.storage.load_response(response_hash, streaming=False, status_code=status_code)

        if cached_response is None:
            self.log.error("Response %s not found for cassette %s reply %d", response_hash, index_row.content_hash, reply_num)
            return CachedResponse(
                status_code=500,
                headers={"Content-Type": "application/json"},
                body={"error": {
                    "message": "Internal error: response file missing",
                    "type": "internal_error"
                }},
                is_streaming=False,
            )

        self.log.debug("Serving reply %d/%d from cassette %s", reply_num, index_row.replies, index_row.content_hash)
        return cached_response

    def _maybe_force_streaming(self, cached_request: CachedRequest, original_client_streaming: bool) -> None:
        """
        Force streaming on the upstream request unless the model or path is excluded.
        """
        model_name = cached_request.body.get("model") if cached_request.body else None
        path_supports_streaming = cached_request.path.startswith(_FORCE_STREAMING_PATH_PREFIXES)
        model_supports_streaming = model_name not in self.non_streaming_models if model_name else True
        should_force_streaming = path_supports_streaming and model_supports_streaming

        if should_force_streaming and cached_request.body is not None and isinstance(cached_request.body, dict):
            cached_request.body["stream"] = True
            # Also enable stream_options.include_usage so usage data is preserved
            if "stream_options" not in cached_request.body:
                cached_request.body["stream_options"] = {"include_usage": True}
            elif isinstance(cached_request.body["stream_options"], dict):
                cached_request.body["stream_options"]["include_usage"] = True
            self.log.debug("Forced streaming=True for upstream request (client wanted streaming=%s)", original_client_streaming)
