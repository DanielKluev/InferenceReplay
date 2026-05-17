"""
Storage layer for caching API calls (v2 tape format).

Uses a directory structure with human-readable `.tape` cassette files
(YAML frontmatter + MIME body), content-addressed response files, and
a TSV index for fast multi-tier lookup.

Directory layout:
```
{cache_dir}/
  index.tsv
  requests/{content_hash}__{slug}.tape
  responses/{hash}.json
  responses/{hash}.chunks.ndjson
  assets/{hash}.{ext}
```

Key classes: `CacheStorage`, `CachedRequest`, `CachedResponse`, `CacheEntry`
"""

import asyncio
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from inference_gate.recording.atomic_io import atomic_write_text
from inference_gate.recording.hashing import (compute_content_hash, compute_prompt_hash, compute_prompt_model_hash, compute_response_hash,
                                              extract_first_user_message, generate_slug)
from inference_gate.recording.models import IndexRow, ReplyInfo, SamplingParams, SectionKind, TapeMetadata, extract_sampling_params
from inference_gate.recording.tape_index import TapeIndex
from inference_gate.recording.tape_parser import parse_tape
from inference_gate.recording.tape_writer import (append_reply_to_tape, build_message_sections, build_reply_section, generate_boundary,
                                                  write_tape, write_tape_to_file)


class CachedRequest(BaseModel):
    """
    Cached request data.

    Transport model used by the router and inflow/outflow layers.
    """

    method: str
    path: str
    headers: dict[str, str]
    body: dict[str, Any] | None = None
    query_params: dict[str, str] | None = None


class CachedResponse(BaseModel):
    """
    Cached response data.

    Transport model carrying either a JSON body or streaming chunks.
    """

    status_code: int
    headers: dict[str, str]
    body: dict[str, Any] | None = None
    chunks: list[str] | None = None  # For streaming responses
    is_streaming: bool = False


class CacheEntry(BaseModel):
    """
    A single cache entry with request/response pair.

    Transport model used between router and storage. The storage layer
    persists these as tape files + response files on disk.
    """

    request: CachedRequest
    response: CachedResponse
    model: str | None = None
    temperature: float | None = None
    prompt_hash: str | None = None
    original_client_streaming: bool | None = None


class CacheStorage:
    """
    File-based storage for API call caching using the v2 tape format.

    Manages a directory containing:
    - `requests/` — human-readable `.tape` cassette files
    - `responses/` — content-addressed response JSON / NDJSON files
    - `assets/` — content-addressed binary files (images, audio)
    - `index.tsv` — TSV index for fast lookup

    Supports multi-reply cassettes for non-greedy sampling and tiered
    fuzzy lookup via the `TapeIndex`.
    """

    # Headers stripped from stored cassettes to prevent leaking credentials.
    _SANITIZED_HEADERS = {"authorization", "x-api-key", "proxy-authorization"}

    def __init__(self, cache_dir: str | Path = ".inference_cache") -> None:
        """
        Initialize cache storage with v2 directory structure.

        Creates the `requests/`, `responses/`, and `assets/` subdirectories
        and loads (or creates) the `index.tsv`.

        `cache_dir` is the root directory for all storage.
        """
        self.log = logging.getLogger("CacheStorage")
        self.cache_dir = Path(cache_dir)
        self.requests_dir = self.cache_dir / "requests"
        self.responses_dir = self.cache_dir / "responses"
        self.assets_dir = self.cache_dir / "assets"
        # ``requests_raw/`` is a debug-only mirror of every incoming request as JSON
        # keyed by content_hash.  Unlike the human-readable ``.tape`` cassettes in
        # ``requests/`` (which only exist post-record), this directory is written
        # on EVERY request (replay or record) so a developer can correlate exactly
        # what the upstream model received with the cassette miss/hit outcome.
        self.requests_raw_dir = self.cache_dir / "requests_raw"

        # Create directory structure
        for d in (self.cache_dir, self.requests_dir, self.responses_dir, self.assets_dir, self.requests_raw_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Load or create the index
        self.index = TapeIndex(self.cache_dir / "index.tsv")

        # Async coordination primitives.  ``_index_lock`` serialises mutations to
        # the in-memory index and the on-disk TSV.  ``_content_locks`` provides
        # per-content-hash locks so concurrent identical record requests do not
        # double-record (one wins, others append/replay).  Both are created
        # lazily so that synchronous code paths (CLI, tools) do not require an
        # asyncio loop.
        self._index_lock: asyncio.Lock | None = None
        self._content_locks: dict[str, asyncio.Lock] = {}
        self._content_locks_lock: asyncio.Lock | None = None

        # Migrate any v1 tapes to v2 format (adds explicit Status: 200 and bumps tape_version).
        # Runs before any reindex so that the rebuilt index picks up the new columns.
        migrated = self._migrate_v1_tapes()

        # Auto-rebuild if the index was written by an older schema version,
        # or if tapes were just migrated (columns may be stale).
        if self.index._needs_rebuild or migrated > 0:
            self.log.info("Rebuilding index to match current schema (migrated=%d)", migrated)
            self.index.rebuild(self.requests_dir)
            self.index._needs_rebuild = False

    def dump_raw_request(self, *, method: str, path: str, headers: dict[str, str], body: dict[str, Any] | None,
                         query_params: dict[str, str] | None, content_hash: str, prompt_model_hash: str | None = None,
                         prompt_hash: str | None = None, outcome: str | None = None, extra: dict[str, Any] | None = None) -> Path | None:
        """
        Persist a verbatim JSON dump of an incoming request under
        ``requests_raw/<content_hash>.json`` for debugging.

        This is a write-once-per-content-hash debug mirror: when the file
        already exists we skip rewriting it (the body is content-addressed
        so identical requests produce identical files).  Sensitive headers
        listed in :attr:`_SANITIZED_HEADERS` are dropped before persisting.

        ``outcome`` is an optional label (``"hit"`` / ``"miss"`` / ``"forwarded"``)
        and ``extra`` is a free-form dict for any per-request annotation
        (engine routing decisions, fuzzy-match details, etc.).

        Returns the resulting path on success, or ``None`` when the dump
        could not be written (errors are swallowed and logged at WARNING
        level — a debug aid must never break the live request path).
        """
        try:
            target = self.requests_raw_dir / f"{content_hash}.json"
            scrubbed_headers = {k: v for k, v in headers.items() if k.lower() not in self._SANITIZED_HEADERS}
            payload: dict[str, Any] = {
                "content_hash": content_hash,
                "prompt_model_hash": prompt_model_hash,
                "prompt_hash": prompt_hash,
                "method": method,
                "path": path,
                "query_params": query_params or {},
                "headers": scrubbed_headers,
                "body": body,
            }
            if outcome is not None:
                payload["outcome"] = outcome
            if extra:
                payload["extra"] = extra
            # Atomic write: we don't want partial files on crash.  The file is
            # rewritten in-place each call so the latest ``outcome`` / ``extra``
            # for a given content_hash is preserved (handy when iterating in
            # record mode).
            tmp = target.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            tmp.replace(target)
            return target
        except Exception as exc:  # pylint: disable=broad-except
            self.log.warning("Failed to dump raw request for content_hash=%s: %s", content_hash, exc)
            return None

    def get(self, request: CachedRequest) -> CacheEntry | None:
        """
        Look up a cached response by exact content hash.

        Computes the content hash from the request, looks it up in the index,
        loads the tape file and response data, and returns a `CacheEntry`.

        Returns None on cache miss.
        """
        content_hash = compute_content_hash(request.method, request.path, request.body)
        index_row = self.index.by_content_hash.get(content_hash)
        if index_row is None:
            return None

        return self._load_entry(content_hash, request)

    def get_by_hashes(self, content_hash: str | None = None, prompt_model_hash: str | None = None,
                      prompt_hash: str | None = None) -> IndexRow | None:
        """
        Look up an index row by any of the three hash tiers.

        Tries exact match first, then prompt_model_hash, then prompt_hash.
        Returns the first matching `IndexRow`, or None.
        """
        if content_hash:
            row = self.index.by_content_hash.get(content_hash)
            if row is not None:
                return row

        if prompt_model_hash:
            rows = self.index.by_prompt_model_hash.get(prompt_model_hash, [])
            if rows:
                return rows[0]

        if prompt_hash:
            rows = self.index.by_prompt_hash.get(prompt_hash, [])
            if rows:
                return rows[0]

        return None

    def put(self, entry: CacheEntry, max_replies: int = 1, extra_metadata: dict[str, str] | None = None) -> str:
        """
        Store a new cache entry as a tape file + response files.

        Creates the tape with the first reply. If a tape for this request
        already exists (same content hash), appends the reply instead.

        ``extra_metadata`` carries the parsed ``X-InferenceGate-Metadata-*``
        fields (engine, engine_version, test_node_id, worker_id, recorded_by)
        which are written into ``TapeMetadata.metadata`` for new cassettes.
        For appended replies the value is currently ignored — first-write wins
        for cassette-level metadata.

        Returns the content hash used as the cassette key.
        """
        body = entry.request.body
        method = entry.request.method
        path = entry.request.path

        # Compute all three hash tiers
        content_hash = compute_content_hash(method, path, body)
        pm_hash = compute_prompt_model_hash(method, path, body)
        p_hash = compute_prompt_hash(body)

        # Check if this cassette already exists (append reply)
        existing_row = self.index.by_content_hash.get(content_hash)
        if existing_row is not None:
            return self._append_reply(content_hash, entry, existing_row)

        # Store response files
        response_hash, has_stream = self._store_response(entry.response, request_path=path)

        # Extract metadata from request
        sampling = extract_sampling_params(body) if body else SamplingParams()
        slug = generate_slug(body)
        first_user_msg = extract_first_user_message(body)

        # Build reply info
        reply_info = self._build_reply_info(1, response_hash, has_stream, entry.response, request_path=entry.request.path)

        # Build tape metadata
        tools_summary = []
        if body and body.get("tools"):
            for tool in body["tools"]:
                if isinstance(tool, dict):
                    fname = tool.get("function", {}).get("name", "") if isinstance(tool.get("function"), dict) else ""
                    tools_summary.append(fname or tool.get("name", ""))

        # Gather all text content that will appear in the tape for boundary collision check.
        # Include reasoning content since it is written as its own section body.
        message_sections = build_message_sections(body, "")  # boundary placeholder
        all_text = "\n".join(s.body for s in message_sections if s.body)
        all_text += "\n" + reply_info.text
        if reply_info.reasoning:
            all_text += "\n" + reply_info.reasoning
        boundary = generate_boundary(all_text)

        metadata = TapeMetadata(
            tape_version=2,
            content_hash=content_hash,
            prompt_model_hash=pm_hash,
            prompt_hash=p_hash,
            model=entry.model,
            endpoint=path,
            sampling=sampling,
            max_tokens=body.get("max_tokens") if body else None,
            stop_sequences=body.get("stop") or body.get("stop_sequences") or [] if body else [],
            tools=tools_summary,
            tool_choice=body.get("tool_choice") if body else None,
            logprobs=bool(body.get("logprobs")) if body else False,
            top_logprobs=body.get("top_logprobs") if body else None,
            recorded=datetime.now(timezone.utc),
            replies=1,
            max_replies=max_replies,
            status_code=entry.response.status_code,
            boundary=boundary,
            metadata=dict(extra_metadata) if extra_metadata else {},
        )

        # Build sections with correct boundary
        sections = build_message_sections(body, boundary)
        sections.extend(build_reply_section(reply_info))

        # Write tape file
        tape_filename = f"{content_hash}__{slug}.tape" if slug else f"{content_hash}.tape"
        tape_path = self.requests_dir / tape_filename
        write_tape_to_file(tape_path, metadata, sections)

        # Update index
        index_row = IndexRow.from_tape_metadata(
            metadata,
            slug=slug,
            first_user_message=first_user_msg,
            tokens_in=str(reply_info.input_tokens or ""),
            tokens_out=str(reply_info.output_tokens or ""),
            has_reasoning=bool(reply_info.reasoning),
        )
        self.index.add(index_row)

        self.log.info("Stored new cassette %s (model=%s, replies=1, status=%d)", content_hash, entry.model, entry.response.status_code)
        return content_hash

    async def put_async(self, entry: CacheEntry, max_replies: int = 1, extra_metadata: dict[str, str] | None = None) -> str:
        """
        Async wrapper around :meth:`put` that serialises concurrent record-mode
        writes via two layers of locking:

        - A per-``content_hash`` :class:`asyncio.Lock` so two coroutines issuing
          the same request collapse to one tape mutation (the second one sees
          the cassette already exists and falls into the append-reply path or
          replays, depending on caller behaviour).
        - The shared index lock so the in-memory ``TapeIndex`` and on-disk TSV
          stay consistent under concurrent appends from different cassettes.

        File I/O remains synchronous (see ``recording.atomic_io``); only the
        critical section is awaited.  Once aiofiles is adopted this wrapper is
        the place to switch to await-able I/O.
        """
        content_hash = compute_content_hash(entry.request.method, entry.request.path, entry.request.body)
        per_hash_lock = await self._get_content_lock(content_hash)
        async with per_hash_lock:
            index_lock = self._ensure_index_lock()
            async with index_lock:
                return self.put(entry, max_replies=max_replies, extra_metadata=extra_metadata)

    def _ensure_index_lock(self) -> asyncio.Lock:
        """
        Lazily create and return the index lock bound to the running event loop.
        """
        if self._index_lock is None:
            self._index_lock = asyncio.Lock()
        return self._index_lock

    async def _get_content_lock(self, content_hash: str) -> asyncio.Lock:
        """
        Return the per-``content_hash`` lock, creating it on first access.

        Acquisition of the registry lock is itself protected so that two
        coroutines hitting the same brand-new content_hash receive the same
        lock object.
        """
        if self._content_locks_lock is None:
            self._content_locks_lock = asyncio.Lock()
        async with self._content_locks_lock:
            lock = self._content_locks.get(content_hash)
            if lock is None:
                lock = asyncio.Lock()
                self._content_locks[content_hash] = lock
            return lock

    def _append_reply(self, content_hash: str, entry: CacheEntry, existing_row: IndexRow) -> str:
        """
        Append a new reply to an existing tape file.

        Stores the response files, adds a reply section to the tape,
        and updates the index.
        """
        # Store response files
        response_hash, has_stream = self._store_response(entry.response, request_path=entry.request.path)

        # Find the tape file
        tape_path = self._find_tape_file(content_hash)
        if tape_path is None:
            self.log.error("Tape file for %s not found during append", content_hash)
            # Fall through to create a new tape
            return self.put(entry, max_replies=existing_row.max_replies)

        new_reply_count = existing_row.replies + 1
        reply_info = self._build_reply_info(new_reply_count, response_hash, has_stream, entry.response, request_path=entry.request.path)

        # Append to tape file
        append_reply_to_tape(tape_path, reply_info, new_reply_count)

        # Update index
        self.index.update_replies(
            content_hash,
            new_reply_count,
            tokens_in=str(reply_info.input_tokens or ""),
            tokens_out=str(reply_info.output_tokens or ""),
            has_reasoning=bool(reply_info.reasoning),
        )

        self.log.info("Appended reply %d to cassette %s", new_reply_count, content_hash)
        return content_hash

    def _update_max_replies(self, content_hash: str, new_max_replies: int) -> None:
        """
        Update the `max_replies` field in a cassette's tape frontmatter and index.

        Rewrites the tape file with updated metadata and updates the index row.
        """
        tape_path = self._find_tape_file(content_hash)
        if tape_path is None:
            self.log.warning("Cannot update max_replies: tape file for %s not found", content_hash)
            return

        result = self.load_tape(content_hash)
        if result is None:
            return

        metadata, sections = result
        metadata.max_replies = new_max_replies
        write_tape_to_file(tape_path, metadata, sections)

        # Update index
        row = self.index.by_content_hash.get(content_hash)
        if row is not None:
            updated = row.model_copy(update={"max_replies": new_max_replies})
            self.index._remove_row(row)
            self.index._add_row_to_indexes(updated)
            self.index._write_tsv()

        self.log.info("Updated max_replies to %d for cassette %s", new_max_replies, content_hash)

    def load_response(self, response_hash: str, streaming: bool = False, status_code: int = 200) -> CachedResponse | None:
        """
        Load a response from the `responses/` directory by hash.

        If `streaming` is True and a `.chunks.ndjson` file exists, loads
        SSE chunks. Otherwise loads the `.json` response body.  The
        ``status_code`` argument is attached to the returned ``CachedResponse``
        so that recorded non-200 error responses replay with their original
        HTTP status (v1 tapes without an explicit status default to 200).

        Returns a `CachedResponse` or None if the file is not found.
        """
        if streaming:
            ndjson_path = self.responses_dir / f"{response_hash}.chunks.ndjson"
            if ndjson_path.exists():
                chunks = ndjson_path.read_text(encoding="utf-8").splitlines()
                # Re-wrap as SSE data lines
                sse_chunks = [f"data: {line}\n\n" for line in chunks if line.strip()]
                sse_chunks.append("data: [DONE]\n\n")
                return CachedResponse(
                    status_code=status_code,
                    headers={"Content-Type": "text/event-stream"},
                    chunks=sse_chunks,
                    is_streaming=True,
                )

        json_path = self.responses_dir / f"{response_hash}.json"
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            return CachedResponse(
                status_code=status_code,
                headers={"Content-Type": "application/json"},
                body=data,
                is_streaming=False,
            )

        return None

    def load_tape(self, content_hash: str) -> tuple[TapeMetadata, list] | None:
        """
        Load and parse a tape file by content hash.

        Returns a tuple of (`TapeMetadata`, list of `TapeSection`) or None.
        """
        tape_path = self._find_tape_file(content_hash)
        if tape_path is None:
            return None
        content = tape_path.read_text(encoding="utf-8")
        return parse_tape(content)

    def get_reply_response_hashes(self, content_hash: str) -> list[str]:
        """
        Get the response hashes for all replies in a cassette.

        Returns a list of response hash strings, ordered by reply number.
        """
        result = self.load_tape(content_hash)
        if result is None:
            return []

        _, sections = result
        hashes = []
        for section in sections:
            if section.kind == SectionKind.REPLY:
                resp = section.metadata.get("Response", "")
                # Strip .json extension if present
                resp_hash = resp.removesuffix(".json")
                if resp_hash:
                    hashes.append(resp_hash)
        return hashes

    def get_reply_metadata(self, content_hash: str) -> list[tuple[str, int]]:
        """
        Get (response_hash, status_code) tuples for all replies in a cassette.

        Used when the caller needs to faithfully replay the original HTTP status
        (including 4xx/5xx errors).  Tapes predating the ``Status:`` header
        default to 200 per backward-compatibility rules.

        Returns a list of tuples ordered by reply number.
        """
        result = self.load_tape(content_hash)
        if result is None:
            return []

        _, sections = result
        meta_list: list[tuple[str, int]] = []
        for section in sections:
            if section.kind == SectionKind.REPLY:
                resp = section.metadata.get("Response", "")
                resp_hash = resp.removesuffix(".json")
                if not resp_hash:
                    continue
                status_str = section.metadata.get("Status", "200")
                try:
                    status_code = int(status_str)
                except ValueError:
                    status_code = 200
                meta_list.append((resp_hash, status_code))
        return meta_list

    def exists(self, request: CachedRequest) -> bool:
        """
        Check if a request has a cached cassette.
        """
        content_hash = compute_content_hash(request.method, request.path, request.body)
        return content_hash in self.index

    def clear(self) -> int:
        """
        Clear all cached entries (tapes, responses, assets, index).

        Returns the number of tape files removed.
        """
        count = 0
        for tape_file in self.requests_dir.glob("*.tape"):
            tape_file.unlink()
            count += 1
        # Clear responses and assets
        for resp_file in self.responses_dir.iterdir():
            resp_file.unlink()
        for asset_file in self.assets_dir.iterdir():
            asset_file.unlink()
        # Reset index
        self.index = TapeIndex(self.cache_dir / "index.tsv")
        index_path = self.cache_dir / "index.tsv"
        if index_path.exists():
            index_path.unlink()
        self.index = TapeIndex(index_path)
        return count

    def list_entries(self) -> list[tuple[str, CacheEntry]]:
        """
        List all cached entries.

        Returns a list of (content_hash, CacheEntry) tuples.
        For the CacheEntry, only metadata fields are populated; the full
        response body is not loaded (use `load_response()` for that).
        """
        entries = []
        for row in self.index._rows:
            # Build a lightweight CacheEntry from index data
            entry = CacheEntry(
                request=CachedRequest(method="POST", path="", headers={}, body=None),
                response=CachedResponse(status_code=200, headers={}, is_streaming=False),
                model=row.model,
                temperature=float(row.temperature) if row.temperature else None,
                prompt_hash=row.prompt_hash,
            )
            entries.append((row.content_hash, entry))
        return entries

    def reconstruct_request_body(self, content_hash: str) -> dict[str, Any] | None:
        """
        Reconstruct the original request body from a tape file.

        Rebuilds the Chat Completions / Responses API body from the tape's
        YAML metadata (model, sampling, max_tokens, tools, etc.) and MIME
        body sections (messages). Returns None if the tape cannot be loaded.

        Note: multimodal attachments (images) may lose data if the URL was
        truncated during recording.
        """
        result = self.load_tape(content_hash)
        if result is None:
            return None

        metadata, sections = result
        body: dict[str, Any] = {}

        # Model and endpoint
        if metadata.model:
            body["model"] = metadata.model

        # Reconstruct messages from sections
        messages: list[dict[str, Any]] = []
        for section in sections:
            if section.kind == SectionKind.SYSTEM:
                messages.append({"role": "system", "content": section.body})
            elif section.kind == SectionKind.USER:
                messages.append({"role": "user", "content": section.body})
            elif section.kind == SectionKind.USER_ATTACHMENT:
                # Try to reconstruct multimodal content; attach to previous user message or create new one
                attach_type = section.metadata.get("Type", "")
                url = section.metadata.get("URL", "")
                if attach_type == "image" and url:
                    part = {"type": "image_url", "image_url": {"url": url}}
                    if messages and messages[-1].get("role") == "user":
                        # Convert last user message to multimodal if needed
                        prev = messages[-1]
                        if isinstance(prev["content"], str):
                            prev["content"] = [{"type": "text", "text": prev["content"]}, part]
                        elif isinstance(prev["content"], list):
                            prev["content"].append(part)
                    else:
                        messages.append({"role": "user", "content": [part]})
            elif section.kind == SectionKind.ASSISTANT_PREFILL:
                messages.append({"role": "assistant", "content": section.body})
            elif section.kind == SectionKind.TOOLS:
                # Tools section body is JSON
                try:
                    body["tools"] = json.loads(section.body)
                except (json.JSONDecodeError, ValueError):
                    self.log.warning("Failed to parse tools JSON in cassette %s", content_hash)

        if messages:
            body["messages"] = messages

        # Sampling parameters
        if metadata.sampling.temperature is not None:
            body["temperature"] = metadata.sampling.temperature
        if metadata.sampling.top_p is not None:
            body["top_p"] = metadata.sampling.top_p
        if metadata.sampling.top_k is not None:
            body["top_k"] = metadata.sampling.top_k
        if metadata.sampling.min_p is not None:
            body["min_p"] = metadata.sampling.min_p
        if metadata.sampling.repetition_penalty is not None:
            body["repetition_penalty"] = metadata.sampling.repetition_penalty
        if metadata.sampling.frequency_penalty is not None:
            body["frequency_penalty"] = metadata.sampling.frequency_penalty
        if metadata.sampling.presence_penalty is not None:
            body["presence_penalty"] = metadata.sampling.presence_penalty
        if metadata.sampling.seed is not None:
            body["seed"] = metadata.sampling.seed

        # Other request parameters
        if metadata.max_tokens is not None:
            body["max_tokens"] = metadata.max_tokens
        if metadata.stop_sequences:
            body["stop"] = metadata.stop_sequences
        if metadata.tool_choice is not None:
            body["tool_choice"] = metadata.tool_choice
        if metadata.logprobs:
            body["logprobs"] = True
        if metadata.top_logprobs is not None:
            body["top_logprobs"] = metadata.top_logprobs

        return body

    def reindex(self) -> int:
        """
        Rebuild the index from tape files on disk.

        Returns the number of entries indexed.
        """
        self.index.rebuild(self.requests_dir)
        return len(self.index)

    def resolve_prefix(self, prefix: str) -> list[IndexRow]:
        """
        Resolve a content_hash prefix to matching IndexRow(s).

        Supports both exact 12-char hashes and shorter prefixes (like git short hashes).
        Returns all rows whose `content_hash` starts with `prefix`.
        """
        prefix_lower = prefix.lower()
        # Fast path: exact match
        exact = self.index.by_content_hash.get(prefix_lower)
        if exact is not None:
            return [exact]
        # Prefix search
        return [row for row in self.index._rows if row.content_hash.startswith(prefix_lower)]

    def delete_entry(self, content_hash: str) -> bool:
        """
        Delete a single cassette by content_hash.

        Removes the tape file from `requests/`, all associated response files
        from `responses/`, and removes the entry from the TSV index.

        Returns True if the entry was found and deleted, False otherwise.
        """
        # Must find tape to discover response hashes before deleting
        tape_path = self._find_tape_file(content_hash)
        if tape_path is None:
            self.log.warning("Cannot delete: tape file for %s not found", content_hash)
            return False

        # Get response hashes before deleting
        response_hashes = self.get_reply_response_hashes(content_hash)

        # Delete response files
        for resp_hash in response_hashes:
            json_path = self.responses_dir / f"{resp_hash}.json"
            ndjson_path = self.responses_dir / f"{resp_hash}.chunks.ndjson"
            if json_path.exists():
                json_path.unlink()
            if ndjson_path.exists():
                ndjson_path.unlink()

        # Delete tape file
        tape_path.unlink()

        # Remove from index
        self.index.remove(content_hash)

        self.log.info("Deleted cassette %s (%d response files)", content_hash, len(response_hashes))
        return True

    def search_entries(self, query: str, model: str | None = None, limit: int = 20) -> list[IndexRow]:
        """
        Search cassettes by substring match on `first_user_message` and `slug`.

        Case-insensitive matching. Optionally filters by model name (substring).
        Returns up to `limit` matching IndexRows.
        """
        query_lower = query.lower()
        results: list[IndexRow] = []
        for row in self.index._rows:
            if query_lower in row.first_user_message.lower() or query_lower in row.slug.lower():
                if model is None or model.lower() in row.model.lower():
                    results.append(row)
                    if len(results) >= limit:
                        break
        return results

    def filter_entries(self, model: str | None = None, greedy: bool | None = None, has_tools: bool | None = None,
                       has_logprobs: bool | None = None, after: str | None = None, before: str | None = None, sort_by: str = "recorded",
                       limit: int | None = None) -> list[IndexRow]:
        """
        Filter and sort index rows by various criteria.

        `model` filters by substring match on model name (case-insensitive).
        `greedy` filters by is_greedy flag (True = only greedy, False = only non-greedy).
        `has_tools` filters by has_tool_use flag.
        `has_logprobs` filters by has_logprobs flag.
        `after` / `before` filter by recorded timestamp (ISO 8601 prefix match).
        `sort_by` is the field name to sort by: `recorded`, `model`, `tokens_in`, `tokens_out`.
        `limit` caps the number of results returned.

        Returns a list of matching IndexRows.
        """
        rows = list(self.index._rows)

        if model is not None:
            model_lower = model.lower()
            rows = [r for r in rows if model_lower in r.model.lower()]

        if greedy is not None:
            rows = [r for r in rows if r.is_greedy == greedy]

        if has_tools is not None:
            rows = [r for r in rows if r.has_tool_use == has_tools]

        if has_logprobs is not None:
            rows = [r for r in rows if r.has_logprobs == has_logprobs]

        if after is not None:
            rows = [r for r in rows if r.recorded >= after]

        if before is not None:
            rows = [r for r in rows if r.recorded < before]

        # Sort
        sort_key_map = {
            "recorded": lambda r: r.recorded,
            "model": lambda r: r.model.lower(),
            "tokens_in": lambda r: int(r.tokens_in) if r.tokens_in.isdigit() else 0,
            "tokens_out": lambda r: int(r.tokens_out) if r.tokens_out.isdigit() else 0,
        }
        key_fn = sort_key_map.get(sort_by, sort_key_map["recorded"])
        rows.sort(key=key_fn, reverse=(sort_by == "recorded"))

        if limit is not None:
            rows = rows[:limit]

        return rows

    def get_disk_size(self) -> int:
        """
        Calculate the total disk size of the cache directory in bytes.

        Walks `requests/`, `responses/`, `assets/`, and `index.tsv`.
        """
        total = 0
        for dirpath in (self.requests_dir, self.responses_dir, self.assets_dir):
            if dirpath.exists():
                for f in dirpath.iterdir():
                    if f.is_file():
                        total += f.stat().st_size
        index_path = self.cache_dir / "index.tsv"
        if index_path.exists():
            total += index_path.stat().st_size
        return total

    @staticmethod
    def compute_prompt_hash(messages: list[dict[str, Any]]) -> str:
        """
        Compute a hash for prompt messages (convenience static method).

        Delegates to the hashing module.
        """
        return compute_prompt_hash({"messages": messages})

    # ---- Internal helpers ----

    def _store_response(self, response: CachedResponse, request_path: str = "") -> tuple[str, bool]:
        """
        Store a response to the `responses/` directory.

        Handles both streaming (chunks → NDJSON) and non-streaming (body → JSON).
        Computes a content hash for deduplication.

        `request_path` is the originating HTTP path (e.g. ``/v1/completions``) used
        by the reassembly dispatcher to pick the correct reassembler for streaming
        responses.  Passing an empty string falls back to Chat Completions shape,
        which silently drops text for ``/v1/completions`` streams — callers should
        always thread the real path through.

        Returns a tuple of (response_hash, has_stream).
        """
        has_stream = False

        if response.is_streaming and response.chunks:
            # Reassemble the response to get a JSON body for hashing and storage
            from inference_gate.recording.reassembly import parse_sse_events, reassemble_streaming_response
            reassembled = reassemble_streaming_response(response.chunks, request_path)
            if reassembled:
                response_hash = compute_response_hash(reassembled)
            else:
                # Fallback: hash the raw chunks
                import hashlib
                raw = "".join(response.chunks)
                response_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

            # Store NDJSON chunks (stripped SSE framing)
            ndjson_path = self.responses_dir / f"{response_hash}.chunks.ndjson"
            if not ndjson_path.exists():
                # ``response.chunks`` are raw network chunks, not guaranteed
                # to align with SSE event boundaries.  Parse complete events
                # before writing NDJSON so replay never sees partial JSON
                # fragments as standalone ``data:`` events.
                lines = [json.dumps(event, ensure_ascii=False) for event in parse_sse_events(response.chunks)]
                atomic_write_text(ndjson_path, "\n".join(lines) + "\n")
            has_stream = True

            # Store reassembled JSON
            json_path = self.responses_dir / f"{response_hash}.json"
            if not json_path.exists() and reassembled:
                atomic_write_text(json_path, json.dumps(reassembled, ensure_ascii=False))

        elif response.body is not None:
            response_hash = compute_response_hash(response.body)
            json_path = self.responses_dir / f"{response_hash}.json"
            if not json_path.exists():
                atomic_write_text(json_path, json.dumps(response.body, ensure_ascii=False))
        else:
            # Empty response (e.g. error with no body)
            import hashlib
            response_hash = hashlib.sha256(b"empty").hexdigest()[:12]

        return response_hash, has_stream

    def _build_reply_info(self, reply_number: int, response_hash: str, has_stream: bool, response: CachedResponse,
                          request_path: str = "") -> ReplyInfo:
        """
        Build a `ReplyInfo` from a response, extracting text, reasoning, and token counts.

        Faithfully records the upstream HTTP status code (including 4xx/5xx error codes)
        via `response.status_code`.  Reasoning content (from models that expose
        chain-of-thought such as DeepSeek-R1, gpt-oss, Qwen3-thinking) is extracted
        from `message.reasoning_content` / `message.reasoning` and stored as a
        separate field for readability in tapes.

        `request_path` is threaded through to the streaming reassembler so that
        ``/v1/completions`` (text completion) responses produce a body with
        populated ``choices[*].text`` instead of being coerced to an empty
        chat-completion shell.
        """
        text = ""
        reasoning = ""
        stop_reason = None
        input_tokens = None
        output_tokens = None
        tool_calls: list[tuple[str, str, str]] = []

        # Try to extract info from JSON body or reassembled response
        body = response.body
        if body is None and response.is_streaming and response.chunks:
            from inference_gate.recording.reassembly import reassemble_streaming_response
            body = reassemble_streaming_response(response.chunks, request_path)

        if body and isinstance(body, dict):
            # Chat Completions format
            choices = body.get("choices", [])
            if choices:
                choice = choices[0]
                message = choice.get("message", {})
                text = message.get("content") or ""
                # Text-completion (/v1/completions) shape: no `message`, text is on the choice directly.
                if not text and choice.get("text"):
                    text = choice["text"]
                # Reasoning content key varies between providers; prefer reasoning_content,
                # then reasoning (DeepSeek-R1 and a few others use the shorter form).
                reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
                stop_reason = choice.get("finish_reason")

                # Extract tool calls
                for tc in message.get("tool_calls", []):
                    fn = tc.get("function", {})
                    tool_calls.append((
                        fn.get("name", ""),
                        tc.get("id", ""),
                        fn.get("arguments", "{}"),
                    ))

            # Responses API format
            if not choices and body.get("output"):
                outputs = body["output"]
                for item in outputs:
                    if isinstance(item, dict) and item.get("type") == "message":
                        for content in item.get("content", []):
                            if isinstance(content, dict) and content.get("type") == "output_text":
                                text = content.get("text", "")
                stop_reason = body.get("status")

            # Usage
            usage = body.get("usage", {})
            if usage:
                input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
                output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")

        return ReplyInfo(
            reply_number=reply_number,
            response_hash=response_hash,
            has_stream=has_stream,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            status_code=response.status_code,
            text=text,
            reasoning=reasoning,
            tool_calls=tool_calls,
        )

    def _find_tape_file(self, content_hash: str) -> Path | None:
        """
        Find the tape file for a given content hash.

        Tape filenames are `{content_hash}__{slug}.tape` or `{content_hash}.tape`.
        """
        # Check index for slug
        matches = list(self.requests_dir.glob(f"{content_hash}*.tape"))
        if matches:
            return matches[0]
        return None

    def _migrate_v1_tapes(self) -> int:
        """
        Migrate any ``tape_version=1`` tape files in-place to the v2 format.

        The v2 format adds an explicit HTTP ``status_code`` in YAML frontmatter and
        a per-reply ``Status:`` metadata header.  Since v1 tapes predate error
        recording, every reply is stamped ``status_code=200`` (assumed-successful).
        The tape body is otherwise left unchanged — no reasoning is synthesized
        since v1 tapes never captured reasoning content.

        Returns the number of tapes migrated.  Runs lazily on first ``CacheStorage``
        open and is a no-op for directories that already contain only v2 tapes.
        """
        migrated = 0
        for tape_path in self.requests_dir.glob("*.tape"):
            try:
                content = tape_path.read_text(encoding="utf-8")
                metadata, sections = parse_tape(content)
            except (ValueError, OSError) as exc:
                self.log.warning("Skipping unreadable tape %s during migration: %s", tape_path.name, exc)
                continue

            if metadata.tape_version >= 2:
                continue

            # Bump version, stamp primary status_code (assumed 200 for pre-existing tapes).
            metadata.tape_version = 2
            metadata.status_code = 200

            # Inject Status: 200 into every reply section that is missing it.  Preserve
            # any other existing metadata verbatim (Response, Stream, Stop-Reason, etc.).
            for section in sections:
                if section.kind == SectionKind.REPLY and "Status" not in section.metadata:
                    section.metadata["Status"] = "200"

            try:
                write_tape_to_file(tape_path, metadata, sections)
            except OSError as exc:
                self.log.warning("Failed to rewrite %s during migration: %s", tape_path.name, exc)
                continue
            migrated += 1

        if migrated > 0:
            self.log.info("Migrated %d tape(s) from v1 to v2 in %s", migrated, self.requests_dir)
        return migrated

    def _load_entry(self, content_hash: str, request: CachedRequest | None = None) -> CacheEntry | None:
        """
        Load a full CacheEntry from tape + response files.

        Loads the first reply's response by default.
        """
        result = self.load_tape(content_hash)
        if result is None:
            return None

        metadata, sections = result

        # Find first reply section's response hash + status
        reply_meta = self.get_reply_metadata(content_hash)
        if not reply_meta:
            return None

        # Load the first reply's response (caller can request specific via load_response)
        response_hash, status_code = reply_meta[0]
        cached_response = self.load_response(response_hash, streaming=True, status_code=status_code)

        # Fall back to non-streaming if no NDJSON
        if cached_response is None:
            cached_response = self.load_response(response_hash, streaming=False, status_code=status_code)

        if cached_response is None:
            self.log.warning("Response %s not found for cassette %s", response_hash, content_hash)
            return None

        # Build transport CacheEntry
        if request is None:
            request = CachedRequest(method="POST", path=metadata.endpoint, headers={}, body=None)

        return CacheEntry(
            request=request,
            response=cached_response,
            model=metadata.model,
            temperature=metadata.sampling.temperature,
            prompt_hash=metadata.prompt_hash,
            original_client_streaming=None,
        )
