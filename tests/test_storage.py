"""Tests for InferenceGate recording/storage module (v2 tape format)."""

import json
import tempfile
from pathlib import Path

import pytest

from inference_gate.recording.hashing import (compute_content_hash, compute_prompt_hash, extract_first_user_message, generate_slug)
from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry, CacheStorage
from inference_gate.recording.tape_writer import build_message_sections


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def storage(temp_cache_dir):
    """Create a cache storage instance."""
    return CacheStorage(temp_cache_dir)


class TestCacheStorage:
    """Tests for CacheStorage class (v2 tape format)."""

    def test_init_creates_directory(self, temp_cache_dir):
        """Test that CacheStorage constructor creates the cache directory and subdirectories."""
        cache_dir = Path(temp_cache_dir) / "new_cache"
        s = CacheStorage(cache_dir)
        assert cache_dir.exists()
        assert (cache_dir / "requests").exists()
        assert (cache_dir / "responses").exists()
        assert (cache_dir / "assets").exists()

    def test_put_and_get(self, storage):
        """Test that storing an entry and retrieving it returns the same data."""
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Hello"
                }],
                "temperature": 0,
            },
        )
        response = CachedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={"choices": [{
                "message": {
                    "content": "Hi there!"
                }
            }]},
        )
        entry = CacheEntry(request=request, response=response, model="gpt-4", temperature=0)

        cache_key = storage.put(entry)
        assert cache_key is not None

        retrieved = storage.get(request)
        assert retrieved is not None
        assert retrieved.model == "gpt-4"

    def test_get_nonexistent(self, storage):
        """Test that getting a non-existent entry returns None."""
        request = CachedRequest(method="GET", path="/v1/models", headers={})
        result = storage.get(request)
        assert result is None

    def test_exists(self, storage):
        """Test that exists() correctly reports whether a request is cached."""
        request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={"model": "gpt-4", "temperature": 0})

        assert not storage.exists(request)

        response = CachedResponse(status_code=200, headers={}, body={"choices": []})
        entry = CacheEntry(request=request, response=response)
        storage.put(entry)

        assert storage.exists(request)

    def test_clear(self, storage):
        """Test that clear() removes all entries and returns the correct count."""
        # Add some entries
        for i in range(3):
            request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={"model": f"model-{i}", "temperature": 0})
            response = CachedResponse(status_code=200, headers={}, body={"choices": []})
            entry = CacheEntry(request=request, response=response)
            storage.put(entry)

        count = storage.clear()
        assert count == 3
        assert storage.list_entries() == []

    def test_list_entries(self, storage):
        """Test that list_entries() returns all stored entries."""
        # Add entries
        for i in range(2):
            request = CachedRequest(method="POST", path=f"/v1/path-{i}", headers={}, body={"temperature": 0})
            response = CachedResponse(status_code=200, headers={}, body={"choices": []})
            entry = CacheEntry(request=request, response=response)
            storage.put(entry)

        entries = storage.list_entries()
        assert len(entries) == 2

    def test_compute_prompt_hash(self):
        """Test that prompt hash is deterministic and different for different inputs."""
        messages = [
            {
                "role": "system",
                "content": "You are helpful."
            },
            {
                "role": "user",
                "content": "Hello"
            },
        ]
        hash1 = CacheStorage.compute_prompt_hash(messages)
        hash2 = CacheStorage.compute_prompt_hash(messages)
        assert hash1 == hash2

        different_messages = [{"role": "user", "content": "Different"}]
        hash3 = CacheStorage.compute_prompt_hash(different_messages)
        assert hash1 != hash3

    def test_streaming_response(self, storage):
        """Test that streaming responses with chunks are stored and retrieved correctly."""
        request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={
            "model": "gpt-4",
            "stream": True,
            "temperature": 0
        })
        response = CachedResponse(
            status_code=200,
            headers={},
            chunks=[
                "data: {\"choices\":[{\"delta\":{\"content\":\"chunk1\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"chunk2\"}}]}\n\n", "data: [DONE]\n\n"
            ],
            is_streaming=True,
        )
        entry = CacheEntry(request=request, response=response)
        storage.put(entry)

        retrieved = storage.get(request)
        assert retrieved is not None

    def test_cache_key_ignores_stream_field(self, storage):
        """Test that stream=True and stream=False requests produce the same content hash."""
        body_streaming = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "stream": True}
        body_non_streaming = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "stream": False}
        body_no_stream = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}

        key1 = compute_content_hash("POST", "/v1/chat/completions", body_streaming)
        key2 = compute_content_hash("POST", "/v1/chat/completions", body_non_streaming)
        key3 = compute_content_hash("POST", "/v1/chat/completions", body_no_stream)

        assert key1 == key2, "stream=True and stream=False should produce the same cache key"
        assert key1 == key3, "stream=True and no stream field should produce the same cache key"

    def test_cache_key_differs_for_different_content(self, storage):
        """Test that different prompt content produces different content hashes."""
        body1 = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}], "stream": True}
        body2 = {"model": "gpt-4", "messages": [{"role": "user", "content": "Goodbye"}], "stream": True}

        key1 = compute_content_hash("POST", "/v1/chat/completions", body1)
        key2 = compute_content_hash("POST", "/v1/chat/completions", body2)

        assert key1 != key2

    def test_streaming_request_hits_non_streaming_cache(self, storage):
        """Test that a streaming request can find a cassette stored by a non-streaming request (same content hash)."""
        # Store with stream=True and stream_options (as recorded by forced-streaming router)
        request_stored = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Test"
                }],
                "stream": True,
                "stream_options": {
                    "include_usage": True
                },
                "temperature": 0,
            },
        )
        response = CachedResponse(status_code=200, headers={},
                                  chunks=["data: {\"choices\":[{\"delta\":{\"content\":\"chunk\"}}]}\n\n",
                                          "data: [DONE]\n\n"], is_streaming=True)
        entry = CacheEntry(request=request_stored, response=response)
        storage.put(entry)

        # Look up with stream=False body (as the non-streaming client would send)
        request_lookup = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Test"
                }],
                "stream": False,
                "temperature": 0,
            },
        )
        retrieved = storage.get(request_lookup)
        assert retrieved is not None

        # Also look up with NO stream field at all (as a plain client would send)
        request_plain = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Test"
                }],
                "temperature": 0,
            },
        )
        retrieved2 = storage.get(request_plain)
        assert retrieved2 is not None

    def test_cache_key_ignores_stream_options_field(self, storage):
        """Test that stream_options field is excluded from cache key computation."""
        body_with = {
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": "Hello"
            }],
            "stream": True,
            "stream_options": {
                "include_usage": True
            },
        }
        body_without = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}

        key_with = compute_content_hash("POST", "/v1/chat/completions", body_with)
        key_without = compute_content_hash("POST", "/v1/chat/completions", body_without)

        assert key_with == key_without, "stream_options should be excluded from cache key"

    def test_return_token_ids_changes_content_hash_but_not_prompt_hash(self):
        """
        Test that `return_token_ids` (vLLM extension) is deliberately part of `content_hash`.

        `return_token_ids: true` and the absent/false setting must produce *different* content
        hashes because the extension changes the response shape (adds `prompt_token_ids` /
        `token_ids`).  The `prompt_hash` (model-fuzzy, messages-only) should still match across
        the two so fuzzy lookup can find compatible cassettes when callers don't care about
        token IDs.
        """
        messages = [{"role": "user", "content": "Hello"}]
        body_without = {"model": "gpt-4", "messages": messages, "temperature": 0}
        body_with = {"model": "gpt-4", "messages": messages, "temperature": 0, "return_token_ids": True}

        content_without = compute_content_hash("POST", "/v1/chat/completions", body_without)
        content_with = compute_content_hash("POST", "/v1/chat/completions", body_with)
        prompt_without = compute_prompt_hash(body_without)
        prompt_with = compute_prompt_hash(body_with)

        assert content_without != content_with, "return_token_ids must distinguish content hashes"
        assert prompt_without == prompt_with, "return_token_ids should not affect model-fuzzy prompt_hash"

    def test_response_with_prompt_token_ids_round_trips_verbatim(self, storage):
        """
        Test that responses carrying `prompt_token_ids` (vLLM extension) survive put -> get.

        The storage layer must not strip unknown response fields; `prompt_token_ids` enables
        token-level correctness assertions in downstream tests (pyFADE Gemma4 correctness).
        """
        request = CachedRequest(
            method="POST",
            path="/v1/chat/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Hi"
                }],
                "temperature": 0,
                "return_token_ids": True,
            },
        )
        prompt_token_ids = [1, 2, 3, 4, 5]
        response = CachedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={
                "choices": [{
                    "message": {
                        "content": "Hello!"
                    },
                    "token_ids": [10, 11]
                }],
                "prompt_token_ids": prompt_token_ids,
            },
        )
        entry = CacheEntry(request=request, response=response, model="gpt-4", temperature=0)
        storage.put(entry)

        retrieved = storage.get(request)
        assert retrieved is not None
        assert retrieved.response.body is not None
        assert retrieved.response.body.get("prompt_token_ids") == prompt_token_ids
        assert retrieved.response.body["choices"][0].get("token_ids") == [10, 11]

    def test_text_completion_streaming_preserves_text(self, storage, temp_cache_dir):
        """Test that a streaming /v1/completions response reassembles with non-empty text.

        Regression: previously `_store_response` and `_build_reply_info` passed an
        empty request_path to the reassembler, causing the dispatcher to fall back
        to Chat Completions shape which discards the `choices[*].text` field and
        records an empty body.  The stored JSON must now carry the text.
        """
        request = CachedRequest(method="POST", path="/v1/completions", headers={}, body={
            "model": "gpt-4",
            "prompt": "Hello",
            "stream": True,
            "temperature": 0,
        })
        response = CachedResponse(
            status_code=200,
            headers={},
            chunks=[
                'data: {"id":"cmpl-1","object":"text_completion","created":1700000000,"model":"gpt-4","choices":[{"index":0,"text":"Hel","finish_reason":null}]}\n\n',
                'data: {"id":"cmpl-1","object":"text_completion","created":1700000000,"model":"gpt-4","choices":[{"index":0,"text":"lo","finish_reason":"stop"}]}\n\n',
                'data: [DONE]\n\n',
            ],
            is_streaming=True,
        )
        entry = CacheEntry(request=request, response=response, model="gpt-4")
        storage.put(entry)

        # The reassembled JSON body must be persisted with object=text_completion
        # and non-empty text (not coerced to an empty chat.completion).
        responses_dir = Path(temp_cache_dir) / "responses"
        json_files = list(responses_dir.glob("*.json"))
        assert json_files, "Expected at least one reassembled JSON response file."
        reassembled = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert reassembled["object"] == "text_completion", \
            f"Expected text_completion, got {reassembled.get('object')!r} - request path was not threaded through."
        assert reassembled["choices"][0]["text"] == "Hello", \
            f"Expected concatenated text 'Hello', got {reassembled['choices'][0].get('text')!r}"

    def test_streaming_ndjson_stores_complete_sse_events(self, storage, temp_cache_dir):
        """Streaming sidecars are written from parsed SSE events, not raw network lines.

        Regression: raw network chunks can split a single ``data:`` event in
        the middle of a tool-call argument string.  Storing each raw split line
        as NDJSON corrupts replay because the missing fragment never reaches
        the reassembler.
        """
        from inference_gate.recording.reassembly import reassemble_chat_completion
        request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": "call a tool",
            }],
            "stream": True,
        })
        response = CachedResponse(
            status_code=200,
            headers={},
            chunks=[
                'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4",'
                '"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_abc",'
                '"type":"function","function":{"name":"lookup","arguments":""}}]},"finish_reason":null}]}\n\n',
                'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4",'
                '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"route_label\\": '
                '\\"openrouter-deepseek-tool-con',
                'tract\\", \\"hops\\": 7}"}}]},"finish_reason":null}]}\n\n',
                'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4",'
                '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
                'data: [DONE]\n\n',
            ],
            is_streaming=True,
        )
        entry = CacheEntry(request=request, response=response, model="gpt-4")
        storage.put(entry)

        ndjson_files = list((Path(temp_cache_dir) / "responses").glob("*.chunks.ndjson"))
        assert len(ndjson_files) == 1
        lines = [line for line in ndjson_files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
        assert all(json.loads(line) for line in lines)

        replay_chunks = [f"data: {line}\n\n" for line in lines]
        replayed = reassemble_chat_completion(replay_chunks)
        arguments = replayed["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        assert json.loads(arguments) == {"route_label": "openrouter-deepseek-tool-contract", "hops": 7}

    def test_original_client_streaming_metadata(self, storage):
        """Test that original_client_streaming metadata is stored in CacheEntry."""
        request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={
            "model": "gpt-4",
            "stream": True,
            "temperature": 0
        })
        response = CachedResponse(status_code=200, headers={},
                                  chunks=["data: {\"choices\":[{\"delta\":{\"content\":\"test\"}}]}\n\n",
                                          "data: [DONE]\n\n"], is_streaming=True)
        entry = CacheEntry(request=request, response=response, original_client_streaming=False)
        storage.put(entry)

        retrieved = storage.get(request)
        assert retrieved is not None
        # original_client_streaming is a CacheEntry transport field, not persisted in tape

    def test_index_lookup_by_prompt_hash(self, storage):
        """Test that the index supports lookup by prompt hash."""
        messages = [{"role": "user", "content": "Hello fuzzy"}]
        prompt_hash = CacheStorage.compute_prompt_hash(messages)
        request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={
            "model": "gpt-4",
            "messages": messages,
            "temperature": 0
        })
        response = CachedResponse(status_code=200, headers={}, body={"choices": [{"message": {"content": "Hi!"}}]})
        entry = CacheEntry(request=request, response=response, model="gpt-4", prompt_hash=prompt_hash)
        storage.put(entry)

        rows = storage.index.by_prompt_hash.get(prompt_hash, [])
        assert len(rows) >= 1
        assert rows[0].model == "gpt-4"

    def test_index_lookup_by_prompt_hash_not_found(self, storage):
        """Test that index lookup by nonexistent prompt hash returns empty list."""
        rows = storage.index.by_prompt_hash.get("nonexistenthash", [])
        assert rows == []

    def test_index_lookup_ignores_model(self, storage):
        """Test that prompt hash lookup finds entries regardless of model used."""
        messages = [{"role": "user", "content": "Hello from model A"}]
        request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={
            "model": "model-a",
            "messages": messages,
            "temperature": 0
        })
        response = CachedResponse(status_code=200, headers={}, body={"choices": [{"message": {"content": "Response from A"}}]})
        entry = CacheEntry(request=request, response=response, model="model-a")
        storage.put(entry)

        prompt_hash = CacheStorage.compute_prompt_hash(messages)
        rows = storage.index.by_prompt_hash.get(prompt_hash, [])
        assert len(rows) >= 1
        assert rows[0].model == "model-a"

    def test_multi_reply_append(self, storage):
        """Test that putting the same request twice appends a reply to the existing tape."""
        request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={
            "model": "gpt-4",
            "messages": [{
                "role": "user",
                "content": "Multi"
            }],
            "temperature": 0.7
        })
        response1 = CachedResponse(status_code=200, headers={}, body={"choices": [{"message": {"content": "Reply 1"}}]})
        response2 = CachedResponse(status_code=200, headers={}, body={"choices": [{"message": {"content": "Reply 2"}}]})

        entry1 = CacheEntry(request=request, response=response1, model="gpt-4")
        entry2 = CacheEntry(request=request, response=response2, model="gpt-4")

        content_hash = storage.put(entry1, max_replies=5)
        storage.put(entry2, max_replies=5)

        # Index should show 2 replies
        row = storage.index.by_content_hash.get(content_hash)
        assert row is not None
        assert row.replies == 2

        # Both reply response hashes should be gettable
        hashes = storage.get_reply_response_hashes(content_hash)
        assert len(hashes) == 2

    def test_reindex(self, storage):
        """Test that reindex() rebuilds the index from tape files."""
        # Store an entry
        request = CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={"model": "gpt-4", "temperature": 0})
        response = CachedResponse(status_code=200, headers={}, body={"choices": []})
        entry = CacheEntry(request=request, response=response)
        storage.put(entry)

        # Wipe and rebuild index
        count = storage.reindex()
        assert count == 1


def _make_entry(model: str = "gpt-4", message: str = "Hello", temperature: float = 0) -> CacheEntry:
    """Helper to create a simple entry for testing."""
    return CacheEntry(
        request=CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={
            "model": model,
            "messages": [{
                "role": "user",
                "content": message
            }],
            "temperature": temperature
        }),
        response=CachedResponse(status_code=200, headers={}, body={"choices": [{
            "message": {
                "content": f"Reply to: {message}"
            }
        }]}),
        model=model,
        temperature=temperature,
    )


class TestDeleteEntry:
    """Tests for CacheStorage.delete_entry()."""

    def test_delete_existing_entry(self, storage):
        """Test that deleting an existing cassette removes tape, responses, and index entry."""
        entry = _make_entry(message="delete me")
        content_hash = storage.put(entry)

        assert content_hash in storage.index
        assert storage.delete_entry(content_hash) is True
        assert content_hash not in storage.index
        # Tape file should be gone
        assert storage._find_tape_file(content_hash) is None

    def test_delete_nonexistent_entry(self, storage):
        """Test that deleting a non-existent cassette returns False."""
        assert storage.delete_entry("nonexistent123") is False

    def test_delete_removes_response_files(self, storage):
        """Test that deleting a cassette also removes its response JSON/NDJSON files."""
        entry = _make_entry(message="delete responses")
        content_hash = storage.put(entry)

        # Get response hashes before delete
        response_hashes = storage.get_reply_response_hashes(content_hash)
        assert len(response_hashes) > 0

        storage.delete_entry(content_hash)
        for resp_hash in response_hashes:
            assert not (storage.responses_dir / f"{resp_hash}.json").exists()


class TestResolvePrefix:
    """Tests for CacheStorage.resolve_prefix()."""

    def test_exact_match(self, storage):
        """Test that an exact content_hash returns exactly one match."""
        entry = _make_entry(message="prefix test")
        content_hash = storage.put(entry)
        matches = storage.resolve_prefix(content_hash)
        assert len(matches) == 1
        assert matches[0].content_hash == content_hash

    def test_prefix_match(self, storage):
        """Test that a short prefix resolves to the correct entry."""
        entry = _make_entry(message="prefix short")
        content_hash = storage.put(entry)
        # Use first 6 chars as prefix
        matches = storage.resolve_prefix(content_hash[:6])
        assert any(m.content_hash == content_hash for m in matches)

    def test_no_match(self, storage):
        """Test that a non-matching prefix returns empty list."""
        matches = storage.resolve_prefix("zzzzzzzzzzzz")
        assert matches == []


class TestSearchEntries:
    """Tests for CacheStorage.search_entries()."""

    def test_search_by_message(self, storage):
        """Test searching by user message content."""
        storage.put(_make_entry(message="The quick brown fox"))
        storage.put(_make_entry(message="Lazy dog sleeps"))
        storage.put(_make_entry(message="Quick rabbit jumps"))

        results = storage.search_entries("quick")
        assert len(results) == 2

    def test_search_with_model_filter(self, storage):
        """Test search with additional model filter."""
        storage.put(_make_entry(model="gpt-4", message="Hello from gpt"))
        storage.put(_make_entry(model="claude-3", message="Hello from claude"))

        results = storage.search_entries("hello", model="gpt")
        assert len(results) == 1
        assert results[0].model == "gpt-4"

    def test_search_limit(self, storage):
        """Test that search respects the limit parameter."""
        for i in range(10):
            storage.put(_make_entry(message=f"Search target {i}", model=f"model-{i}"))

        results = storage.search_entries("search target", limit=3)
        assert len(results) == 3

    def test_search_no_results(self, storage):
        """Test that search returns empty list when nothing matches."""
        storage.put(_make_entry(message="Something unrelated"))
        results = storage.search_entries("nonexistent query")
        assert results == []


class TestFilterEntries:
    """Tests for CacheStorage.filter_entries()."""

    def test_filter_by_model(self, storage):
        """Test filtering by model name."""
        storage.put(_make_entry(model="gpt-4", message="A"))
        storage.put(_make_entry(model="claude-3", message="B"))
        storage.put(_make_entry(model="gpt-4-turbo", message="C"))

        results = storage.filter_entries(model="gpt")
        assert len(results) == 2

    def test_filter_by_greedy(self, storage):
        """Test filtering by greedy flag (temperature=0 is greedy)."""
        storage.put(_make_entry(temperature=0, message="Greedy"))
        storage.put(_make_entry(temperature=0.7, message="Non-greedy"))

        greedy = storage.filter_entries(greedy=True)
        assert len(greedy) == 1
        assert greedy[0].is_greedy is True

        non_greedy = storage.filter_entries(greedy=False)
        assert len(non_greedy) == 1
        assert non_greedy[0].is_greedy is False

    def test_filter_with_limit(self, storage):
        """Test that filter respects the limit parameter."""
        for i in range(5):
            storage.put(_make_entry(message=f"Limit test {i}", model=f"model-{i}"))

        results = storage.filter_entries(limit=2)
        assert len(results) == 2

    def test_filter_all_returns_all(self, storage):
        """Test that filtering with no criteria returns all entries."""
        storage.put(_make_entry(message="One"))
        storage.put(_make_entry(message="Two"))
        results = storage.filter_entries()
        assert len(results) == 2


class TestGetDiskSize:
    """Tests for CacheStorage.get_disk_size()."""

    def test_empty_cache_size(self, storage):
        """Test that an empty cache has zero (or minimal) disk size."""
        size = storage.get_disk_size()
        assert size == 0

    def test_size_increases_after_put(self, storage):
        """Test that disk size increases after storing an entry."""
        storage.put(_make_entry(message="Size test"))
        size = storage.get_disk_size()
        assert size > 0


class TestReconstructRequestBody:
    """Tests for CacheStorage.reconstruct_request_body()."""

    def test_basic_reconstruction(self, storage):
        """Test that a simple chat completion request body is reconstructed from tape."""
        entry = _make_entry(model="gpt-4", message="Hello world", temperature=0.7)
        content_hash = storage.put(entry)

        body = storage.reconstruct_request_body(content_hash)
        assert body is not None
        assert body["model"] == "gpt-4"
        assert body["temperature"] == 0.7
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == "Hello world"

    def test_reconstruction_with_system_message(self, storage):
        """Test reconstruction of request with system + user messages."""
        entry = CacheEntry(
            request=CachedRequest(
                method="POST", path="/v1/chat/completions", headers={}, body={
                    "model": "gpt-4",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are helpful."
                        },
                        {
                            "role": "user",
                            "content": "Hi"
                        },
                    ],
                    "temperature": 0.5,
                }),
            response=CachedResponse(status_code=200, headers={}, body={"choices": [{
                "message": {
                    "content": "Hello!"
                }
            }]}),
            model="gpt-4",
        )
        content_hash = storage.put(entry)

        body = storage.reconstruct_request_body(content_hash)
        assert body is not None
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "You are helpful."
        assert body["messages"][1]["role"] == "user"
        assert body["messages"][1]["content"] == "Hi"

    def test_reconstruction_with_sampling_params(self, storage):
        """Test that sampling parameters (top_p, top_k, etc.) are reconstructed."""
        entry = CacheEntry(
            request=CachedRequest(
                method="POST", path="/v1/chat/completions", headers={}, body={
                    "model": "test-model",
                    "messages": [{
                        "role": "user",
                        "content": "Test"
                    }],
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_tokens": 200,
                }),
            response=CachedResponse(status_code=200, headers={}, body={"choices": [{
                "message": {
                    "content": "OK"
                }
            }]}),
            model="test-model",
        )
        content_hash = storage.put(entry)

        body = storage.reconstruct_request_body(content_hash)
        assert body is not None
        assert body["temperature"] == 0.8
        assert body["top_p"] == 0.9
        assert body["top_k"] == 40
        assert body["max_tokens"] == 200

    def test_reconstruction_with_tools(self, storage):
        """Test that tool definitions are reconstructed from the tape."""
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
        entry = CacheEntry(
            request=CachedRequest(
                method="POST", path="/v1/chat/completions", headers={}, body={
                    "model": "gpt-4",
                    "messages": [{
                        "role": "user",
                        "content": "Weather?"
                    }],
                    "tools": tools,
                    "temperature": 0.5,
                }),
            response=CachedResponse(status_code=200, headers={}, body={"choices": [{
                "message": {
                    "content": "Sunny"
                }
            }]}),
            model="gpt-4",
        )
        content_hash = storage.put(entry)

        body = storage.reconstruct_request_body(content_hash)
        assert body is not None
        assert "tools" in body
        assert body["tools"] == tools

    def test_reconstruction_nonexistent(self, storage):
        """Test that reconstruct_request_body returns None for a nonexistent cassette."""
        body = storage.reconstruct_request_body("nonexistent123")
        assert body is None

    def test_reconstruction_with_stop_sequences(self, storage):
        """Test that stop sequences are reconstructed."""
        entry = CacheEntry(
            request=CachedRequest(
                method="POST", path="/v1/chat/completions", headers={}, body={
                    "model": "gpt-4",
                    "messages": [{
                        "role": "user",
                        "content": "Test"
                    }],
                    "stop": ["\n", "END"],
                    "temperature": 0.5,
                }),
            response=CachedResponse(status_code=200, headers={}, body={"choices": [{
                "message": {
                    "content": "OK"
                }
            }]}),
            model="gpt-4",
        )
        content_hash = storage.put(entry)

        body = storage.reconstruct_request_body(content_hash)
        assert body is not None
        assert body["stop"] == ["\n", "END"]


class TestUpdateMaxReplies:
    """Tests for CacheStorage._update_max_replies()."""

    def test_update_max_replies(self, storage):
        """Test that max_replies is updated in tape metadata and index."""
        entry = _make_entry(model="gpt-4", message="Update max", temperature=0.7)
        content_hash = storage.put(entry, max_replies=5)

        row = storage.index.by_content_hash.get(content_hash)
        assert row.max_replies == 5

        storage._update_max_replies(content_hash, 10)

        # Verify index is updated
        row = storage.index.by_content_hash.get(content_hash)
        assert row.max_replies == 10

        # Verify tape metadata is updated
        tape_data = storage.load_tape(content_hash)
        assert tape_data is not None
        metadata, _ = tape_data
        assert metadata.max_replies == 10


# ===========================================================================
# Tests for raw Completions API prompt field support
# ===========================================================================


class TestRawPromptSupport:
    """
    Tests for raw Completions API (``/v1/completions``, ``/completion``) prompt handling.

    Ensures that requests with a ``prompt`` field (pre-formatted by chat template)
    produce meaningful hashes, slugs, user messages, and tape sections — not empty values.
    """

    def test_compute_prompt_hash_raw_prompt(self):
        """Test that compute_prompt_hash returns a non-null hash for raw prompt bodies."""
        body = {"prompt": "<user>\nWhat is 2+2?\n<assistant>\n", "model": "test"}
        empty_body_hash = compute_prompt_hash(None)
        prompt_hash = compute_prompt_hash(body)
        assert prompt_hash != empty_body_hash

    def test_compute_prompt_hash_raw_prompt_deterministic(self):
        """Test that compute_prompt_hash is deterministic for the same raw prompt."""
        body = {"prompt": "Hello, world!", "model": "test"}
        h1 = compute_prompt_hash(body)
        h2 = compute_prompt_hash(body)
        assert h1 == h2

    def test_compute_prompt_hash_different_prompts_differ(self):
        """Test that different raw prompts produce different hashes."""
        body_a = {"prompt": "What is 2+2?", "model": "test"}
        body_b = {"prompt": "What is 3+3?", "model": "test"}
        assert compute_prompt_hash(body_a) != compute_prompt_hash(body_b)

    def test_compute_prompt_hash_raw_vs_messages_differ(self):
        """Test that a raw prompt hash differs from a messages-based hash with the same text."""
        raw_body = {"prompt": "What is 2+2?"}
        messages_body = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        # Different formats should never cross-match at tier 3
        assert compute_prompt_hash(raw_body) != compute_prompt_hash(messages_body)

    def test_generate_slug_raw_prompt(self):
        """Test that generate_slug extracts text from raw prompt field."""
        body = {"prompt": "<user>\nHello world\n<assistant>", "model": "test"}
        slug = generate_slug(body)
        assert slug != ""
        assert "user" in slug or "hello" in slug  # Should contain some text from prompt

    def test_generate_slug_prefers_messages_over_prompt(self):
        """Test that messages field takes priority over prompt field when both present."""
        body = {"messages": [{"role": "user", "content": "From messages"}], "prompt": "From prompt"}
        slug = generate_slug(body)
        assert "from-messages" == slug

    def test_extract_first_user_message_raw_prompt(self):
        """Test that extract_first_user_message returns text from raw prompt field."""
        body = {"prompt": "Tell me about cats", "model": "test"}
        msg = extract_first_user_message(body)
        assert msg == "Tell me about cats"

    def test_extract_first_user_message_raw_prompt_sanitizes(self):
        """Test that newlines and tabs in raw prompts are sanitized for TSV."""
        body = {"prompt": "Line one\nLine two\tTabbed", "model": "test"}
        msg = extract_first_user_message(body)
        assert "\n" not in msg
        assert "\t" not in msg
        assert "Line one Line two Tabbed" == msg

    def test_extract_first_user_message_truncates(self):
        """Test that raw prompts are truncated to max_length."""
        body = {"prompt": "x" * 200, "model": "test"}
        msg = extract_first_user_message(body, max_length=50)
        assert len(msg) == 50

    def test_build_message_sections_raw_prompt(self):
        """Test that build_message_sections creates a USER section for raw prompt."""
        body = {"prompt": "<user>Hello</user>", "model": "test"}
        sections = build_message_sections(body, "abc123")
        assert len(sections) == 1
        assert sections[0].kind.value == "user"
        assert sections[0].body == "<user>Hello</user>"

    def test_build_message_sections_prefers_messages(self):
        """Test that messages field takes priority over prompt in build_message_sections."""
        body = {"messages": [{"role": "user", "content": "From messages"}], "prompt": "From prompt"}
        sections = build_message_sections(body, "abc123")
        assert len(sections) == 1
        assert sections[0].body == "From messages"

    def test_build_message_sections_token_id_prompt(self):
        """
        vLLM accepts pre-tokenized prompts on ``/v1/completions`` as ``list[int]``.

        Verify the recording layer renders them into a non-empty USER section
        instead of silently dropping the prompt (which produced empty tapes).
        """
        body = {"prompt": [105, 9731, 107, 98, 106], "model": "test"}
        sections = build_message_sections(body, "abc123")
        assert len(sections) == 1
        assert sections[0].kind.value == "user"
        assert "token_ids" in sections[0].body
        assert "[105, 9731, 107, 98, 106]" in sections[0].body

    def test_extract_first_user_message_token_id_prompt(self):
        """Token-id prompts should still produce a non-empty TSV preview."""
        body = {"prompt": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "model": "test"}
        msg = extract_first_user_message(body)
        assert msg.startswith("token_ids[10]:")
        assert "1,2,3,4,5,6,7,8" in msg
        assert msg.endswith("...")

    def test_build_message_sections_assistant_tool_calls_and_tool_results(self):
        """
        Multi-turn tool-calling conversations must render the assistant's prior
        ``tool_calls`` and the matching ``tool`` result messages as their own
        tape sections.  Without this, two prompts that differ only in tool-call
        history (the typical LLM-loop case) become byte-identical in the tape
        body even though the underlying request bodies differ — making cassettes
        unreadable for debugging and indistinguishable to a human reviewer.
        """
        body = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What is the weather?"
                },
                {
                    "role":
                        "assistant",
                    "content":
                        "",
                    "tool_calls": [{
                        "id": "call_001",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}'
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_001",
                    "name": "get_weather",
                    "content": "22C sunny"
                },
                {
                    "role": "user",
                    "content": "Thanks!"
                },
            ]
        }
        sections = build_message_sections(body, "abc123")
        kinds = [s.kind.value for s in sections]
        # system, user, assistant_tool_call, tool_result, user
        assert kinds == ["system", "user", "assistant_tool_call", "tool_result", "user"]
        atc = sections[2]
        assert atc.header == "assistant tool_call get_weather call_001"
        assert atc.body == '{"city":"Paris"}'
        assert atc.tool_name == "get_weather"
        assert atc.tool_call_id == "call_001"
        tr = sections[3]
        assert tr.header == "tool call_001"
        assert tr.metadata.get("Name") == "get_weather"
        assert tr.body == "22C sunny"
        assert tr.tool_call_id == "call_001"

    def test_tape_round_trip_assistant_tool_call_and_tool_result(self):
        """
        Writer→parser round-trip preserves ``assistant tool_call`` and ``tool``
        sections so replay-time consumers can reconstruct the prompt history.
        """
        from inference_gate.recording.models import SectionKind, TapeMetadata
        from inference_gate.recording.tape_parser import parse_tape
        from inference_gate.recording.tape_writer import write_tape

        body = {
            "messages": [
                {
                    "role": "user",
                    "content": "Hi"
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "tc_42",
                        "type": "function",
                        "function": {
                            "name": "ping",
                            "arguments": '{"x":1}'
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tc_42",
                    "name": "ping",
                    "content": "pong"
                },
            ]
        }
        boundary = "deadbe"
        sections = build_message_sections(body, boundary)
        meta = TapeMetadata(boundary=boundary, model="test", endpoint="/v1/chat/completions")
        tape_text = write_tape(meta, sections)
        parsed_meta, parsed_sections = parse_tape(tape_text)
        assert parsed_meta.boundary == boundary
        kinds = [s.kind for s in parsed_sections]
        assert SectionKind.ASSISTANT_TOOL_CALL in kinds
        assert SectionKind.TOOL_RESULT in kinds
        atc = next(s for s in parsed_sections if s.kind == SectionKind.ASSISTANT_TOOL_CALL)
        assert atc.tool_name == "ping"
        assert atc.tool_call_id == "tc_42"
        assert atc.body == '{"x":1}'
        tr = next(s for s in parsed_sections if s.kind == SectionKind.TOOL_RESULT)
        assert tr.tool_call_id == "tc_42"
        assert tr.metadata.get("Name") == "ping"
        assert tr.body == "pong"

    def test_generate_slug_token_id_prompt(self):
        """Token-id prompts should produce a non-empty slug."""
        body = {"prompt": [105, 9731, 107], "model": "test"}
        slug = generate_slug(body)
        assert slug != ""
        assert "token-ids" in slug

    def test_put_and_get_raw_completions(self, storage):
        """Test that a raw /v1/completions request is stored with prompt in tape and index."""
        request = CachedRequest(
            method="POST",
            path="/v1/completions",
            headers={"content-type": "application/json"},
            body={
                "model": "test-model",
                "prompt": "<user>\nWrite about cats\n<assistant>\n",
                "max_tokens": 128,
                "temperature": 0.7,
            },
        )
        response = CachedResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body={"choices": [{
                "text": "Cats are wonderful creatures."
            }]},
        )
        entry = CacheEntry(request=request, response=response, model="test-model", temperature=0.7)
        content_hash = storage.put(entry)

        # Verify index entry has non-empty slug and first_user_message
        row = storage.index.by_content_hash.get(content_hash)
        assert row is not None
        assert row.slug != ""
        assert row.first_user_message != ""
        assert row.endpoint == "/v1/completions"
        assert "cats" in row.first_user_message.lower() or "user" in row.first_user_message.lower()

        # Verify the prompt_hash is not the null hash
        null_hash = compute_prompt_hash(None)
        assert row.prompt_hash != null_hash

        # Verify tape file contains the raw prompt as a user section
        tape_data = storage.load_tape(content_hash)
        assert tape_data is not None
        metadata, sections = tape_data
        user_sections = [s for s in sections if s.kind.value == "user"]
        assert len(user_sections) == 1
        assert "cats" in user_sections[0].body.lower()

    def test_raw_completions_prompt_hash_enables_fuzzy_match(self, storage):
        """Test that two raw prompt requests with the same prompt but different sampling share prompt_hash."""
        body_base = {"model": "test", "prompt": "Hello raw prompt", "max_tokens": 100}

        request1 = CachedRequest(method="POST", path="/v1/completions", headers={}, body={**body_base, "temperature": 0.5})
        request2 = CachedRequest(method="POST", path="/v1/completions", headers={}, body={**body_base, "temperature": 0.9})

        response = CachedResponse(status_code=200, headers={}, body={"choices": [{"text": "OK"}]})
        storage.put(CacheEntry(request=request1, response=response, model="test"))

        # Both should share the same prompt_hash
        hash1 = compute_prompt_hash(request1.body)
        hash2 = compute_prompt_hash(request2.body)
        assert hash1 == hash2

        # prompt_hash lookup should find the stored entry
        rows = storage.index.by_prompt_hash.get(hash1, [])
        assert len(rows) == 1


# ===========================================================================
# Tests for endpoint column in index.tsv
# ===========================================================================


class TestEndpointColumn:
    """
    Tests for the ``endpoint`` column in index.tsv.

    Ensures that the request path (e.g. ``/v1/chat/completions``,
    ``/v1/completions``) is stored in the index for each cassette.
    """

    def test_endpoint_stored_chat_completions(self, storage):
        """Test that /v1/chat/completions entries have endpoint in index."""
        entry = _make_entry(model="gpt-4", message="Endpoint test")
        content_hash = storage.put(entry)

        row = storage.index.by_content_hash.get(content_hash)
        assert row is not None
        assert row.endpoint == "/v1/chat/completions"

    def test_endpoint_stored_raw_completions(self, storage):
        """Test that /v1/completions entries have endpoint in index."""
        request = CachedRequest(method="POST", path="/v1/completions", headers={}, body={
            "model": "test",
            "prompt": "Hello",
            "temperature": 0,
        })
        response = CachedResponse(status_code=200, headers={}, body={"choices": [{"text": "Hi"}]})
        entry = CacheEntry(request=request, response=response, model="test")
        content_hash = storage.put(entry)

        row = storage.index.by_content_hash.get(content_hash)
        assert row is not None
        assert row.endpoint == "/v1/completions"

    def test_endpoint_in_tsv_roundtrip(self, storage):
        """Test that endpoint column survives TSV write/read cycle."""
        entry = _make_entry(model="gpt-4", message="TSV roundtrip")
        content_hash = storage.put(entry)

        # Read the TSV directly
        tsv_text = (storage.cache_dir / "index.tsv").read_text(encoding="utf-8")
        assert "endpoint" in tsv_text.split("\n")[0]  # Header contains endpoint
        assert "/v1/chat/completions" in tsv_text

        # Reload storage from disk and verify
        storage2 = CacheStorage(storage.cache_dir)
        row = storage2.index.by_content_hash.get(content_hash)
        assert row is not None
        assert row.endpoint == "/v1/chat/completions"

    def test_endpoint_in_reindex(self, storage):
        """Test that endpoint is populated during reindex from tape frontmatter."""
        entry = _make_entry(model="gpt-4", message="Reindex endpoint")
        content_hash = storage.put(entry)

        # Wipe and rebuild index
        storage.reindex()

        row = storage.index.by_content_hash.get(content_hash)
        assert row is not None
        assert row.endpoint == "/v1/chat/completions"
