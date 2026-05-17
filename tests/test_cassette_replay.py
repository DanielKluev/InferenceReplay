"""
Integration tests using production cassettes recorded from a real upstream API.

These tests verify end-to-end replay behavior by loading pre-recorded
streaming cassettes and serving them through InferenceGate in replay-only mode.
Both streaming (SSE) and non-streaming (reassembled JSON) client delivery
paths are exercised against real-world API responses.
"""

import http.client
import json
import urllib.parse
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from inference_gate.inflow.server import InflowServer
from inference_gate.modes import Mode
from inference_gate.recording.storage import CacheStorage
from inference_gate.router.router import Router

# Path to the production cassettes directory
CASSETTES_DIR = str(Path(__file__).parent / "cassettes")

# The model used when recording the production cassettes
CASSETTE_MODEL = "openai/gpt-oss-120b"

# Prompts used when recording the production cassettes
OK_PROMPT = 'This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.'
MATH_PROMPT = "What is 2+2? Reply with only the number."
CASSETTE_MAX_TOKENS = 200


@pytest.fixture
def cassette_router():
    """Create a Router in replay-only mode using production cassettes."""
    storage = CacheStorage(CASSETTES_DIR)
    return Router(mode=Mode.REPLAY_ONLY, storage=storage)


@pytest.fixture
def cassette_server(cassette_router):
    """Create an InflowServer backed by production cassettes."""
    return InflowServer(host="127.0.0.1", port=0, router=cassette_router)


@pytest.fixture
async def cassette_client(cassette_server):
    """Create an aiohttp test client backed by production cassettes."""
    app = cassette_server._create_app()
    async with TestClient(TestServer(app)) as client:
        yield client


def _post_default_mode_chat_completion(inference_gate_url: str, prompt: str) -> dict:
    """
    Send one production cassette prompt through the pytest-managed Gate without forcing replay mode.

    In replay sessions this proves the cassette is usable through the normal session mode. In record
    sessions, after a developer intentionally deletes ``tests/cassettes``, this request is allowed to
    reach the configured upstream and recreate the cassette before the replay-only tests below run.
    """
    parsed = urllib.parse.urlparse(inference_gate_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)
    try:
        body = json.dumps({
            "model": CASSETTE_MODEL,
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "max_tokens": CASSETTE_MAX_TOKENS,
        })
        conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        payload = resp.read()
        assert resp.status == 200, f"status={resp.status} body={payload[:500]!r}"
        return json.loads(payload)
    finally:
        conn.close()


class TestCassetteReplayDefaultModeSeed:
    """Tests for the production cassette prompts through the pytest-managed default mode."""

    def test_default_mode_serves_or_records_replay_prompts(self, inference_gate_url):
        """The replay-only cassette prompts are first exercised without a per-request replay override."""
        ok_data = _post_default_mode_chat_completion(inference_gate_url, OK_PROMPT)
        ok_content = ok_data["choices"][0]["message"]["content"]
        assert "OK" in ok_content

        math_data = _post_default_mode_chat_completion(inference_gate_url, MATH_PROMPT)
        math_content = math_data["choices"][0]["message"]["content"]
        assert "4" in math_content


class TestCassetteReplayNonStreaming:
    """Tests for non-streaming client responses reassembled from streaming cassettes."""

    async def test_ok_prompt_non_streaming(self, cassette_client):
        """Test that the OK prompt cassette is served as reassembled JSON with 'OK.' content."""
        resp = await cassette_client.post(
            "/v1/chat/completions",
            json={
                "model": CASSETTE_MODEL,
                "messages": [{
                    "role": "user",
                    "content": OK_PROMPT
                }],
                "max_tokens": CASSETTE_MAX_TOKENS,
            },
        )
        assert resp.status == 200
        assert resp.content_type == "application/json"
        data = await resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        content = data["choices"][0]["message"]["content"]
        assert "OK" in content

    async def test_ok_prompt_reassembled_structure(self, cassette_client):
        """Test that the reassembled JSON response has proper Chat Completion structure."""
        resp = await cassette_client.post(
            "/v1/chat/completions",
            json={
                "model": CASSETTE_MODEL,
                "messages": [{
                    "role": "user",
                    "content": OK_PROMPT
                }],
                "max_tokens": CASSETTE_MAX_TOKENS,
            },
        )
        data = await resp.json()
        # Verify standard Chat Completion response structure
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert "model" in data
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "stop"

    async def test_math_prompt_non_streaming(self, cassette_client):
        """Test that the 2+2 prompt cassette is served as reassembled JSON with '4' content."""
        resp = await cassette_client.post(
            "/v1/chat/completions",
            json={
                "model": CASSETTE_MODEL,
                "messages": [{
                    "role": "user",
                    "content": MATH_PROMPT
                }],
                "max_tokens": CASSETTE_MAX_TOKENS,
            },
        )
        assert resp.status == 200
        assert resp.content_type == "application/json"
        data = await resp.json()
        content = data["choices"][0]["message"]["content"]
        assert "4" in content


class TestCassetteReplayStreaming:
    """Tests for streaming (SSE) client responses from streaming cassettes."""

    async def test_ok_prompt_streaming(self, cassette_client):
        """Test that the OK prompt cassette is served as SSE chunks containing 'OK'."""
        resp = await cassette_client.post(
            "/v1/chat/completions",
            json={
                "model": CASSETTE_MODEL,
                "messages": [{
                    "role": "user",
                    "content": OK_PROMPT
                }],
                "max_tokens": CASSETTE_MAX_TOKENS,
                "stream": True,
            },
        )
        assert resp.status == 200
        assert resp.content_type == "text/event-stream"
        text = await resp.text()
        assert "data:" in text
        assert "[DONE]" in text
        # Parse SSE data lines and verify assembled content contains "OK"
        content_parts = _extract_sse_content(text)
        assembled = "".join(content_parts)
        assert "OK" in assembled

    async def test_math_prompt_streaming(self, cassette_client):
        """Test that the 2+2 prompt cassette is served as SSE chunks containing '4'."""
        resp = await cassette_client.post(
            "/v1/chat/completions",
            json={
                "model": CASSETTE_MODEL,
                "messages": [{
                    "role": "user",
                    "content": MATH_PROMPT
                }],
                "max_tokens": CASSETTE_MAX_TOKENS,
                "stream": True,
            },
        )
        assert resp.status == 200
        assert resp.content_type == "text/event-stream"
        text = await resp.text()
        content_parts = _extract_sse_content(text)
        assembled = "".join(content_parts)
        assert "4" in assembled

    async def test_streaming_has_done_sentinel(self, cassette_client):
        """Test that streamed SSE response ends with [DONE] sentinel."""
        resp = await cassette_client.post(
            "/v1/chat/completions",
            json={
                "model": CASSETTE_MODEL,
                "messages": [{
                    "role": "user",
                    "content": OK_PROMPT
                }],
                "max_tokens": CASSETTE_MAX_TOKENS,
                "stream": True,
            },
        )
        text = await resp.text()
        assert "data: [DONE]" in text


class TestCassetteReplayCacheMiss:
    """Tests for cache miss behavior with production cassettes in replay-only mode."""

    async def test_unknown_prompt_returns_503(self, cassette_client):
        """Test that a request with an unknown prompt returns 503 in replay-only mode."""
        resp = await cassette_client.post(
            "/v1/chat/completions",
            json={
                "model": CASSETTE_MODEL,
                "messages": [{
                    "role": "user",
                    "content": "This prompt is not in any cassette"
                }],
                "max_tokens": CASSETTE_MAX_TOKENS,
            },
        )
        assert resp.status == 503
        data = await resp.json()
        assert "error" in data
        assert "no_cached_response" in data["error"]["code"]

    async def test_different_model_returns_503(self, cassette_client):
        """Test that a request with a different model returns 503 even if the prompt matches."""
        resp = await cassette_client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{
                    "role": "user",
                    "content": OK_PROMPT
                }],
                "max_tokens": CASSETTE_MAX_TOKENS,
            },
        )
        assert resp.status == 503


class TestCassetteReplayCacheKeyNormalization:
    """Tests verifying that streaming-related fields are ignored for cache key lookup."""

    async def test_stream_true_hits_cassette(self, cassette_client):
        """Test that stream=True resolves to the same cassette as no stream field."""
        resp = await cassette_client.post(
            "/v1/chat/completions",
            json={
                "model": CASSETTE_MODEL,
                "messages": [{
                    "role": "user",
                    "content": OK_PROMPT
                }],
                "max_tokens": CASSETTE_MAX_TOKENS,
                "stream": True,
            },
        )
        assert resp.status == 200

    async def test_stream_false_hits_cassette(self, cassette_client):
        """Test that stream=False resolves to the same cassette as no stream field."""
        resp = await cassette_client.post(
            "/v1/chat/completions",
            json={
                "model": CASSETTE_MODEL,
                "messages": [{
                    "role": "user",
                    "content": OK_PROMPT
                }],
                "max_tokens": CASSETTE_MAX_TOKENS,
                "stream": False,
            },
        )
        assert resp.status == 200

    async def test_stream_options_ignored_for_cache_key(self, cassette_client):
        """Test that stream_options is excluded from cache key lookup."""
        resp = await cassette_client.post(
            "/v1/chat/completions",
            json={
                "model": CASSETTE_MODEL,
                "messages": [{
                    "role": "user",
                    "content": OK_PROMPT
                }],
                "max_tokens": CASSETTE_MAX_TOKENS,
                "stream": True,
                "stream_options": {
                    "include_usage": True
                },
            },
        )
        assert resp.status == 200


def _extract_sse_content(sse_text: str) -> list[str]:
    """
    Extract content deltas from SSE text.

    Parses `data:` lines from SSE text and extracts content fields
    from Chat Completion streaming chunks.

    Returns a list of content delta strings.
    """
    content_parts = []
    for line in sse_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data_str = line[len("data:"):].strip()
        if data_str == "[DONE]":
            continue
        try:
            event = json.loads(data_str)
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if "content" in delta and delta["content"] is not None:
                    content_parts.append(delta["content"])
        except (json.JSONDecodeError, ValueError):
            continue
    return content_parts
