"""
`tests_performance` exercises end-to-end replay throughput of the InferenceGate
proxy via the in-tree pytest plugin.

The single test in this module fires a fixed loop of cassette-backed
``/v1/chat/completions`` requests through the live subprocess Gate and asserts
a wall-clock floor.  It is gated by ``@pytest.mark.performance`` so a busy
laptop can deselect it with ``-m "not performance"``.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.parse

import pytest

# Cassettes bundled in ``tests/cassettes/`` were recorded against this model.
CASSETTE_MODEL = "openai/gpt-oss-120b"
CASSETTE_MAX_TOKENS = 200

# Two distinct prompts, each backed by a committed cassette.  We alternate
# between them so the in-memory replay cache (when enabled) actually demonstrates
# its lookup-by-hash characteristics and we are not just replaying one entry.
PROMPTS = [
    'This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.',
    "What is 2+2? Reply with only the number.",
]

# Request loop tuning.  100 requests is a sensible smoke target for the replay
# hot path on a developer laptop \u2014 with cassette hits it should comfortably
# clear 50 rps (i.e. <2.0 s wall time end-to-end).
REQUEST_COUNT = 100
WALL_CLOCK_BUDGET_S = 2.0


def test_default_mode_serves_or_records_performance_prompts(inference_gate_url: str) -> None:
    """
    Exercise the throughput prompts without forcing replay mode before the performance replay loop.

    Record-mode runs can use this test to recreate the two production cassettes before the throughput
    assertion pins every request to replay mode with ``X-InferenceGate-Control-Mode: replay``.
    """
    parsed = urllib.parse.urlparse(inference_gate_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)
    headers = {
        "Content-Type": "application/json",
        "Connection": "keep-alive",
    }

    try:
        for prompt in PROMPTS:
            body = json.dumps({
                "model": CASSETTE_MODEL,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
                "max_tokens": CASSETTE_MAX_TOKENS,
            })
            conn.request("POST", "/v1/chat/completions", body=body, headers=headers)
            resp = conn.getresponse()
            payload = resp.read()
            assert resp.status == 200, f"status={resp.status} body={payload[:500]!r}"
    finally:
        conn.close()


@pytest.mark.performance
def test_replay_throughput_floor(inference_gate_url: str) -> None:
    """
    Sequentially fire ``REQUEST_COUNT`` cassette-backed requests through the
    Gate and assert the entire run completes within ``WALL_CLOCK_BUDGET_S``.

    The Gate runs in replay-only mode by default; every request must hit a
    cassette and return HTTP 200.  A wall-clock breach typically means either
    a regression in the hot path (e.g. accidental disk re-read per request)
    or a cassette miss (which would also fail the per-iteration status check).
    """
    parsed = urllib.parse.urlparse(inference_gate_url)
    # Single keep-alive HTTP connection so we measure proxy throughput, not
    # TCP setup cost.  This mirrors how a real client SDK would behave.
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
    headers = {
        "Content-Type": "application/json",
        "Connection": "keep-alive",
        "X-InferenceGate-Control-Mode": "replay",
        "X-InferenceGate-Control-Reply-Strategy": "first",
    }

    started = time.monotonic()
    try:
        for index in range(REQUEST_COUNT):
            prompt = PROMPTS[index % len(PROMPTS)]
            body = json.dumps({
                "model": CASSETTE_MODEL,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
                "max_tokens": CASSETTE_MAX_TOKENS,
            })
            conn.request("POST", "/v1/chat/completions", body=body, headers=headers)
            resp = conn.getresponse()
            payload = resp.read()
            assert resp.status == 200, f"request {index}: status={resp.status} body={payload[:200]!r}"
    finally:
        conn.close()
    elapsed = time.monotonic() - started

    rps = REQUEST_COUNT / elapsed if elapsed > 0 else float("inf")
    # Surface throughput in pytest -s output for trend tracking.
    print(f"InferenceGate replay throughput: {REQUEST_COUNT} requests in {elapsed:.3f}s ({rps:.1f} rps)")
    assert elapsed < WALL_CLOCK_BUDGET_S, (f"replay loop took {elapsed:.3f}s for {REQUEST_COUNT} requests "
                                           f"(budget {WALL_CLOCK_BUDGET_S}s, ~{rps:.1f} rps)")
