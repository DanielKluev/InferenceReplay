# Pytest Integration Guide

InferenceGate ships a built-in pytest plugin that automatically starts a proxy server within your test session and provides fixtures to route your AI client through it. Tests run against cached responses (cassettes) by default — no API keys, no network calls, fully deterministic.

## Quick Start

### 1. Install

```bash
pip install inference-gate[test]
```

The plugin auto-registers via the `pytest11` entry point. No `conftest.py` changes needed.

### 2. Write a test

```python
def test_my_ai_feature(inference_gate_url):
    from openai import OpenAI

    client = OpenAI(base_url=f"{inference_gate_url}/v1", api_key="unused")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
    )
    assert response.choices[0].message.content
```

### 3. Record cassettes (first time only)

```bash
# Configure InferenceGate with your real API credentials (one-time setup)
inference-gate config init

# Run tests in record mode — requests go to real API, responses are cached
pytest --inferencegate-mode record
```

### 4. Run tests (subsequent runs, CI)

```bash
# Default replay mode — uses cached cassettes, no API key needed
pytest
```

Commit the `tests/cassettes/` directory to version control.

---

## How It Works

When a test requests the `inference_gate_url` (or `inference_gate`) fixture, the plugin:

1. **Starts an InferenceGate proxy** in a background thread on an OS-assigned port
2. **Provides the URL** (e.g. `http://127.0.0.1:54321`) to the test
3. **Routes requests** through the proxy:
   - **Replay mode** (default): Returns cached responses from cassettes. Returns HTTP 503 on cache miss.
   - **Record mode**: Forwards cache misses to the real AI API, records responses as cassettes, and returns them.
4. **Stops the server** cleanly after all tests finish

The proxy handles both streaming and non-streaming requests transparently. All cassettes are stored as streaming responses internally and adapted to match each client's preference at delivery time.

---

## Configuration

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--inferencegate-mode` | `replay` | `replay` (cached only) or `record` (cache + upstream) |
| `--inferencegate-cache-dir` | `tests/cassettes` | Directory for stored cassettes |
| `--inferencegate-config` | auto-detect | Path to InferenceGate `config.yaml` |
| `--inferencegate-port` | `0` | Server port (`0` = OS-assigned) |
| `--inferencegate-proxy` | none | HTTP proxy URL for upstream requests |
| `--inferencegate-max-live-requests` | none | Global limit on the number of live upstream requests |

### Environment Variables

| Variable | Equivalent Flag |
|---|---|
| `INFERENCEGATE_MODE` | `--inferencegate-mode` |
| `INFERENCEGATE_CACHE_DIR` | `--inferencegate-cache-dir` |
| `INFERENCEGATE_CONFIG` | `--inferencegate-config` |
| `INFERENCEGATE_PORT` | `--inferencegate-port` |
| `INFERENCEGATE_PROXY` | `--inferencegate-proxy` |
| `INFERENCEGATE_MAX_LIVE_REQUESTS` | `--inferencegate-max-live-requests` |

### ini Options (`pytest.ini` / `pyproject.toml`)

```toml
[tool.pytest.ini_options]
inferencegate_mode = "replay"
inferencegate_cache_dir = "tests/cassettes"
inferencegate_proxy = "http://127.0.0.1:8888/"
inferencegate_max_live_requests = "5"
```

### Resolution Order

**CLI flag > environment variable > ini option > default**

---

## Fixtures

### `inference_gate_url` (session-scoped)

The primary fixture most tests should use. Returns a string like `"http://127.0.0.1:54321"`.

```python
def test_chat(inference_gate_url):
    from openai import OpenAI
    client = OpenAI(base_url=f"{inference_gate_url}/v1", api_key="unused")
    # ... use client normally
```

### `inference_gate` (session-scoped)

Returns the `InferenceGate` instance directly. Useful for advanced inspection (mode, storage, port, etc.).

```python
def test_advanced(inference_gate):
    assert inference_gate.mode.value == "replay-only"
    assert inference_gate.actual_port > 0
    print(f"Server running at {inference_gate.base_url}")
```

### `inference_gate_factory` (session-scoped)

A factory fixture that creates `InferenceGate` instances with custom overrides. Useful for multi-model setups where you need to pass `model_routes` or other per-project configuration.

The factory returns a callable `create(**overrides)` that:
- Merges your overrides with the default session config (from CLI flags, env vars, ini options).
- Starts the InferenceGate server in a background thread.
- Returns the `InferenceGate` instance.
- Cleans up all created servers at session teardown.

```python
from inference_gate.outflow.model_router import UpstreamConfig

@pytest.fixture(scope="session")
def inference_gate(inference_gate_factory):
    """Project-level override: start a Gate with model-based routing."""
    model_routes = {
        "Gemma4:E4B-it-Q4_K_M": UpstreamConfig(url="http://127.0.0.1:8125"),
        "Gemma-4-31B": UpstreamConfig(url="http://127.0.0.1:8000", api_key="vllm-key"),
    }
    return inference_gate_factory(model_routes=model_routes)
```

When called with no arguments, `inference_gate_factory()` behaves identically to the default `inference_gate` fixture.

---

## Markers

### `@pytest.mark.requires_recording`

Mark tests that don't have cassettes yet and need recording before they can pass:

```python
import pytest

@pytest.mark.requires_recording
def test_new_feature(inference_gate_url):
    # This test will be SKIPPED in replay mode
    # Run with: pytest --inferencegate-mode record
    ...
```

In replay mode (default), these tests are automatically skipped. In record mode, they run normally.

---

## Workflows

### Bootstrap: Writing New Tests

1. **One-time setup** — configure InferenceGate with your API credentials:
   ```bash
   inference-gate config init
   # Edit ~/.InferenceGate/config.yaml:
   #   upstream: https://api.openai.com   (or your provider)
   #   api_key is loaded from OPENAI_API_KEY env var
   ```

2. **Write your test** using the `inference_gate_url` fixture:
   ```python
   def test_summarize(inference_gate_url):
       from openai import OpenAI
       client = OpenAI(base_url=f"{inference_gate_url}/v1", api_key="unused")
       resp = client.chat.completions.create(
           model="gpt-4o-mini",
           messages=[{"role": "user", "content": "Summarize: The cat sat on the mat."}],
           max_tokens=50,
       )
       assert "cat" in resp.choices[0].message.content.lower()
   ```

3. **Record the cassette**:
   ```bash
   OPENAI_API_KEY=sk-... pytest --inferencegate-mode record tests/test_summarize.py
   ```

4. **Commit the cassettes**:
   ```bash
   git add tests/cassettes/
   git commit -m "Add inference cassettes for summarize tests"
   ```

5. **Subsequent runs** — just `pytest`, no API key needed.

### CI / Default Runs

```bash
# Cassettes committed to VCS, no configuration needed
pytest
```

All responses come from cached cassettes. No API key required. No network access. Fully deterministic.

### Refreshing Stale Cassettes

When the upstream API changes behavior or you modify test prompts:

```bash
# Delete specific stale cassettes (or all)
rm tests/cassettes/*.json

# Re-record
OPENAI_API_KEY=sk-... pytest --inferencegate-mode record

# Commit updated cassettes
git add tests/cassettes/
git commit -m "Refresh inference cassettes"
```

### Multi-Model Testing

When your project tests multiple models served by different backends (e.g. a local llama.cpp server and a remote vLLM instance), use `model_routes` to route each model to the correct upstream during recording.

**1. Define model routes in your project's `conftest.py`:**

```python
# tests/conftest.py
import pytest
from inference_gate.outflow.model_router import UpstreamConfig

TEST_MODEL_LLAMACPP = "Gemma4:E4B-it-Q4_K_M"
TEST_MODEL_VLLM = "Gemma-4-31B"

@pytest.fixture(scope="session")
def inference_gate(inference_gate_factory):
    """Override the default Gate with multi-model routing."""
    model_routes = {
        TEST_MODEL_LLAMACPP: UpstreamConfig(url="http://127.0.0.1:8125"),
        TEST_MODEL_VLLM: UpstreamConfig(url="http://127.0.0.1:8000", api_key="vllm-key"),
    }
    return inference_gate_factory(model_routes=model_routes)
```

**2. Write per-model tests:**

```python
def test_llamacpp_generation(inference_gate_url):
    from openai import OpenAI
    client = OpenAI(base_url=f"{inference_gate_url}/v1", api_key="unused")
    resp = client.chat.completions.create(
        model="Gemma4:E4B-it-Q4_K_M",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert resp.choices[0].message.content

def test_vllm_generation(inference_gate_url):
    from openai import OpenAI
    client = OpenAI(base_url=f"{inference_gate_url}/v1", api_key="unused")
    resp = client.chat.completions.create(
        model="Gemma-4-31B",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert resp.choices[0].message.content
```

**3. Record:** Run `pytest --inferencegate-mode record` — each model's requests are forwarded to its configured upstream. All cassettes are stored in the same `tests/cassettes/` directory.

**4. Replay:** Just `pytest` — InferenceGate matches cached responses by model name (or with `fuzzy_model` enabled, falls back across models).

**Configuration file alternative:** Instead of hardcoding routes in `conftest.py`, you can define `model_routes` in a YAML config file (see [Configuration File Reference](configuration_file.md#model-based-upstream-routing)) and point to it with `--inferencegate-config`.

---

## Security

- **API keys never appear in test code.** The `api_key="unused"` in test clients is a placeholder — InferenceGate's proxy injects the real key only in record mode, loaded from `~/.InferenceGate/config.yaml` or `OPENAI_API_KEY` env var.
- **Cassettes are sanitized.** `Authorization`, `X-Api-Key`, and `Proxy-Authorization` headers are automatically stripped from stored cassettes before writing to disk.
- **Safe to commit.** Cassettes contain only the request body (prompts, model selection) and response data (completions). No credentials.

---

## Example: Complete Test File

```python
"""Tests for AI-powered summarization feature."""

import pytest


def test_summarize_short_text(inference_gate_url):
    """Test summarization of a short paragraph."""
    from openai import OpenAI

    client = OpenAI(base_url=f"{inference_gate_url}/v1", api_key="unused")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": "Summarize in one sentence: The quick brown fox jumps over the lazy dog.",
        }],
        max_tokens=50,
    )
    summary = response.choices[0].message.content
    assert summary
    assert len(summary) < 200


def test_summarize_returns_valid_json(inference_gate_url):
    """Test that JSON-mode summarization returns valid JSON."""
    import json
    from openai import OpenAI

    client = OpenAI(base_url=f"{inference_gate_url}/v1", api_key="unused")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": "Always respond in JSON format.",
        }, {
            "role": "user",
            "content": 'Summarize: "The cat sat on the mat." Return {"summary": "..."}',
        }],
        max_tokens=100,
    )
    data = json.loads(response.choices[0].message.content)
    assert "summary" in data


@pytest.mark.requires_recording
def test_new_experimental_prompt(inference_gate_url):
    """New test without a cassette — skipped in replay, runs in record mode."""
    from openai import OpenAI

    client = OpenAI(base_url=f"{inference_gate_url}/v1", api_key="unused")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Explain quantum computing in one sentence."}],
        max_tokens=100,
    )
    assert response.choices[0].message.content
```

---

## Notes for AI Coding Agents

When generating tests for a project that uses InferenceGate:

1. **Always use the `inference_gate_url` fixture** — it handles server lifecycle automatically.
2. **Set `api_key="unused"`** in OpenAI client constructors — the proxy manages authentication.
3. **Use `base_url=f"{inference_gate_url}/v1"`** — the `/v1` prefix is part of the API path.
4. **Cassettes must exist for replay mode.** If writing a new test, mark it with `@pytest.mark.requires_recording` so it's properly skipped until cassettes are recorded.
5. **Be deterministic in requests.** Same model, same messages, same `max_tokens` → same cache key → same cassette. Changing any of these creates a new cache miss.
6. **Don't set `stream=True/False` explicitly** unless the test specifically validates streaming behavior. InferenceGate's cache key ignores the `stream` field, so both variants hit the same cassette.

---

## Disabling the Plugin

```bash
# Temporarily disable InferenceGate plugin
pytest -p no:inferencegate
```

This cleanly removes all fixtures and CLI options. Useful if a project installs InferenceGate but some test runs don't need it.

## Running under pytest-xdist

The plugin is xdist-aware. When the controller starts, it spawns a single `inference-gate serve --print-url --parent-pid <pid>` subprocess and exports its URL via `INFERENCEGATE_URL` so all xdist workers share the same Gate.

```bash
pytest -n auto --dist loadscope
```

Use `--dist loadscope` so tests in the same module land on the same worker — this maximises cassette reuse.

In record mode, concurrent writes from multiple workers are safe: Gate uses `asyncio.Lock` on the index, per-content-hash dedup locks, and atomic `os.replace` on tape/JSON/NDJSON writes.

For cassette debugging, fall back to serial execution: `pytest -n 0`.


