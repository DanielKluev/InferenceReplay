# Configuration File Reference

InferenceGate uses a YAML configuration file to store settings. This document describes the file format, location, and all supported fields.

## File Location

By default, the configuration file is located at:

```
$USERDIR/.InferenceGate/config.yaml
```

Where `$USERDIR` is the user's home directory (`%USERPROFILE%` on Windows, `$HOME` on Unix).

A custom path can be specified using the `--config` / `-C` global CLI option:

```bash
inference-gate -C /path/to/my/config.yaml start
```

If the default config file does not exist, InferenceGate creates it automatically with default values on first run.

## Configuration Priority

Settings are resolved in the following order (later values override earlier ones):

1. **Built-in defaults** — hardcoded in the application
2. **Configuration file** — values from the YAML file
3. **Environment variables** — `OPENAI_API_KEY` sets `api_key` if not specified in the file
4. **Command-line flags** — CLI options override everything

## File Format

The configuration file is a standard YAML file. All fields are optional — omitted fields use their built-in defaults.

### Example

```yaml
host: 127.0.0.1
port: 8080
upstream: "http://127.0.0.1:8000/"
cache_dir: caches/my_project
verbose: false
proxy: "http://127.0.0.1:8888/"
upstream_timeout: 300.0
fuzzy_model: false
fuzzy_sampling: "off"
max_non_greedy_replies: 5
max_live_requests: null
non_streaming_models:
  - "o1-preview"
  - "o1-mini"
test_model: gpt-4o-mini
test_prompt: 'This is a test prompt. Reply with **ONLY** "OK."'
api_key: "sk-..."
```

## Field Reference

### Server Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | string | `127.0.0.1` | Host address to bind the proxy server to. Use `0.0.0.0` to listen on all interfaces. |
| `port` | integer | `8080` | Port number for the proxy server. |

### Upstream API Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `upstream` | string | `https://api.openai.com` | Base URL of the upstream AI API endpoint. Can be any OpenAI-compatible API. |
| `api_key` | string | `null` | API key for upstream authentication. Sent as a `Bearer` token in the `Authorization` header. Falls back to the `OPENAI_API_KEY` environment variable if not set. **Note:** For security, this field is excluded when saving the config file via `config init`. |
| `upstream_timeout` | float | `120.0` | Timeout in seconds for upstream API requests. If an upstream request takes longer than this, InferenceGate returns a `504 Gateway Timeout` to the client. Increase this for long-running inference tasks (e.g. large reasoning models). |
| `proxy` | string | `null` | Optional HTTP proxy URL for routing upstream requests through a proxy server (e.g. `http://127.0.0.1:8888/`). When set, all outbound requests to the upstream AI endpoint are sent via the specified proxy. Useful for debugging with tools like mitmproxy or for network environments that require a proxy. |

### Model-Based Upstream Routing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_routes` | dict | `{}` | Maps model names (or glob patterns) to per-model upstream configurations. When set and the server is in `record_and_replay` mode, each incoming request is dispatched to the upstream that matches its `model` field. Unmatched models fall back to the default `upstream`. |

Each entry under `model_routes` is keyed by a model name or an `fnmatch`-style glob pattern and contains:

| Sub-field | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | string | *(required)* | Base URL of the upstream API for this model. |
| `api_key` | string | `null` | Optional API key for this upstream. |
| `timeout` | float | `120.0` | Request timeout for this upstream. |
| `proxy` | string | `null` | Optional HTTP proxy for this upstream. |

**Resolution order:**
1. Exact model name match.
2. First `fnmatch` glob pattern match (e.g. `"Gemma4:*"`).
3. Default `upstream` from the top-level config.

#### Example

```yaml
upstream: "http://127.0.0.1:8000/"
model_routes:
  "Gemma4:E4B-it-Q4_K_M":
    url: "http://127.0.0.1:8125"
  "Gemma4:*":
    url: "http://127.0.0.1:8125"
    api_key: "my-local-key"
  "Gemma-4-31B":
    url: "http://127.0.0.1:8000"
    api_key: "vllm-key"
```

In this example, requests for `Gemma4:E4B-it-Q4_K_M` go to `http://127.0.0.1:8125` (exact match); any other `Gemma4:*` model also goes there (glob match); `Gemma-4-31B` goes to the vLLM server; everything else falls back to the top-level `upstream`.

### Storage Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cache_dir` | string | `.inference_cache` | Directory path for storing cached inference responses (cassettes). Can be relative or absolute. |

### Matching Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fuzzy_model` | boolean | `false` | When enabled, on a cache miss, InferenceGate looks for cached entries with the same prompt but a different model name. Useful when switching between equivalent models. |
| `fuzzy_sampling` | string | `"off"` | Controls fuzzy matching of sampling parameters. `"off"` = exact match only. `"soft"` = match non-greedy requests with different sampling params (greedy stays exact). `"aggressive"` = match across any sampling param difference, including greedy ↔ non-greedy. |
| `max_non_greedy_replies` | integer | `5` | Maximum number of distinct replies to collect in a cassette for non-greedy (temperature > 0) requests before cycling through existing replies. Greedy requests always store exactly 1 reply. |
| `max_live_requests` | integer or null | `null` | Global limit on the number of live (non-cassette) upstream requests InferenceGate will perform per session. Once this limit is reached, InferenceGate will return an HTTP 503 error for any further upstream requests. `null` means infinite. |

### Streaming Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `non_streaming_models` | list of strings | `[]` | List of model names that do not support streaming. Requests for these models will never be forced to use streaming, even if the client requests it. |

### Logging Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `verbose` | boolean | `false` | Enable verbose (DEBUG-level) logging output. |

### Test Command Settings

These fields configure the defaults for the `test-gate` and `test-upstream` CLI commands.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `test_model` | string | `gpt-4o-mini` | Default model name used by test commands. |
| `test_prompt` | string | `This is a test prompt. Reply with **ONLY** "OK."...` | Default prompt text used by test commands. |

## CLI Override Mapping

The following table shows how configuration file fields map to CLI options on the `start` and `replay` commands:

| Config Field | CLI Option | Available In |
|-------------|------------|--------------|
| `host` | `--host` / `-h` | `start`, `replay` |
| `port` | `--port` / `-p` | `start`, `replay` |
| `upstream` | `--upstream` / `-u` | `start` |
| `api_key` | `--api-key` / `-k` | `start` |
| `upstream_timeout` | `--upstream-timeout` | `start` |
| `cache_dir` | `--cache-dir` / `-c` | `start`, `replay` |
| `fuzzy_model` | `--fuzzy-model` / `--no-fuzzy-model` | `start`, `replay` |
| `fuzzy_sampling` | `--fuzzy-sampling` | `start`, `replay` |
| `max_non_greedy_replies` | `--max-non-greedy-replies` | `start`, `replay` |
| `max_live_requests` | `--max-live-requests` | `start`, `serve` |
| `verbose` | `--verbose` / `-v` | `start`, `replay` |

## Managing Configuration

Use the `config` CLI commands to manage the configuration file:

```bash
# Show current effective configuration
inference-gate config show

# Create a default configuration file
inference-gate config init

# Show the configuration file path
inference-gate config path
```
