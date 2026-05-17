# InferenceGate

Python library for efficient and convenient AI inference replay in testing, debugging and development, saving costs and time on repeated prompts.

## Installation

```bash
pip install inference-gate
```

## Features

- **Record-and-Replay Mode**: Record new requests to cache, replay from cache when available
- **Replay-Only Mode**: Only serve cached responses (for unit tests and CI)
- **Web UI Dashboard**: Optional web-based dashboard for browsing cache entries, viewing statistics, and inspecting request/response details
- Supports OpenAI Chat Completions API and Responses API
- Supports streaming responses
- Preserves prompt, temperature, model, and other metadata
- YAML configuration file for persistent settings
- CLI tools for easy management

## Quick Start

### 1. Initialize Configuration (Optional)

```bash
inference-gate config init
```

This creates a configuration file at `$USERDIR/.InferenceGate/config.yaml`.

### 2. Test Your Upstream API Connection

```bash
inference-gate test-upstream --api-key $OPENAI_API_KEY
```

### 3. Start the Proxy

```bash
inference-gate start --api-key $OPENAI_API_KEY
```

### 4. Test the Running Proxy

```bash
inference-gate test-gate
```

### 5. Point Your Client to the Proxy

```python
from openai import OpenAI

client = OpenAI(
    api_key="any-key",  # Not needed in replay mode
    base_url="http://localhost:8080/v1"
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## CLI Commands

### Server Commands

#### `start` - Record-and-Replay Mode (Default)

Replays cached inferences when available. On cache miss, forwards to upstream, records the response, and stores it for future replays.

```bash
inference-gate start [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--port, -p` | Server port | 8080 |
| `--host, -h` | Server host | 127.0.0.1 |
| `--cache-dir, -c` | Cache directory | .inference_cache |
| `--upstream, -u` | Upstream API URL | https://api.openai.com |
| `--api-key, -k` | OpenAI API key | $OPENAI_API_KEY |
| `--max-live-requests` | Global limit on live upstream requests | (infinite) |
| `--web-ui` | Enable web UI dashboard | false |
| `--web-ui-port` | Web UI server port | 8081 |
| `--verbose, -v` | Enable verbose logging | false |

#### `replay` - Replay-Only Mode

Only returns cached responses. Returns an error if a matching inference is not found in the cache. Useful for unit tests and CI pipelines.

```bash
inference-gate replay [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--port, -p` | Server port | 8080 |
| `--host, -h` | Server host | 127.0.0.1 |
| `--cache-dir, -c` | Cache directory | .inference_cache |
| `--web-ui` | Enable web UI dashboard | false |
| `--web-ui-port` | Web UI server port | 8081 |
| `--verbose, -v` | Enable verbose logging | false |

### Test Commands

#### `test-gate` - Test a Running InferenceGate Instance

Sends a test prompt to a running InferenceGate proxy. Uses the same host/port from config, so no API key or extra options needed.

```bash
inference-gate test-gate [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|--------|
| `--host, -h` | Host of the running instance | 127.0.0.1 |
| `--port, -p` | Port of the running instance | 8080 |
| `--model, -m` | Model to use | gpt-4o-mini |
| `--prompt` | Custom test prompt | (built-in test prompt) |
| `--verbose, -v` | Enable verbose logging | false |

#### `test-upstream` - Test Upstream API Directly

Sends a test prompt directly to the upstream API (bypassing InferenceGate) to verify the API key and endpoint.

```bash
inference-gate test-upstream [OPTIONS]
```

**Options:**

| Option | Description | Default |
|--------|-------------|--------|
| `--upstream, -u` | Upstream API URL | https://api.openai.com |
| `--api-key, -k` | OpenAI API key | $OPENAI_API_KEY |
| `--model, -m` | Model to use | gpt-4o-mini |
| `--prompt` | Custom test prompt | (built-in test prompt) |
| `--verbose, -v` | Enable verbose logging | false |

### Cache Management

#### `cache list` - List Cached Entries

```bash
inference-gate cache list [--cache-dir PATH]
```

#### `cache info` - Show Cache Statistics

```bash
inference-gate cache info [--cache-dir PATH]
```

#### `cache clear` - Clear All Cached Entries

```bash
inference-gate cache clear [--cache-dir PATH] [--yes]
```

## Web UI Dashboard

InferenceGate includes an optional web-based dashboard for browsing cached inference entries, viewing statistics, and inspecting request/response details.

### Enabling the Web UI

Add the `--web-ui` flag when starting InferenceGate:

```bash
# Record-and-replay mode with web UI
inference-gate start --api-key $OPENAI_API_KEY --web-ui

# Replay-only mode with web UI
inference-gate replay --web-ui
```

The web UI will be available at `http://localhost:8081` by default. You can customize the port with `--web-ui-port`:

```bash
inference-gate start --web-ui --web-ui-port 3000
```

### Features

- **Dashboard**: View cache statistics, current mode, and configuration at a glance
- **Cache List**: Browse all cached entries in a sortable, filterable table
- **Entry Details**: Inspect full request and response details including headers, body, and metadata
- **Search**: Filter cache entries by ID, model, path, or method
- **Streaming Support**: View streaming response chunks for SSE endpoints

### Screenshots

**Dashboard Page**

![Dashboard](https://github.com/user-attachments/assets/6ec5916c-6e0e-40a7-a9e8-1289af7ed2e8)

**Cache List Page**

![Cache List](https://github.com/user-attachments/assets/01fe025c-7922-4f64-bf20-b2ea6158060e)

**Entry Detail Page**

![Entry Detail](https://github.com/user-attachments/assets/3a858019-b978-4893-9c04-ceb466dea67c)

### Building the Frontend (Development Only)

The web UI frontend is pre-built and included in the package. You only need to build it if you're developing or modifying the frontend:

```bash
cd webui-frontend
npm install
npm run build
# Output goes to src/inference_gate/webui/static/
```

**Requirements:**
- Node.js 16+ and npm (only for frontend development)
- No runtime dependencies - the built static files are served by the Python backend

### Configuration Management

#### `config show` - Show Current Configuration

```bash
inference-gate config show
```

#### `config init` - Initialize Configuration File

```bash
inference-gate config init [--force]
```

#### `config path` - Show Configuration File Path

```bash
inference-gate config path
```

## Configuration File

InferenceGate uses a YAML configuration file to store default settings. The file is located at:

- **Windows**: `%USERPROFILE%\.InferenceGate\config.yaml`
- **macOS/Linux**: `~/.InferenceGate/config.yaml`

You can specify a custom path using the `--config` global option:

```bash
inference-gate --config /path/to/config.yaml start
```

### Configuration Options

```yaml
# Server settings
host: "127.0.0.1"
port: 8080
max_live_requests: null  # Optional global limit on live upstream requests

# Upstream API settings
upstream: "https://api.openai.com"
# api_key is not stored in the config file for security
# Use OPENAI_API_KEY environment variable instead

# Storage settings
cache_dir: ".inference_cache"

# Logging settings
verbose: false

# Test command settings
test_model: "gpt-4o-mini"
test_prompt: "This is a test prompt. Reply with **ONLY** \"OK.\" to confirm that everything is ok. DO NOT output anything else."
```

### Configuration Priority

Settings are loaded in the following order (later overrides earlier):

1. Built-in defaults
2. Configuration file
3. Environment variables (`OPENAI_API_KEY`)
4. Command-line options

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (used in record/test modes) |

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check src/ tests/
```

## License

MIT License
