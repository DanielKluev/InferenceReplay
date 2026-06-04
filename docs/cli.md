# Command-Line Interface (CLI) Reference

InferenceGate provides a comprehensive CLI for managing the inference proxy server, cache, and configuration.

## Global Options

These options can be used with any command:

| Option | Description |
|--------|-------------|
| `--config, -C` | Path to configuration file (default: `$USERDIR/.InferenceGate/config.yaml`) |
| `--version` | Show version information |
| `--help` | Show help message |

## Server Commands

### `start` - Record-and-Replay Mode

Start the proxy server in record-and-replay mode. This is the default mode for development.

**Behavior:**
- Attempts to replay inferences from the local cache
- On cache miss, forwards the request to the upstream API
- Records the response and stores it for future replays

```bash
inference-gate start [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--port` | `-p` | Port to run the server on | 8080 |
| `--host` | `-h` | Host to bind the server to | 127.0.0.1 |
| `--cache-dir` | `-c` | Directory to store cached responses | .inference_cache |
| `--upstream` | `-u` | Upstream OpenAI API base URL | https://api.openai.com |
| `--api-key` | `-k` | OpenAI API key | $OPENAI_API_KEY |
| `--max-live-requests` | | Global limit on the number of non-cassette upstream requests | none |
| `--upstream-timeout` | | Timeout in seconds for upstream API requests before 504 | 120.0 |
| `--proxy` | | HTTP proxy URL for upstream requests | none |
| `--verbose` | `-v` | Enable verbose (DEBUG) logging | false |

**Example:**

```bash
# Basic usage with API key from environment
inference-gate start

# Custom port and verbose logging
inference-gate start --port 9000 --verbose

# Custom upstream endpoint (e.g., Azure OpenAI)
inference-gate start --upstream https://my-resource.openai.azure.com

# Route upstream requests through an HTTP proxy
inference-gate start --proxy http://127.0.0.1:8888/
```

### `replay` - Replay-Only Mode

Start the proxy server in replay-only mode. This mode is intended for unit tests and CI pipelines.

**Behavior:**
- Only attempts to replay inferences from the local cache
- Returns an error response if a matching inference is not found
- Does not forward any requests to the upstream API
- Does not require an API key

```bash
inference-gate replay [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--port` | `-p` | Port to run the server on | 8080 |
| `--host` | `-h` | Host to bind the server to | 127.0.0.1 |
| `--cache-dir` | `-c` | Directory to store cached responses | .inference_cache |
| `--verbose` | `-v` | Enable verbose (DEBUG) logging | false |

**Example:**

```bash
# Start replay server for testing
inference-gate replay --cache-dir ./test_cache

# Use in CI with specific cache directory
inference-gate replay --cache-dir ./fixtures/api_cache
```

## Test Commands

### `test-gate` - Test a Running InferenceGate Instance

Send a test prompt to a running InferenceGate proxy to verify it is accepting and processing requests correctly.

**Behavior:**
- Sends a test prompt to the InferenceGate proxy at the configured host/port
- By default, uses a prompt asking the model to reply with "OK."
- No API key is needed — the running instance already has it configured
- Uses the same host/port from the configuration file, so no extra options needed in most cases
- Reports success or failure with response details

```bash
inference-gate test-gate [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--host` | `-h` | Host of the running InferenceGate instance | 127.0.0.1 |
| `--port` | `-p` | Port of the running InferenceGate instance | 8080 |
| `--model` | `-m` | Model to use for the test | gpt-4o-mini |
| `--prompt` | | Custom prompt to send | (built-in test prompt) |
| `--verbose` | `-v` | Enable verbose (DEBUG) logging | false |

**Default Test Prompt:**

```
This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.
```

**Example:**

```bash
# Test the running instance (uses host/port from config)
inference-gate test-gate

# Test instance on a custom port
inference-gate test-gate --port 9000

# Test with a custom model
inference-gate test-gate --model gpt-4

# Test with a custom prompt
inference-gate test-gate --prompt "Say hello!"
```

**Exit Codes:**
- `0` - Test passed successfully
- `1` - Test failed (connection refused, proxy error, etc.)

### `test-upstream` - Test Upstream API Directly

Send a test prompt directly to the upstream API, bypassing InferenceGate, to verify that the API key and endpoint are working correctly.

**Behavior:**
- Sends a test prompt directly to the upstream API
- By default, uses a prompt asking the model to reply with "OK."
- Requires an API key (via `--api-key`, `OPENAI_API_KEY` env var, or config file)
- Reports success or failure with response details

```bash
inference-gate test-upstream [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--upstream` | `-u` | Upstream OpenAI API base URL | https://api.openai.com |
| `--api-key` | `-k` | OpenAI API key | $OPENAI_API_KEY |
| `--model` | `-m` | Model to use for the test | gpt-4o-mini |
| `--prompt` | | Custom prompt to send | (built-in test prompt) |
| `--verbose` | `-v` | Enable verbose (DEBUG) logging | false |

**Example:**

```bash
# Test with API key from environment
inference-gate test-upstream

# Test with specific API key
inference-gate test-upstream --api-key sk-...

# Test with custom model
inference-gate test-upstream --model gpt-4

# Test Azure OpenAI endpoint
inference-gate test-upstream --upstream https://my-resource.openai.azure.com
```

**Exit Codes:**
- `0` - Test passed successfully
- `1` - Test failed (connection error, authentication error, etc.)

## Cassette Management Commands

All cassette commands support `--cache-dir` to override the cache directory and are designed for both human and AI agentic consumption via the `--json` flag. Cassette IDs support prefix matching (like git short hashes).

### `cassette list` - List Cached Cassettes

List all cached cassettes with filtering and sorting.

```bash
inference-gate cassette list [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--cache-dir` | `-c` | Directory where cached responses are stored | .inference_cache |
| `--model` | `-m` | Filter by model name (substring match) | |
| `--greedy/--non-greedy` | | Filter by greedy / non-greedy sampling | |
| `--has-tools` | | Filter cassettes that use tools | |
| `--has-logprobs` | | Filter cassettes that have logprobs | |
| `--after` | | Only cassettes recorded after this ISO 8601 date | |
| `--before` | | Only cassettes recorded before this ISO 8601 date | |
| `--sort` | | Sort by: `recorded`, `model`, `tokens_in`, `tokens_out` | recorded |
| `--limit` | `-n` | Maximum number of results | |
| `--json` | | Output as JSON array | false |

**Example:**

```bash
# List all cassettes
inference-gate cassette list

# Filter by model
inference-gate cassette list --model gpt-4

# Only greedy cassettes, sorted by token output
inference-gate cassette list --greedy --sort tokens_out

# JSON output for programmatic use
inference-gate cassette list --json

# Limit to 10 most recent
inference-gate cassette list --limit 10
```

**Human Output:**

```
ID             Model                        Temp  Repl  Tok In Tok Out Recorded               First Message
------------------------------------------------------------------------------------------------------------------------
6c72599f3142   openai/gpt-oss-120b                   1      82      34 2026-04-12T02:35:08    What is 2+2? Reply with only the number.
e5ce50e06a1e   openai/gpt-oss-120b           0.7     3     150      75 2026-04-11T18:20:00    Explain quantum computing in simple terms

2 cassette(s)
```

### `cassette search` - Search Cassettes

Full-text search across first user message and slug fields.

```bash
inference-gate cassette search QUERY [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--cache-dir` | `-c` | Directory where cached responses are stored | .inference_cache |
| `--model` | `-m` | Additional model filter (substring) | |
| `--limit` | `-n` | Maximum results | 20 |
| `--json` | | Output as JSON array | false |

**Example:**

```bash
# Search for cassettes about Python
inference-gate cassette search "python"

# Search with model filter and JSON output
inference-gate cassette search "explain" --model gpt-4 --json
```

### `cassette show` - Show Cassette Details

Show metadata, prompt messages, and reply summaries for a cassette. Does **not** show full completion text — use `read` for that.

```bash
inference-gate cassette show CASSETTE_ID [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--cache-dir` | `-c` | Directory where cached responses are stored | .inference_cache |
| `--json` | | Output as JSON object | false |

**Example:**

```bash
# Show by full hash
inference-gate cassette show 6c72599f3142

# Show by prefix (like git short hashes)
inference-gate cassette show 6c72

# JSON output
inference-gate cassette show 6c72599f3142 --json
```

**Human Output:**

```
Cassette: 6c72599f3142
Model: openai/gpt-oss-120b
Endpoint: /v1/chat/completions
Recorded: 2026-04-12T02:35:08+00:00
Temperature: (not set)
Greedy: No
Replies: 1 / 1
Tools: none
Logprobs: No
Content Hash: 6c72599f3142
Prompt+Model Hash: 6c72599f3142
Prompt Hash: 8c81e05fef75a909

--- Prompt ---
[user] What is 2+2? Reply with only the number.

--- Replies ---
Reply 1: stop | 82 in / 34 out
```

### `cassette read` - Read Full Completion Text

Read the full completion text of a cassette's reply or replies.

```bash
inference-gate cassette read CASSETTE_ID [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--cache-dir` | `-c` | Directory where cached responses are stored | .inference_cache |
| `--reply` | `-r` | Show only a specific reply number | all replies |
| `--prompt` | `-p` | Also include the prompt messages | false |
| `--json` | | Output as JSON | false |

**Example:**

```bash
# Read all replies
inference-gate cassette read 6c72599f3142

# Read specific reply
inference-gate cassette read 6c72 --reply 1

# Include prompt in output
inference-gate cassette read 6c72 --prompt

# JSON output for programmatic use
inference-gate cassette read 6c72 --json
```

**JSON Output:**

```json
{
  "replies": [
    {
      "reply_number": 1,
      "text": "4",
      "stop_reason": "stop",
      "input_tokens": "82",
      "output_tokens": "34"
    }
  ]
}
```

### `cassette delete` - Delete a Cassette

Delete a single cassette by ID, removing the tape file, associated response files, and index entry.
Alternatively, delete all cassettes for a specific model using the `--model` flag, or delete all cassettes that recorded an error using `--non-200`.

```bash
inference-gate cassette delete [CASSETTE_ID] [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--cache-dir` | `-c` | Directory where cached responses are stored | .inference_cache |
| `--model` | `-m` | Delete all cassettes matching this model name (substring match) | |
| `--non-200` | | Delete all cassettes with a non-200 HTTP status code | |
| `--yes` | `-y` | Skip confirmation prompt | false |

**Example:**

```bash
# Interactive (asks for confirmation)
inference-gate cassette delete 6c72599f3142

# Delete all cassettes for a specific model
inference-gate cassette delete --model gpt-4

# Delete all cassettes with an error response (non-200)
inference-gate cassette delete --non-200 -c tests/cassettes

# Non-interactive
inference-gate cassette delete 6c72 --yes
```

### `cassette stats` - Show Cache Statistics

Display aggregated statistics about the cassette cache.

```bash
inference-gate cassette stats [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--cache-dir` | `-c` | Directory where cached responses are stored | .inference_cache |
| `--json` | | Output as JSON | false |

**Example:**

```bash
inference-gate cassette stats

# JSON for programmatic consumption
inference-gate cassette stats --json
```

**Human Output:**

```
Cache directory: .inference_cache
Total cassettes: 42
Total replies: 85
Disk size: 5.2 MB
Greedy: 10 | Non-greedy: 32
Total tokens: 15,200 in / 8,400 out

Entries by model:
  gpt-4: 25
  claude-3: 17
```

**JSON Output:**

```json
{
  "total_entries": 42,
  "total_replies": 85,
  "disk_size_bytes": 5242880,
  "greedy_entries": 10,
  "non_greedy_entries": 32,
  "entries_by_model": {"gpt-4": 25, "claude-3": 17},
  "total_tokens_in": 15200,
  "total_tokens_out": 8400
}
```

### `cassette reindex` - Rebuild Index

Rebuild the TSV index from tape files on disk. Useful after manually editing or copying tape files.

```bash
inference-gate cassette reindex [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--cache-dir` | `-c` | Directory where cached responses are stored | .inference_cache |

**Example:**

```bash
inference-gate cassette reindex
```

### `cassette fill` - Fill Cassette with Unique Completions

Re-issue the original request to the upstream endpoint to collect additional unique completions for a non-greedy cassette. Responses are de-duplicated by content hash, so identical completions are never stored twice.

This is useful when you have a cassette with only one or few recorded completions and want to fill it up to a target count for testing variation in non-deterministic (non-greedy) sampling scenarios.

**Behavior:**
- Reconstructs the original request from the tape's metadata and message sections
- Sends the request to the upstream endpoint repeatedly until the target count is reached
- De-duplicates responses by content hash — identical completions are discarded
- Only works with non-greedy cassettes (temperature > 0)
- Stops when the target is reached or when max attempts are exhausted
- Updates the cassette's `max_replies` if the target exceeds the current value

```bash
inference-gate cassette fill CASSETTE_ID [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--count` | `-n` | Target number of unique completions | max_non_greedy_replies from config |
| `--cache-dir` | `-c` | Directory where cached responses are stored | .inference_cache |
| `--upstream` | `-u` | Upstream OpenAI API base URL | from config |
| `--api-key` | `-k` | OpenAI API key | $OPENAI_API_KEY |
| `--upstream-timeout` | | Timeout in seconds for upstream requests | 120.0 |
| `--proxy` | | HTTP proxy URL for upstream requests | from config |
| `--max-attempts` | | Maximum upstream requests before giving up | 3x target count |
| `--json` | | Output as JSON | false |
| `--verbose` | `-v` | Enable verbose (DEBUG) logging | false |

**Example:**

```bash
# Fill cassette to 5 unique completions (default)
inference-gate fill 6c72599f3142 --api-key sk-...

# Fill to specific count
inference-gate cassette fill 6c72 --count 10

# Use custom upstream endpoint
inference-gate cassette fill 6c72 --upstream http://localhost:8000 --api-key test-key

# JSON output for programmatic use
inference-gate cassette fill 6c72 --count 5 --json
```

**Human Output:**

```
Cassette: 6c72599f3142
  Model: gpt-4
  Existing replies: 1
  Target: 5
  Need: 4 more unique completions
  Max attempts: 12
  Upstream: https://api.openai.com

Done: 4 new unique completions added (2 duplicates, 0 errors in 6 attempts)
Cassette now has 5 replies
```

**JSON Output:**

```json
{
  "content_hash": "6c72599f3142",
  "added": 4,
  "duplicates": 2,
  "errors": 0,
  "attempts": 6,
  "total_replies": 5,
  "target": 5
}
```

**Exit Codes:**
- `0` - Command completed (check `added` count for actual result)
- `1` - Invalid input (greedy cassette, not found, no API key)

## Configuration Commands

### `config show` - Show Current Configuration

Display the current configuration settings, including values from the config file, environment variables, and defaults.

```bash
inference-gate config show
```

**Example Output:**

```
Configuration file: C:\Users\User\.InferenceGate\config.yaml
File exists: true

Current settings:
  host: 127.0.0.1
  port: 8080
  upstream: https://api.openai.com
  api_key: ***abc1
  cache_dir: .inference_cache
  verbose: false
  test_model: gpt-4o-mini
  test_prompt: This is a test prompt. Reply with **ONLY** "OK."...
```

### `config init` - Initialize Configuration File

Create a default configuration file at the default location or the path specified with `--config`.

```bash
inference-gate config init [OPTIONS]
```

**Options:**

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--force` | `-f` | Overwrite existing configuration file | false |

**Example:**

```bash
# Create default config
inference-gate config init

# Overwrite existing config
inference-gate config init --force

# Initialize at custom location
inference-gate --config ./my-config.yaml config init
```

### `config path` - Show Configuration File Path

Display the path to the configuration file being used.

```bash
inference-gate config path
```

**Example Output:**

```
C:\Users\User\.InferenceGate\config.yaml
```

## Configuration File

InferenceGate uses a YAML configuration file to store default settings. Command-line options always override configuration file values.

### File Location

- **Windows**: `%USERPROFILE%\.InferenceGate\config.yaml`
- **macOS/Linux**: `~/.InferenceGate/config.yaml`

Use `--config` to specify a custom path:

```bash
inference-gate --config /path/to/config.yaml start
```

### Configuration Options

```yaml
# Server settings
host: "127.0.0.1"          # Host to bind the server to
port: 8080                  # Port to run the server on
record_timeout: 600.0       # Default upstream HTTP timeout in seconds

# Storage settings
cache_dir: ".inference_cache"  # Directory to store cached responses

# Logging settings
verbose: false              # Enable verbose (DEBUG) logging

# Endpoints configuration (supports runtime override via POST /gate/config)
endpoints:
  default_upstream:
    url: "https://api.openai.com"
    # Note: api_key is NOT stored in the config file by default for security
    verify_ssl: true  # Set to false to disable SSL verification on egress requests

# Model routing rules
models:
  - pattern: "*"
    endpoint: "default_upstream"

# Test command settings
test_model: "gpt-4o-mini"   # Default model for the test command
test_prompt: "This is a test prompt..."  # Default prompt for the test command
```

### Runtime Configuration

The server supports modifying settings at runtime via the `POST /gate/config` admin endpoint. You can dynamically push updates to the `endpoints`, `models`, and `mode` without restarting InferenceGate. For example, to disable SSL verification on an endpoint during a test session:

```json
{
  "endpoints": {
    "default_upstream": {
      "url": "https://api.openai.com",
      "verify_ssl": false
    }
  }
}
```

### Configuration Priority

Settings are loaded in the following order (later overrides earlier):

1. **Built-in defaults** - Hardcoded default values
2. **Configuration file** - Values from YAML config file
3. **Environment variables** - `OPENAI_API_KEY` (for API key only)
4. **Command-line options** - Explicit options passed to the command

### Security Note

The `api_key` setting is intentionally NOT saved to the configuration file when using `config init`. Always use the `OPENAI_API_KEY` environment variable or the `--api-key` command-line option for API key management.

## Environment Variables

| Variable | Description | Used By |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | `start`, `test-upstream` |

## Examples

### Development Workflow

```bash
# 1. Initialize configuration
inference-gate config init

# 2. Test upstream API connection
export OPENAI_API_KEY=sk-...
inference-gate test-upstream

# 3. Start the proxy
inference-gate start

# 4. Test the running proxy (in another terminal)
inference-gate test-gate

# 5. Run your application pointing to http://localhost:8080/v1
```

### CI/CD Workflow

```bash
# Pre-recorded responses in fixtures directory
inference-gate replay --cache-dir ./fixtures/api_cache &

# Run tests against the replay server
pytest tests/
```

### Custom Configuration

```bash
# Use project-specific configuration
inference-gate --config ./.inference-gate.yaml start

# Initialize project-specific config
inference-gate --config ./.inference-gate.yaml config init
```