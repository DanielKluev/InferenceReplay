"""Tests for InferenceGate CLI module."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from inference_gate.cli import main
from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry, CacheStorage


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return str(cache_dir)


class TestMainCommand:
    """Tests for main CLI commands."""

    def test_version(self, runner):
        """Test that --version flag prints version and exits cleanly."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self, runner):
        """Test that --help shows available commands including start and replay."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "InferenceGate" in result.output
        assert "start" in result.output
        assert "replay" in result.output

    def test_start_help(self, runner):
        """Test that start --help shows record-and-replay mode description."""
        result = runner.invoke(main, ["start", "--help"])
        assert result.exit_code == 0
        assert "record-and-replay" in result.output

    def test_replay_help(self, runner):
        """Test that replay --help shows replay-only mode description."""
        result = runner.invoke(main, ["replay", "--help"])
        assert result.exit_code == 0
        assert "replay-only" in result.output


class TestCassetteCommands:
    """Tests for cassette management commands."""

    @staticmethod
    def _store_entry(cache_dir: str, model: str = "gpt-4", message: str = "Hello", temperature: float = 0) -> str:
        """Helper to store a cassette entry and return its content_hash."""
        storage = CacheStorage(cache_dir)
        entry = CacheEntry(
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
        return storage.put(entry)

    def test_cassette_help(self, runner):
        """Test that cassette --help shows available subcommands."""
        result = runner.invoke(main, ["cassette", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "search" in result.output
        assert "show" in result.output
        assert "read" in result.output
        assert "delete" in result.output
        assert "stats" in result.output
        assert "reindex" in result.output
        assert "fill" in result.output

    def test_cassette_list_empty(self, runner, temp_cache_dir):
        """Test that listing empty cache shows 'No cassettes found'."""
        result = runner.invoke(main, ["cassette", "list", "--cache-dir", temp_cache_dir])
        assert result.exit_code == 0
        assert "No cassettes found" in result.output

    def test_cassette_list_with_entries(self, runner, temp_cache_dir):
        """Test that listing cassettes shows stored entries in tabular format."""
        self._store_entry(temp_cache_dir, model="gpt-4", message="What is Python?")
        self._store_entry(temp_cache_dir, model="claude-3", message="Explain ML")

        result = runner.invoke(main, ["cassette", "list", "--cache-dir", temp_cache_dir])
        assert result.exit_code == 0
        assert "gpt-4" in result.output
        assert "claude-3" in result.output
        assert "2 cassette(s)" in result.output

    def test_cassette_list_json(self, runner, temp_cache_dir):
        """Test that --json flag outputs valid JSON array."""
        self._store_entry(temp_cache_dir, message="JSON test")

        result = runner.invoke(main, ["cassette", "list", "--cache-dir", temp_cache_dir, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1
        assert "id" in data[0]
        assert "model" in data[0]

    def test_cassette_list_filter_model(self, runner, temp_cache_dir):
        """Test that --model flag filters by model name."""
        self._store_entry(temp_cache_dir, model="gpt-4", message="GPT test")
        self._store_entry(temp_cache_dir, model="claude-3", message="Claude test")

        result = runner.invoke(main, ["cassette", "list", "--cache-dir", temp_cache_dir, "--model", "gpt", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["model"] == "gpt-4"

    def test_cassette_list_limit(self, runner, temp_cache_dir):
        """Test that --limit caps the number of results."""
        for i in range(5):
            self._store_entry(temp_cache_dir, message=f"Limit entry {i}", model=f"model-{i}")

        result = runner.invoke(main, ["cassette", "list", "--cache-dir", temp_cache_dir, "--limit", "2", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2

    def test_cassette_search(self, runner, temp_cache_dir):
        """Test searching cassettes by message content."""
        self._store_entry(temp_cache_dir, message="The quick brown fox")
        self._store_entry(temp_cache_dir, message="Lazy dog sleeps")

        result = runner.invoke(main, ["cassette", "search", "quick", "--cache-dir", temp_cache_dir])
        assert result.exit_code == 0
        assert "quick" in result.output.lower()
        assert "1 cassette(s)" in result.output

    def test_cassette_search_json(self, runner, temp_cache_dir):
        """Test search with --json output."""
        self._store_entry(temp_cache_dir, message="Search target alpha")
        self._store_entry(temp_cache_dir, message="Search target beta")

        result = runner.invoke(main, ["cassette", "search", "search target", "--cache-dir", temp_cache_dir, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 2

    def test_cassette_search_no_results(self, runner, temp_cache_dir):
        """Test search with no matching results."""
        self._store_entry(temp_cache_dir, message="Something else")

        result = runner.invoke(main, ["cassette", "search", "nonexistent", "--cache-dir", temp_cache_dir])
        assert result.exit_code == 0
        assert "No cassettes found" in result.output

    def test_cassette_show(self, runner, temp_cache_dir):
        """Test showing cassette details."""
        content_hash = self._store_entry(temp_cache_dir, model="gpt-4", message="Show me details")

        result = runner.invoke(main, ["cassette", "show", content_hash, "--cache-dir", temp_cache_dir])
        assert result.exit_code == 0
        assert f"Cassette: {content_hash}" in result.output
        assert "gpt-4" in result.output
        assert "Prompt" in result.output
        assert "Replies" in result.output

    def test_cassette_show_json(self, runner, temp_cache_dir):
        """Test showing cassette details as JSON."""
        content_hash = self._store_entry(temp_cache_dir, message="JSON detail test")

        result = runner.invoke(main, ["cassette", "show", content_hash, "--cache-dir", temp_cache_dir, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["id"] == content_hash
        assert "sections" in data
        assert "sampling" in data

    def test_cassette_show_prefix(self, runner, temp_cache_dir):
        """Test showing cassette by prefix instead of full hash."""
        content_hash = self._store_entry(temp_cache_dir, message="Prefix show test")
        prefix = content_hash[:6]

        result = runner.invoke(main, ["cassette", "show", prefix, "--cache-dir", temp_cache_dir])
        assert result.exit_code == 0
        assert f"Cassette: {content_hash}" in result.output

    def test_cassette_show_not_found(self, runner, temp_cache_dir):
        """Test showing a non-existent cassette."""
        result = runner.invoke(main, ["cassette", "show", "nonexistent123", "--cache-dir", temp_cache_dir])
        assert result.exit_code == 1
        assert "no cassette found" in result.output.lower()

    def test_cassette_read(self, runner, temp_cache_dir):
        """Test reading full completion text."""
        content_hash = self._store_entry(temp_cache_dir, message="Read my reply")

        result = runner.invoke(main, ["cassette", "read", content_hash, "--cache-dir", temp_cache_dir])
        assert result.exit_code == 0
        assert "Reply to: Read my reply" in result.output

    def test_cassette_read_json(self, runner, temp_cache_dir):
        """Test reading completion as JSON."""
        content_hash = self._store_entry(temp_cache_dir, message="JSON read test")

        result = runner.invoke(main, ["cassette", "read", content_hash, "--cache-dir", temp_cache_dir, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "replies" in data
        assert len(data["replies"]) >= 1

    def test_cassette_read_with_prompt(self, runner, temp_cache_dir):
        """Test reading with --prompt includes prompt messages."""
        content_hash = self._store_entry(temp_cache_dir, message="Include my prompt")

        result = runner.invoke(main, ["cassette", "read", content_hash, "--cache-dir", temp_cache_dir, "--prompt"])
        assert result.exit_code == 0
        assert "Prompt" in result.output
        assert "Include my prompt" in result.output

    def test_cassette_delete(self, runner, temp_cache_dir):
        """Test deleting a cassette with --yes flag."""
        content_hash = self._store_entry(temp_cache_dir, message="Delete me")

        result = runner.invoke(main, ["cassette", "delete", content_hash, "--cache-dir", temp_cache_dir, "--yes"])
        assert result.exit_code == 0
        assert "Deleted" in result.output

        # Verify it's gone
        storage = CacheStorage(temp_cache_dir)
        assert content_hash not in storage.index

    def test_cassette_delete_abort(self, runner, temp_cache_dir):
        """Test that delete without --yes prompts and can be aborted."""
        content_hash = self._store_entry(temp_cache_dir, message="Do not delete")

        result = runner.invoke(main, ["cassette", "delete", content_hash, "--cache-dir", temp_cache_dir], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output

        # Verify it's still there
        storage = CacheStorage(temp_cache_dir)
        assert content_hash in storage.index

    def test_cassette_delete_not_found(self, runner, temp_cache_dir):
        """Test deleting a non-existent cassette."""
        result = runner.invoke(main, ["cassette", "delete", "nonexistent123", "--cache-dir", temp_cache_dir, "--yes"])
        assert result.exit_code == 1

    def test_cassette_stats_empty(self, runner, temp_cache_dir):
        """Test stats on empty cache."""
        result = runner.invoke(main, ["cassette", "stats", "--cache-dir", temp_cache_dir])
        assert result.exit_code == 0
        assert "Total cassettes: 0" in result.output

    def test_cassette_stats_with_entries(self, runner, temp_cache_dir):
        """Test stats with cached entries."""
        self._store_entry(temp_cache_dir, model="gpt-4", message="Stats test 1")
        self._store_entry(temp_cache_dir, model="claude-3", message="Stats test 2")

        result = runner.invoke(main, ["cassette", "stats", "--cache-dir", temp_cache_dir])
        assert result.exit_code == 0
        assert "Total cassettes: 2" in result.output
        assert "gpt-4" in result.output
        assert "claude-3" in result.output

    def test_cassette_stats_json(self, runner, temp_cache_dir):
        """Test stats with --json output."""
        self._store_entry(temp_cache_dir, message="JSON stats test")

        result = runner.invoke(main, ["cassette", "stats", "--cache-dir", temp_cache_dir, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total_entries"] == 1
        assert "disk_size_bytes" in data
        assert "entries_by_model" in data

    def test_cassette_reindex(self, runner, temp_cache_dir):
        """Test reindex command."""
        self._store_entry(temp_cache_dir, message="Reindex test")

        result = runner.invoke(main, ["cassette", "reindex", "--cache-dir", temp_cache_dir])
        assert result.exit_code == 0
        assert "Reindexed 1 tape files" in result.output


class TestTestGateCommand:
    """
    Tests for the test-gate command.

    Note: These tests only verify the command structure and error handling.
    Actual connectivity tests require a running InferenceGate instance.
    """

    def test_test_gate_help(self, runner):
        """
        Test that test-gate --help shows correct options (host, port, model, prompt, stream).
        """
        result = runner.invoke(main, ["test-gate", "--help"])
        assert result.exit_code == 0
        assert "host" in result.output
        assert "port" in result.output
        assert "model" in result.output
        assert "prompt" in result.output
        assert "--stream" in result.output
        assert "--no-stream" in result.output
        # Should NOT have api-key or upstream — those belong to test-upstream
        assert "api-key" not in result.output
        assert "upstream" not in result.output

    def test_test_gate_connection_refused(self, runner):
        """
        Test that test-gate reports connection error when no instance is running.
        """
        # Use a port that shouldn't have anything running
        result = runner.invoke(main, ["test-gate", "--port", "19999"])
        assert result.exit_code == 1
        assert "Connection refused" in result.output

    def test_test_gate_streaming_connection_refused(self, runner):
        """
        Test that test-gate --stream reports connection error when no instance is running.
        """
        result = runner.invoke(main, ["test-gate", "--port", "19999", "--stream"])
        assert result.exit_code == 1
        assert "Connection refused" in result.output


class TestTestUpstreamCommand:
    """
    Tests for the test-upstream command.

    Note: These tests only verify the command structure and error handling.
    Actual API connectivity tests require mocking the HTTP client.
    """

    def test_test_upstream_help(self, runner):
        """
        Test that test-upstream --help shows correct options (upstream, api-key, model, prompt, stream).
        """
        result = runner.invoke(main, ["test-upstream", "--help"])
        assert result.exit_code == 0
        assert "upstream" in result.output
        assert "api-key" in result.output
        assert "model" in result.output
        assert "prompt" in result.output
        assert "--stream" in result.output
        assert "--no-stream" in result.output

    def test_test_upstream_no_api_key_error(self, runner, tmp_path):
        """
        Test that test-upstream command fails with helpful error when no upstream/API key provided.
        """
        # Use a temporary config file (config no longer carries upstream/api_key globally;
        # caller must pass --upstream explicitly).
        config_path = tmp_path / "empty_config.yaml"
        config_path.write_text("port: 8080\n")
        result = runner.invoke(main, ["--config", str(config_path), "test-upstream"], env={"OPENAI_API_KEY": ""})
        assert result.exit_code == 1
        # Either error is acceptable — we just want the command to refuse to run without upstream config.
        assert ("No upstream URL provided" in result.output) or ("No API key provided" in result.output)


class TestConfigCommands:
    """
    Tests for configuration management commands.
    """

    def test_config_help(self, runner):
        """
        Test that config --help shows available subcommands.
        """
        result = runner.invoke(main, ["config", "--help"])
        assert result.exit_code == 0
        assert "show" in result.output
        assert "init" in result.output
        assert "path" in result.output

    def test_config_show(self, runner):
        """
        Test that config show displays current configuration (new endpoints/models schema).
        """
        result = runner.invoke(main, ["config", "show"])
        assert result.exit_code == 0
        assert "host:" in result.output
        assert "port:" in result.output

    def test_config_path(self, runner):
        """
        Test that config path shows the configuration file path.
        """
        result = runner.invoke(main, ["config", "path"])
        assert result.exit_code == 0
        assert ".InferenceGate" in result.output
        assert "config.yaml" in result.output

    def test_config_init(self, runner, tmp_path):
        """
        Test that config init creates a configuration file.
        """
        config_path = tmp_path / "test_config.yaml"
        result = runner.invoke(main, ["--config", str(config_path), "config", "init"])
        assert result.exit_code == 0
        assert "Created default configuration file" in result.output
        assert config_path.exists()

    def test_config_init_no_overwrite(self, runner, tmp_path):
        """
        Test that config init doesn't overwrite existing file without --force.
        """
        config_path = tmp_path / "test_config.yaml"
        config_path.touch()

        result = runner.invoke(main, ["--config", str(config_path), "config", "init"])
        assert result.exit_code == 0
        assert "already exists" in result.output

    def test_config_init_force(self, runner, tmp_path):
        """
        Test that config init --force overwrites existing file.
        """
        config_path = tmp_path / "test_config.yaml"
        config_path.touch()

        result = runner.invoke(main, ["--config", str(config_path), "config", "init", "--force"])
        assert result.exit_code == 0
        assert "Created default configuration file" in result.output

    def test_custom_config_path(self, runner, tmp_path):
        """
        Test that --config option uses custom config path.
        """
        config_path = tmp_path / "custom_config.yaml"
        result = runner.invoke(main, ["--config", str(config_path), "config", "path"])
        assert result.exit_code == 0
        assert str(config_path) in result.output


class TestCassetteFillCommand:
    """Tests for the cassette fill command."""

    @staticmethod
    def _store_non_greedy_entry(cache_dir: str, model: str = "gpt-4", message: str = "Hello", temperature: float = 0.7,
                                max_replies: int = 5) -> str:
        """Helper to store a non-greedy cassette entry and return its content_hash."""
        storage = CacheStorage(cache_dir)
        entry = CacheEntry(
            request=CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": message
                }],
                "temperature": temperature,
            }),
            response=CachedResponse(
                status_code=200, headers={}, body={
                    "choices": [{
                        "message": {
                            "content": f"Reply 1 to: {message}"
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20
                    },
                }),
            model=model,
            temperature=temperature,
        )
        return storage.put(entry, max_replies=max_replies)

    def test_fill_help(self, runner):
        """Test that cassette fill --help shows correct options."""
        result = runner.invoke(main, ["cassette", "fill", "--help"])
        assert result.exit_code == 0
        assert "count" in result.output
        assert "upstream" in result.output
        assert "api-key" in result.output
        assert "non-greedy" in result.output.lower()

    def test_fill_greedy_cassette_rejected(self, runner, temp_cache_dir):
        """Test that fill rejects greedy cassettes (temperature=0)."""
        storage = CacheStorage(temp_cache_dir)
        entry = CacheEntry(
            request=CachedRequest(method="POST", path="/v1/chat/completions", headers={}, body={
                "model": "gpt-4",
                "messages": [{
                    "role": "user",
                    "content": "Greedy test"
                }],
                "temperature": 0.0,
            }),
            response=CachedResponse(status_code=200, headers={}, body={"choices": [{
                "message": {
                    "content": "OK"
                }
            }]}),
            model="gpt-4",
        )
        content_hash = storage.put(entry)

        result = runner.invoke(main, ["cassette", "fill", content_hash, "--cache-dir", temp_cache_dir, "--api-key", "test-key"])
        assert result.exit_code == 1
        assert "greedy" in result.output.lower()

    def test_fill_nonexistent_cassette(self, runner, temp_cache_dir):
        """Test that fill fails for non-existent cassette."""
        result = runner.invoke(main, ["cassette", "fill", "nonexistent123", "--cache-dir", temp_cache_dir, "--api-key", "test-key"])
        assert result.exit_code == 1
        assert "no cassette found" in result.output.lower()

    def test_fill_no_api_key(self, runner, temp_cache_dir, tmp_path):
        """Test that fill fails without API key."""
        content_hash = self._store_non_greedy_entry(temp_cache_dir)
        config_path = tmp_path / "empty_config.yaml"
        config_path.write_text(f"cache_dir: {temp_cache_dir}\n")

        result = runner.invoke(main,
                               ["--config", str(config_path), "cassette", "fill", content_hash, "--cache-dir", temp_cache_dir],
                               env={"OPENAI_API_KEY": ""})
        assert result.exit_code == 1
        assert "No API key" in result.output

    def test_fill_already_full(self, runner, temp_cache_dir):
        """Test that fill reports nothing to do when cassette already has enough replies."""
        content_hash = self._store_non_greedy_entry(temp_cache_dir, max_replies=1)

        result = runner.invoke(main,
                               ["cassette", "fill", content_hash, "--cache-dir", temp_cache_dir, "--api-key", "test-key", "--count", "1"])
        assert result.exit_code == 0
        assert "Nothing to do" in result.output

    @patch("inference_gate.cli._fill_cassette")
    def test_fill_calls_fill_function(self, mock_fill, runner, temp_cache_dir):
        """Test that fill command calls the async fill function with correct parameters."""
        content_hash = self._store_non_greedy_entry(temp_cache_dir, message="Fill me up")

        mock_fill.return_value = {
            "content_hash": content_hash,
            "added": 3,
            "duplicates": 1,
            "errors": 0,
            "attempts": 4,
            "total_replies": 4,
            "target": 5,
        }

        result = runner.invoke(main, [
            "cassette", "fill", content_hash, "--cache-dir", temp_cache_dir, "--api-key", "test-key", "--count", "5", "--upstream",
            "http://localhost:8000"
        ])
        assert result.exit_code == 0
        assert "3 new unique completions added" in result.output
        assert "1 duplicates" in result.output
        assert mock_fill.called

    @patch("inference_gate.cli._fill_cassette")
    def test_fill_json_output(self, mock_fill, runner, temp_cache_dir):
        """Test that fill command with --json outputs valid JSON."""
        content_hash = self._store_non_greedy_entry(temp_cache_dir, message="JSON fill test")

        mock_fill.return_value = {
            "content_hash": content_hash,
            "added": 2,
            "duplicates": 0,
            "errors": 0,
            "attempts": 2,
            "total_replies": 3,
            "target": 5,
        }

        result = runner.invoke(main, ["cassette", "fill", content_hash, "--cache-dir", temp_cache_dir, "--api-key", "test-key", "--count", "5", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["added"] == 2
        assert data["total_replies"] == 3

    @patch("inference_gate.cli._fill_cassette")
    def test_fill_with_errors(self, mock_fill, runner, temp_cache_dir):
        """Test that fill command reports errors from upstream."""
        content_hash = self._store_non_greedy_entry(temp_cache_dir, message="Error fill test")

        mock_fill.return_value = {
            "content_hash": content_hash,
            "added": 1,
            "duplicates": 0,
            "errors": 3,
            "attempts": 4,
            "total_replies": 2,
            "target": 5,
        }

        result = runner.invoke(main,
                               ["cassette", "fill", content_hash, "--cache-dir", temp_cache_dir, "--api-key", "test-key", "--count", "5"])
        assert result.exit_code == 0
        assert "3 errors" in result.output
        assert "Warning" in result.output
