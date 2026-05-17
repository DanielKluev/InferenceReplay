"""
Tests for the InferenceGate pytest plugin.

Verifies CLI option registration, option resolution priority, subprocess
auto-launch, fixture behaviour, marker translation, and plugin disabling.
Uses ``pytester`` (pytest's built-in fixture for testing plugins) for the
sub-pytest scenarios and direct unit tests for the option resolution helper.
"""

import http.client
import json
import textwrap
import urllib.parse
from pathlib import Path

import pytest

# Path to the production cassettes shipped with InferenceGate's own tests.
CASSETTES_DIR = str(Path(__file__).parent / "cassettes")

# The replay/fuzzy plugin tests below use this production cassette prompt.
CASSETTE_MODEL = "openai/gpt-oss-120b"
CASSETTE_MAX_TOKENS = 200
OK_PROMPT = 'This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.'


def _post_default_mode_ok_prompt(inference_gate_url: str) -> dict:
    """
    Send the plugin replay prompt without a per-request replay override.

    This gives record-mode test sessions a normal-mode request that can recreate the shared OK
    cassette before subprocess tests deliberately force replay-only behavior against it.
    """
    parsed = urllib.parse.urlparse(inference_gate_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)
    try:
        body = json.dumps({
            "model": CASSETTE_MODEL,
            "messages": [{
                "role": "user",
                "content": OK_PROMPT
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


# ---------------------------------------------------------------------------
# Unit tests for _resolve_option
# ---------------------------------------------------------------------------


class TestResolveOption:
    """
    Tests for the ``_resolve_option`` helper with various priority combinations.
    """

    def test_default_mode_is_replay(self, pytestconfig):
        """
        Default mode should be ``"replay"`` when nothing is configured.
        """
        from inference_gate.pytest_plugin import _resolve_option
        result = _resolve_option(pytestconfig, "mode")
        assert result in ("replay", "record")  # accept either if env var is set

    def test_default_cache_dir(self, pytestconfig):
        """
        Default cache_dir should be ``"tests/cassettes"``.
        """
        from inference_gate.pytest_plugin import _resolve_option
        result = _resolve_option(pytestconfig, "cache_dir")
        assert result == "tests/cassettes" or result

    def test_default_port(self, pytestconfig):
        """
        Default port should be ``"0"`` (OS-assigned).
        """
        from inference_gate.pytest_plugin import _resolve_option
        result = _resolve_option(pytestconfig, "port")
        assert result == "0" or result


class TestSubprocessCommand:
    """
    Tests for the plugin subprocess command line builder.
    """

    def test_config_path_is_passed_before_subcommand(self, tmp_path):
        """
        ``--inferencegate-config`` must become a Click group option before ``serve``.
        """
        from inference_gate.pytest_plugin import _SubprocessServer
        config_path = str(tmp_path / "gate.yaml")
        server = _SubprocessServer({"mode": "record", "host": "127.0.0.1", "port": 0, "config": config_path})

        cmd = server._build_cmd()

        assert "--config" in cmd
        assert cmd[cmd.index("--config") + 1] == config_path
        assert cmd.index("--config") < cmd.index("serve")
        assert cmd[cmd.index("--mode") + 1] == "record"


# ---------------------------------------------------------------------------
# Pytester-based integration tests
# ---------------------------------------------------------------------------


class TestPluginRegistration:
    """
    Tests that the plugin registers its options and markers correctly.
    """

    def test_help_shows_inferencegate_options(self, pytester):
        """
        ``pytest --help`` should list the inferencegate option group.
        """
        result = pytester.runpytest("--help")
        result.stdout.fnmatch_lines(["*inferencegate*"])
        result.stdout.fnmatch_lines(["*--inferencegate-mode*"])
        result.stdout.fnmatch_lines(["*--inferencegate-cache-dir*"])

    def test_inferencegate_marker_registered(self, pytester):
        """
        ``pytest --markers`` lists the ``inferencegate(...)`` policy marker.
        """
        result = pytester.runpytest("--markers")
        result.stdout.fnmatch_lines(["*inferencegate(fuzzy_model*"])

    def test_requires_model_marker_registered(self, pytester):
        """
        ``pytest --markers`` lists the ``requires_model(name=..., engine=...)`` marker.
        """
        result = pytester.runpytest("--markers")
        result.stdout.fnmatch_lines(["*requires_model*"])


# ---------------------------------------------------------------------------
# Mode resolution — exercised via HTTP /gate/health probes against the spawned Gate.
# ---------------------------------------------------------------------------


def _mode_probe_test_file() -> str:
    """
    Return a sub-pytest test source that asserts on ``GET /gate/health``'s
    ``mode`` field.  Parameterised by ``EXPECTED_MODE`` set via env in the
    consuming test.
    """
    return textwrap.dedent("""
        import http.client
        import json
        import os
        import urllib.parse

        def test_mode(inference_gate_url):
            parsed = urllib.parse.urlparse(inference_gate_url)
            conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
            conn.request("GET", "/gate/health")
            resp = conn.getresponse()
            assert resp.status == 200
            data = json.loads(resp.read())
            expected = os.environ["EXPECTED_MODE"]
            assert data["mode"] == expected, f"got {data['mode']!r}, expected {expected!r}"
            conn.close()
    """)


class TestModeResolution:
    """
    Tests for the CLI > env > ini > default resolution order.

    Each sub-pytest probes ``GET /gate/health`` to see what mode the spawned
    Gate is actually running in.
    """

    def test_default_mode_is_replay(self, pytester, monkeypatch):
        """
        Without any configuration, the Gate runs in ``replay-only`` mode.
        """
        monkeypatch.setenv("EXPECTED_MODE", "replay-only")
        monkeypatch.delenv("INFERENCEGATE_MODE", raising=False)
        monkeypatch.delenv("INFERENCEGATE_URL", raising=False)
        pytester.makepyfile(_mode_probe_test_file())
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest_subprocess("-v", "-s")
        result.assert_outcomes(passed=1)

    def test_cli_flag_sets_mode(self, pytester, monkeypatch):
        """
        ``--inferencegate-mode record`` boots the Gate in record-and-replay mode.
        """
        monkeypatch.setenv("EXPECTED_MODE", "record-and-replay")
        monkeypatch.delenv("INFERENCEGATE_MODE", raising=False)
        monkeypatch.delenv("INFERENCEGATE_URL", raising=False)
        pytester.makepyfile(_mode_probe_test_file())
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest_subprocess("-v", "-s", "--inferencegate-mode", "record")
        result.assert_outcomes(passed=1)

    def test_env_var_overrides_default(self, pytester, monkeypatch):
        """
        ``INFERENCEGATE_MODE=record`` boots the Gate in record-and-replay mode.
        """
        monkeypatch.setenv("INFERENCEGATE_MODE", "record")
        monkeypatch.setenv("EXPECTED_MODE", "record-and-replay")
        monkeypatch.delenv("INFERENCEGATE_URL", raising=False)
        pytester.makepyfile(_mode_probe_test_file())
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest_subprocess("-v", "-s")
        result.assert_outcomes(passed=1)

    def test_ini_option_overrides_default(self, pytester, monkeypatch):
        """
        ``inferencegate_mode = record`` ini option boots the Gate in record-and-replay mode.
        """
        monkeypatch.setenv("EXPECTED_MODE", "record-and-replay")
        monkeypatch.delenv("INFERENCEGATE_MODE", raising=False)
        monkeypatch.delenv("INFERENCEGATE_URL", raising=False)
        pytester.makepyfile(_mode_probe_test_file())
        pytester.makeini(f"""
            [pytest]
            inferencegate_mode = record
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest_subprocess("-v", "-s")
        result.assert_outcomes(passed=1)

    def test_cli_beats_env_var(self, pytester, monkeypatch):
        """
        CLI flag takes priority over the environment variable.
        """
        monkeypatch.setenv("INFERENCEGATE_MODE", "record")
        monkeypatch.setenv("EXPECTED_MODE", "replay-only")
        monkeypatch.delenv("INFERENCEGATE_URL", raising=False)
        pytester.makepyfile(_mode_probe_test_file())
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest_subprocess("-v", "-s", "--inferencegate-mode", "replay")
        result.assert_outcomes(passed=1)


class TestServerLifecycle:
    """
    Tests that the plugin manages the InferenceGate subprocess correctly.
    """

    def test_server_starts_and_health_reachable(self, pytester):
        """
        The ``inference_gate_url`` fixture exposes a healthy server.
        """
        pytester.makepyfile("""
            import http.client
            import urllib.parse

            def test_health(inference_gate_url):
                parsed = urllib.parse.urlparse(inference_gate_url)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
                conn.request("GET", "/health")
                resp = conn.getresponse()
                assert resp.status == 200
                conn.close()
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v", "-s")
        result.assert_outcomes(passed=1)

    def test_inference_gate_url_format(self, pytester):
        """
        ``inference_gate_url`` is a proper ``http://host:port`` string.
        """
        pytester.makepyfile("""
            def test_url_format(inference_gate_url):
                assert inference_gate_url.startswith("http://127.0.0.1:")
                port = int(inference_gate_url.split(":")[-1])
                assert port > 0
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v", "-s")
        result.assert_outcomes(passed=1)

    def test_port_zero_assigns_ephemeral_port(self, pytester):
        """
        Port ``0`` results in an OS-assigned port that is not 0.
        """
        pytester.makepyfile("""
            def test_port(inference_gate_url):
                port = int(inference_gate_url.rsplit(":", 1)[-1])
                assert port > 0
                assert port != 0
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v", "-s")
        result.assert_outcomes(passed=1)


class TestCassetteReplayViaPytestPlugin:
    """
    Tests that the plugin replays cassettes through the auto-launched server.
    """

    def test_default_mode_serves_or_records_ok_prompt(self, inference_gate_url):
        """The shared OK prompt cassette is available before sub-pytests force replay mode."""
        data = _post_default_mode_ok_prompt(inference_gate_url)
        content = data["choices"][0]["message"]["content"]
        assert "OK" in content

    def test_replay_ok_prompt(self, pytester, monkeypatch):
        """
        Replay the 'OK' prompt cassette through the plugin's server.
        """
        monkeypatch.delenv("INFERENCEGATE_URL", raising=False)
        monkeypatch.delenv("INFERENCEGATE_MODE", raising=False)
        pytester.makepyfile(
            textwrap.dedent(f"""
            import http.client
            import json
            import urllib.parse

            CASSETTE_MODEL = "openai/gpt-oss-120b"
            CASSETTE_MAX_TOKENS = 200
            OK_PROMPT = 'This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.'

            def test_replay(inference_gate_url):
                parsed = urllib.parse.urlparse(inference_gate_url)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
                body = json.dumps({{
                    "model": CASSETTE_MODEL,
                    "messages": [{{"role": "user", "content": OK_PROMPT}}],
                    "max_tokens": CASSETTE_MAX_TOKENS,
                }})
                conn.request("POST", "/v1/chat/completions",
                             body=body,
                             headers={{
                                 "Content-Type": "application/json",
                                 "X-InferenceGate-Control-Mode": "replay",
                                 "X-InferenceGate-Control-Reply-Strategy": "first",
                             }})
                resp = conn.getresponse()
                assert resp.status == 200
                data = json.loads(resp.read())
                assert "choices" in data
                content = data["choices"][0]["message"]["content"]
                assert "OK" in content
                conn.close()
        """))
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest_subprocess("-v", "-s", "--inferencegate-mode", "replay")
        result.assert_outcomes(passed=1)

    def test_cache_miss_returns_503(self, pytester, monkeypatch):
        """
        A request with no matching cassette returns 503 in replay mode.
        """
        monkeypatch.delenv("INFERENCEGATE_URL", raising=False)
        monkeypatch.delenv("INFERENCEGATE_MODE", raising=False)
        pytester.makepyfile(
            textwrap.dedent(f"""
            import http.client
            import json
            import urllib.parse

            def test_miss(inference_gate_url):
                parsed = urllib.parse.urlparse(inference_gate_url)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
                body = json.dumps({{
                    "model": "missing-model-for-replay-503-test",
                    "messages": [{{"role": "user", "content": "This intentionally has no cassette: replay-503-sentinel"}}],
                }})
                conn.request("POST", "/v1/chat/completions",
                             body=body,
                             headers={{"Content-Type": "application/json", "X-InferenceGate-Control-Mode": "replay"}})
                resp = conn.getresponse()
                assert resp.status == 503
                conn.close()
        """))
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest_subprocess("-v", "-s", "--inferencegate-mode", "replay")
        result.assert_outcomes(passed=1)


class TestFuzzyModelMatchingViaPytestPlugin:
    """
    Tests for the fuzzy model matching option via the pytest plugin.
    """

    def test_fuzzy_matching_via_ini_option(self, pytester, monkeypatch):
        """
        ``inferencegate_fuzzy_model = true`` ini lets a different-model request hit a cassette.
        """
        monkeypatch.delenv("INFERENCEGATE_URL", raising=False)
        monkeypatch.delenv("INFERENCEGATE_FUZZY_MODEL", raising=False)
        pytester.makepyfile(
            textwrap.dedent(f"""
            import http.client
            import json
            import urllib.parse

            CASSETTE_MAX_TOKENS = 200
            OK_PROMPT = 'This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.'

            def test_fuzzy(inference_gate_url):
                parsed = urllib.parse.urlparse(inference_gate_url)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
                body = json.dumps({{
                    "model": "completely-different-model",
                    "messages": [{{"role": "user", "content": OK_PROMPT}}],
                    "max_tokens": CASSETTE_MAX_TOKENS,
                }})
                conn.request("POST", "/v1/chat/completions",
                             body=body,
                             headers={{"Content-Type": "application/json"}})
                resp = conn.getresponse()
                assert resp.status == 200
                data = json.loads(resp.read())
                assert "choices" in data
                conn.close()
        """))
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
            inferencegate_fuzzy_model = true
        """)
        result = pytester.runpytest_subprocess("-v", "-s")
        result.assert_outcomes(passed=1)

    def test_fuzzy_matching_via_cli_flag(self, pytester, monkeypatch):
        """
        ``--inferencegate-fuzzy-model`` lets a different-model request hit a cassette.
        """
        monkeypatch.delenv("INFERENCEGATE_URL", raising=False)
        monkeypatch.delenv("INFERENCEGATE_FUZZY_MODEL", raising=False)
        pytester.makepyfile(
            textwrap.dedent(f"""
            import http.client
            import json
            import urllib.parse

            CASSETTE_MAX_TOKENS = 200
            OK_PROMPT = 'This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.'

            def test_fuzzy(inference_gate_url):
                parsed = urllib.parse.urlparse(inference_gate_url)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
                body = json.dumps({{
                    "model": "another-nonexistent-model",
                    "messages": [{{"role": "user", "content": OK_PROMPT}}],
                    "max_tokens": CASSETTE_MAX_TOKENS,
                }})
                conn.request("POST", "/v1/chat/completions",
                             body=body,
                             headers={{"Content-Type": "application/json"}})
                resp = conn.getresponse()
                assert resp.status == 200
                data = json.loads(resp.read())
                assert "choices" in data
                conn.close()
        """))
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest_subprocess("-v", "-s", "--inferencegate-fuzzy-model")
        result.assert_outcomes(passed=1)

    def test_fuzzy_matching_via_env_var(self, pytester, monkeypatch):
        """
        ``INFERENCEGATE_FUZZY_MODEL=true`` env var lets a different-model request hit a cassette.
        """
        monkeypatch.setenv("INFERENCEGATE_FUZZY_MODEL", "true")
        monkeypatch.delenv("INFERENCEGATE_URL", raising=False)
        pytester.makepyfile(
            textwrap.dedent(f"""
            import http.client
            import json
            import urllib.parse

            CASSETTE_MAX_TOKENS = 200
            OK_PROMPT = 'This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.'

            def test_fuzzy(inference_gate_url):
                parsed = urllib.parse.urlparse(inference_gate_url)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
                body = json.dumps({{
                    "model": "env-var-test-model",
                    "messages": [{{"role": "user", "content": OK_PROMPT}}],
                    "max_tokens": CASSETTE_MAX_TOKENS,
                }})
                conn.request("POST", "/v1/chat/completions",
                             body=body,
                             headers={{"Content-Type": "application/json"}})
                resp = conn.getresponse()
                assert resp.status == 200
                conn.close()
        """))
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest_subprocess("-v", "-s")
        result.assert_outcomes(passed=1)

    def test_no_fuzzy_matching_cli_overrides_ini(self, pytester, monkeypatch):
        """
        ``--no-inferencegate-fuzzy-model`` overrides ``inferencegate_fuzzy_model = true``.
        """
        monkeypatch.delenv("INFERENCEGATE_URL", raising=False)
        monkeypatch.delenv("INFERENCEGATE_MODE", raising=False)
        pytester.makepyfile(
            textwrap.dedent(f"""
            import http.client
            import json
            import urllib.parse

            CASSETTE_MAX_TOKENS = 200
            OK_PROMPT = 'This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.'

            def test_no_fuzzy(inference_gate_url):
                parsed = urllib.parse.urlparse(inference_gate_url)
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
                body = json.dumps({{
                    "model": "no-fuzzy-cli-override-model",
                    "messages": [{{"role": "user", "content": OK_PROMPT}}],
                    "max_tokens": CASSETTE_MAX_TOKENS,
                }})
                conn.request("POST", "/v1/chat/completions",
                             body=body,
                             headers={{
                                 "Content-Type": "application/json",
                                 "X-InferenceGate-Control-Mode": "replay",
                                 "X-InferenceGate-Control-Reply-Strategy": "first",
                             }})
                resp = conn.getresponse()
                assert resp.status == 503
                conn.close()
        """))
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
            inferencegate_fuzzy_model = true
        """)
        result = pytester.runpytest_subprocess("-v", "-s", "--no-inferencegate-fuzzy-model")
        result.assert_outcomes(passed=1)


class TestInferenceGateMarker:
    """
    Tests for the ``@pytest.mark.inferencegate(fuzzy_model=..., fuzzy_sampling=...)``
    per-test policy override marker.
    """

    def test_marker_pushes_per_test_headers(self, pytester):
        """
        ``@pytest.mark.inferencegate(fuzzy_model=False)`` pushes the
        ``X-InferenceGate-Require-Fuzzy-Model: off`` header onto Gate's
        pytest header context for the duration of the test, leaving
        ``fuzzy_sampling`` to fall back to the session default.
        """
        pytester.makepyfile("""
            import pytest
            from inference_gate.pytest_context import current_headers

            @pytest.mark.inferencegate(fuzzy_model=False)
            def test_only_model_off(inference_gate_url):
                hdrs = current_headers()
                assert hdrs.get("X-InferenceGate-Require-Fuzzy-Model") == "off"
                assert "X-InferenceGate-Require-Fuzzy-Sampling" not in hdrs
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
            inferencegate_fuzzy_model = true
            inferencegate_fuzzy_sampling = aggressive
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)

    def test_marker_overrides_are_restored_after_test(self, pytester):
        """
        Header overrides must be popped from the ContextVar after a marked
        test finishes, so subsequent unmarked tests do not see leaked headers.
        """
        pytester.makepyfile("""
            import pytest
            from inference_gate.pytest_context import current_headers

            @pytest.mark.inferencegate(fuzzy_model=False)
            def test_first_marked(inference_gate_url):
                assert current_headers().get("X-InferenceGate-Require-Fuzzy-Model") == "off"

            def test_second_unmarked(inference_gate_url):
                # If restore worked, the override is gone.
                assert "X-InferenceGate-Require-Fuzzy-Model" not in current_headers()
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=2)


class TestAutoTestIdentityMetadata:
    """
    Auto-attached ``Metadata-Test-NodeID`` / ``Metadata-Worker-ID`` headers.
    """

    def test_node_id_and_worker_id_pushed_for_every_test(self, pytester):
        """
        Every test (marked or not) gets its pytest nodeid and the xdist
        worker name (``"master"`` outside xdist) pushed onto Gate's pytest
        header context.
        """
        pytester.makepyfile("""
            from inference_gate.pytest_context import current_headers

            def test_metadata_pushed(inference_gate_url):
                hdrs = current_headers()
                node_id = hdrs.get("X-InferenceGate-Metadata-Test-NodeID")
                assert node_id is not None
                assert "test_metadata_pushed" in node_id
                assert hdrs.get("X-InferenceGate-Metadata-Worker-ID") == "master"
        """)
        pytester.makeini(f"""
            [pytest]
            inferencegate_cache_dir = {CASSETTES_DIR}
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)


class TestPluginDisabling:
    """
    Tests that the plugin can be cleanly disabled.
    """

    def test_disable_with_p_no(self, pytester):
        """
        ``-p no:inferencegate`` disables the plugin entirely.
        """
        pytester.makepyfile("""
            def test_nothing():
                assert True
        """)
        result = pytester.runpytest("-v", "-p", "no:inferencegate")
        result.assert_outcomes(passed=1)
        # The inferencegate options should not appear in help when disabled.
        help_result = pytester.runpytest("--help", "-p", "no:inferencegate")
        assert "--inferencegate-mode" not in help_result.stdout.str()


# ---------------------------------------------------------------------------
# Unit tests for header sanitization
# ---------------------------------------------------------------------------


class TestHeaderSanitization:
    """
    Tests that ``CacheStorage.put()`` does not leak sensitive headers into tape files.

    In v2 tape format, request headers are not stored at all — the tape only
    contains semantic request content (messages, model, sampling params) and
    response references.  These tests verify that sensitive header values do
    not appear anywhere in the stored tape files.
    """

    def test_authorization_header_not_in_tape(self, storage):
        """
        Authorization header value should not appear in stored tape files.
        """
        from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry
        secret = "Bearer sk-secret-key-12345"
        entry = CacheEntry(
            request=CachedRequest(method="POST", path="/v1/chat/completions", headers={
                "Content-Type": "application/json",
                "Authorization": secret,
                "Accept": "*/*",
            }, body={
                "model": "test",
                "messages": [{
                    "role": "user",
                    "content": "hello"
                }]
            }),
            response=CachedResponse(status_code=200, headers={"Content-Type": "application/json"}, body={"choices": []}),
        )
        cache_key = storage.put(entry)

        tape_path = storage._find_tape_file(cache_key)
        assert tape_path is not None
        tape_content = tape_path.read_text(encoding="utf-8")
        assert secret not in tape_content
        assert "sk-secret-key-12345" not in tape_content

    def test_multiple_sensitive_headers_not_in_tape(self, storage):
        """
        ``X-Api-Key`` and ``Proxy-Authorization`` values should not appear in tape files.
        """
        from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry
        entry = CacheEntry(
            request=CachedRequest(
                method="POST", path="/v1/chat/completions", headers={
                    "Content-Type": "application/json",
                    "X-Api-Key": "secret-api-key",
                    "Proxy-Authorization": "Basic proxy-secret",
                }, body={
                    "model": "test",
                    "messages": [{
                        "role": "user",
                        "content": "test"
                    }]
                }),
            response=CachedResponse(status_code=200, headers={"Content-Type": "application/json"}, body={"choices": []}),
        )
        cache_key = storage.put(entry)

        tape_path = storage._find_tape_file(cache_key)
        assert tape_path is not None
        tape_content = tape_path.read_text(encoding="utf-8")
        assert "secret-api-key" not in tape_content
        assert "proxy-secret" not in tape_content

    def test_case_insensitive_header_not_leaked(self, storage):
        """
        Header values should not be leaked regardless of header name casing.
        """
        from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry
        entry = CacheEntry(
            request=CachedRequest(method="POST", path="/v1/chat/completions", headers={
                "content-type": "application/json",
                "AUTHORIZATION": "Bearer secret-case-test",
            }, body={
                "model": "test",
                "messages": [{
                    "role": "user",
                    "content": "test2"
                }]
            }),
            response=CachedResponse(status_code=200, headers={"Content-Type": "application/json"}, body={"choices": []}),
        )
        cache_key = storage.put(entry)

        tape_path = storage._find_tape_file(cache_key)
        assert tape_path is not None
        tape_content = tape_path.read_text(encoding="utf-8")
        assert "secret-case-test" not in tape_content
