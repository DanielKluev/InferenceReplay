"""
Pytest plugin for InferenceGate integration.

Auto-spawns a subprocess InferenceGate proxy for the test session and exposes
its base URL via the ``inference_gate_url`` fixture.  Tests configure their
AI clients to point at ``f"{inference_gate_url}/v1"``.

Subprocess mode is the only supported deployment.  This guarantees a single
shared Gate across both single-process and ``pytest-xdist`` sessions, so
cassettes recorded by any worker are visible to all others without thread
juggling or per-test state leakage.

Usage in an external project::

    # pyproject.toml
    [tool.pytest.ini_options]
    inferencegate_mode = "replay"
    inferencegate_cache_dir = "tests/cassettes"

    # conftest.py
    # nothing needed; plugin auto-discovers via entry point.

    # test_example.py
    def test_ai_feature(inference_gate_url):
        from openai import OpenAI
        client = OpenAI(base_url=f"{inference_gate_url}/v1", api_key="unused")
        ...

The consuming project pushes endpoint and routing config to the running Gate
via ``POST /gate/config`` — typically once at session start, from the project
conftest.  This plugin no longer reads upstream URLs from any config file.
"""

import logging
import os
import subprocess
import sys
import threading
import time
from typing import Any, Generator

import pytest

from inference_gate import pytest_context

log = logging.getLogger("InferenceGatePlugin")

# ---------------------------------------------------------------------------
# Option resolution
# ---------------------------------------------------------------------------

# Each entry maps a logical option name to its CLI flag, env var, ini key and
# default.  ``_resolve_option`` walks these in priority order.
_OPTION_DEFS: dict[str, dict[str, Any]] = {
    "mode": {
        "cli": "--inferencegate-mode",
        "env": "INFERENCEGATE_MODE",
        "ini": "inferencegate_mode",
        "default": "replay",
    },
    "cache_dir": {
        "cli": "--inferencegate-cache-dir",
        "env": "INFERENCEGATE_CACHE_DIR",
        "ini": "inferencegate_cache_dir",
        "default": "tests/cassettes",
    },
    "config": {
        "cli": "--inferencegate-config",
        "env": "INFERENCEGATE_CONFIG",
        "ini": "inferencegate_config",
        "default": None,
    },
    "port": {
        "cli": "--inferencegate-port",
        "env": "INFERENCEGATE_PORT",
        "ini": "inferencegate_port",
        "default": "0",
    },
    "fuzzy_model": {
        "cli": "inferencegate_fuzzy_model",
        "env": "INFERENCEGATE_FUZZY_MODEL",
        "ini": "inferencegate_fuzzy_model",
        # Fuzzy matching is OFF by default.  Tests that want it must opt in
        # explicitly via the ``inferencegate`` marker or the ini setting.
        "default": False,
    },
    "fuzzy_sampling": {
        "cli": "--inferencegate-fuzzy-sampling",
        "env": "INFERENCEGATE_FUZZY_SAMPLING",
        "ini": "inferencegate_fuzzy_sampling",
        "default": "off",
    },
    "max_non_greedy_replies": {
        "cli": "--inferencegate-max-non-greedy-replies",
        "env": "INFERENCEGATE_MAX_NON_GREEDY_REPLIES",
        "ini": "inferencegate_max_non_greedy_replies",
        "default": "5",
    },
    "record_timeout": {
        "cli": "--inferencegate-record-timeout",
        "env": "INFERENCEGATE_RECORD_TIMEOUT",
        "ini": "inferencegate_record_timeout",
        # 600s = generous default for slow upstream recording (large local
        # models can easily take a few minutes per response).  Gate passes
        # this to its upstream HTTP client; consuming projects should align
        # their pytest timeout so a slow record does not silently abort.
        "default": "600",
    },
    "max_live_requests": {
        "cli": "--inferencegate-max-live-requests",
        "env": "INFERENCEGATE_MAX_LIVE_REQUESTS",
        "ini": "inferencegate_max_live_requests",
        "default": None,
    },
}


def _resolve_option(config: pytest.Config, name: str) -> Any:
    """
    Resolve an InferenceGate option using CLI > env var > ini > default.

    `name` is a key in :data:`_OPTION_DEFS`.  Boolean CLI flags (e.g.
    ``fuzzy_model``) may return ``bool`` directly; everything else is
    returned as the underlying type produced by pytest/argparse.
    """
    defn = _OPTION_DEFS[name]

    # 1. CLI flag (highest priority)
    cli_val = config.getoption(defn["cli"], default=None)
    if cli_val is not None:
        return cli_val

    # 2. Environment variable
    env_val = os.environ.get(defn["env"])
    if env_val:
        return env_val

    # 3. Ini option
    ini_val = config.getini(defn["ini"])
    if ini_val:
        return ini_val

    # 4. Default
    return defn["default"]


# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Register InferenceGate CLI arguments and ini options.
    """
    group = parser.getgroup("inferencegate", "InferenceGate inference replay proxy")

    group.addoption("--inferencegate-mode", choices=["replay", "record"], default=None,
                    help='Operating mode: "replay" (cached only, default) or "record" (cache + upstream).')
    group.addoption("--inferencegate-cache-dir", default=None, help="Directory for cached cassettes (default: tests/cassettes).")
    group.addoption("--inferencegate-config", default=None, help="Path to InferenceGate config.yaml (default: auto-detect).")
    group.addoption("--inferencegate-port", default=None, help="Proxy server port. 0 = OS-assigned (default: 0).")
    group.addoption("--inferencegate-fuzzy-model", action="store_true", dest="inferencegate_fuzzy_model", default=None,
                    help="Enable fuzzy model matching: on cache miss, reuse entries with the same prompt but a different model.")
    group.addoption("--no-inferencegate-fuzzy-model", action="store_false", dest="inferencegate_fuzzy_model", default=None,
                    help="Disable fuzzy model matching, overriding ini/env settings for a single test run.")
    group.addoption("--inferencegate-fuzzy-sampling", default=None, choices=["off", "soft", "aggressive"],
                    help="Sampling parameter fuzzy matching level: off, soft, or aggressive.")
    group.addoption("--inferencegate-max-non-greedy-replies", default=None, type=int,
                    help="Max replies to collect per non-greedy cassette before cycling (default: 5).")
    group.addoption("--inferencegate-max-live-requests", default=None, type=int,
                    help="Global limit on the number of live upstream requests per session (default: infinite).")
    group.addoption("--inferencegate-record-timeout", default=None, type=float,
                    help="Default upstream HTTP timeout in seconds used during recording (default: 600).")

    parser.addini("inferencegate_mode", default="", help="InferenceGate operating mode: replay or record.")
    parser.addini("inferencegate_cache_dir", default="", help="Directory for cached cassettes.")
    parser.addini("inferencegate_config", default="", help="Path to InferenceGate config.yaml.")
    parser.addini("inferencegate_port", default="", help="Proxy server port. 0 = OS-assigned.")
    parser.addini("inferencegate_fuzzy_model", default="", help="Enable fuzzy model matching (true/false).")
    parser.addini("inferencegate_fuzzy_sampling", default="", help="Sampling fuzzy matching level: off, soft, or aggressive.")
    parser.addini("inferencegate_max_non_greedy_replies", default="", help="Max replies per non-greedy cassette.")
    parser.addini("inferencegate_max_live_requests", default="", help="Global limit on the number of live upstream requests.")
    parser.addini("inferencegate_record_timeout", default="", help="Default upstream HTTP timeout in seconds (record mode).")


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """
    Register custom markers, then — on the controller — spawn the shared
    subprocess Gate so all (xdist or single-process) tests see the same URL.

    Recording mode disables the project's pytest timeout because real
    upstream calls can legitimately take much longer than the per-test
    replay budget.  This is a no-op in replay sessions.

    Marked ``tryfirst=True`` so the record-mode timeout override lands before
    ``pytest-timeout``'s own ``pytest_configure`` reads
    ``config.getvalue("timeout")`` and caches the project's strict 2s budget.
    Without this ordering, the override is applied too late and recording
    sessions still abort mid-call.
    """
    config.addinivalue_line(
        "markers", "inferencegate(fuzzy_model=None, fuzzy_sampling=None): override cassette "
        "matching policy for a single test.  Either/both kwargs may be set; omitted kwargs "
        "keep the session default.")
    config.addinivalue_line(
        "markers", "requires_model(name=..., engine=None): declare that a test requires a "
        "specific model in the project's TEST_CONTRACT.  Validated at collection time by "
        "consuming projects' conftests via the InferenceGlue plugin.")

    # In recording mode, real upstream calls can blow past the project's strict
    # per-test timeout (typically 2s).  Disable timeouts globally for the
    # duration of recording sessions — cassette replay tests run elsewhere.
    mode = _resolve_option(config, "mode")
    if str(mode) == "record" and hasattr(config.option, "timeout"):
        config.option.timeout = 0
        log.info("InferenceGate record mode active — disabled pytest timeout for the session")

    # Always spawn the subprocess on the controller so xdist workers inherit
    # ``INFERENCEGATE_URL``; in single-process sessions the controller is the
    # only process so this still works.
    _maybe_spawn_subprocess(config)


def pytest_unconfigure(config: pytest.Config) -> None:
    """
    Tear down the subprocess Gate on session end (controller only).
    """
    _maybe_stop_subprocess(config)


# ---------------------------------------------------------------------------
# Subprocess management
# ---------------------------------------------------------------------------


class _SubprocessServer:
    """
    Runs an InferenceGate server in a child process via ``inference-gate serve``.

    The parent reads ``INFERENCEGATE_URL=<url>`` from the child's stdout once
    the server is listening, then exposes it via the env var.  A
    ``--parent-pid`` watchdog ensures the child exits when the parent dies,
    avoiding orphaned servers in CI.
    """

    def __init__(self, kwargs: dict[str, Any]) -> None:
        """
        Capture the kwargs that should be forwarded to ``inference-gate serve``.
        """
        self.kwargs = kwargs
        self._proc: subprocess.Popen | None = None
        self.base_url: str | None = None

    def start(self) -> str:
        """
        Spawn the subprocess and read the ``INFERENCEGATE_URL=...`` line.

        Returns the announced base URL.  Raises :class:`RuntimeError` when the
        child exits early or fails to announce within 30 seconds.
        """
        cmd = self._build_cmd()
        log.debug("Spawning InferenceGate subprocess: %s", cmd)
        # Pipe BOTH stdout and stderr so the child does not inherit the
        # parent's stderr handle.  Inheriting stderr causes pipelines on
        # Windows PowerShell (``pytest ... 2>&1 | ...``) to hang after
        # pytest exits because the child still holds a write-end of the
        # console pipe.  Draining stdout/stderr in daemon threads
        # surfaces child logs to the parent's stderr without sharing
        # the underlying handle.
        # ``encoding`` + ``errors='replace'`` are critical on Windows: the
        # default text-mode decoder uses the OEM/ANSI codepage (``cp932``,
        # ``cp1252`` etc.) which crashes the drain thread with
        # ``UnicodeDecodeError`` whenever the child logs non-ASCII (for
        # example chat-template tokens, alembic checkmark glyphs, or
        # tokenizer metadata).  Pin UTF-8 with replacement so the drain
        # never raises and ``cassette ...`` payloads round-trip safely.
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, encoding="utf-8",
                                      errors="replace", env=dict(os.environ))

        # Start an early stderr drain so the child cannot deadlock on a
        # full stderr pipe while the parent is still waiting for the
        # URL announcement on stdout.  ``_start_stdout_drain`` later
        # spawns the stdout drainer (and a redundant stderr drainer that
        # is a no-op once this one consumed the stream).
        if self._proc.stderr is not None:
            threading.Thread(target=lambda s=self._proc.stderr: [sys.stderr.write(line) for line in s], daemon=True,
                             name="inferencegate-subproc-stderr-early").start()

        deadline = time.monotonic() + 30.0
        url: str | None = None
        while time.monotonic() < deadline:
            assert self._proc.stdout is not None
            line = self._proc.stdout.readline()
            if not line:
                rc = self._proc.poll()
                raise RuntimeError(f"InferenceGate subprocess exited before announcing URL (rc={rc})")
            line = line.strip()
            if line.startswith("INFERENCEGATE_URL="):
                url = line[len("INFERENCEGATE_URL="):]
                break
        if url is None:
            self.request_stop()
            raise RuntimeError("InferenceGate subprocess did not announce URL within 30s")

        self.base_url = url
        self._start_stdout_drain()
        return url

    def _build_cmd(self) -> list[str]:
        """
        Translate the resolved gate kwargs into a ``inference-gate serve`` command line.

        Endpoints/models are deliberately *not* passed via CLI flags: consuming
        projects push them via ``POST /gate/config`` after the server is up.
        """
        mode_str = "record" if str(self.kwargs.get("mode", "replay")).lower() == "record" else "replay"

        cmd = [sys.executable, "-m", "inference_gate.cli"]
        config_path = self.kwargs.get("config")
        if config_path:
            cmd.extend(["--config", str(config_path)])
        cmd.extend([
            "serve", "--mode", mode_str, "--host",
            str(self.kwargs.get("host", "127.0.0.1")), "--port",
            str(self.kwargs.get("port", 0) or 0), "--print-url", "--parent-pid",
            str(os.getpid())
        ])
        cache_dir = self.kwargs.get("cache_dir")
        if cache_dir:
            cmd.extend(["--cache-dir", str(cache_dir)])
        if self.kwargs.get("fuzzy_model"):
            cmd.append("--fuzzy-model")
        else:
            cmd.append("--no-fuzzy-model")
        fs = self.kwargs.get("fuzzy_sampling")
        if fs:
            cmd.extend(["--fuzzy-sampling", str(fs)])
        max_replies = self.kwargs.get("max_non_greedy_replies")
        if max_replies is not None:
            cmd.extend(["--max-non-greedy-replies", str(max_replies)])
        max_live_requests = self.kwargs.get("max_live_requests")
        if max_live_requests is not None:
            cmd.extend(["--max-live-requests", str(max_live_requests)])
        record_timeout = self.kwargs.get("record_timeout")
        if record_timeout is not None:
            cmd.extend(["--record-timeout", str(record_timeout)])
        return cmd

    def _start_stdout_drain(self) -> None:
        """
        Forward subsequent child stdout lines to parent stderr.  The
        stderr stream is already being drained by the early thread
        spawned in :meth:`start`; a second drainer here would race
        with no benefit.
        """

        def _drain(stream) -> None:
            """Pump ``stream`` line-by-line to the parent's stderr."""
            try:
                for line in stream:
                    sys.stderr.write(line)
            except Exception:  # pylint: disable=broad-except
                pass

        assert self._proc is not None
        threading.Thread(target=_drain, args=(self._proc.stdout,), daemon=True, name="inferencegate-subproc-stdout-drain").start()

    def request_stop(self) -> None:
        """
        Terminate the subprocess and reap it.
        """
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except Exception:  # pylint: disable=broad-except
                    self._proc.kill()
                    self._proc.wait(timeout=5)
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("Error stopping InferenceGate subprocess: %s", exc)
        finally:
            self._proc = None


def _is_xdist_worker(config: pytest.Config) -> bool:
    """
    Return True when running inside a pytest-xdist worker process.
    """
    return hasattr(config, "workerinput")


def _maybe_spawn_subprocess(config: pytest.Config) -> None:
    """
    Controller-side hook: spawn the session-wide subprocess Gate (always).

    No-op on workers (they consume the inherited ``INFERENCEGATE_URL``) or
    when the env var is already set by an outer harness.
    """
    if _is_xdist_worker(config):
        return
    if os.environ.get("INFERENCEGATE_URL"):
        log.info("Reusing externally-provided INFERENCEGATE_URL=%s", os.environ["INFERENCEGATE_URL"])
        return
    kwargs = _resolve_gate_kwargs(config)
    server = _SubprocessServer(kwargs)
    url = server.start()
    os.environ["INFERENCEGATE_URL"] = url
    config._inferencegate_subprocess = server  # type: ignore[attr-defined]
    log.info("InferenceGate subprocess listening at %s", url)


def _maybe_stop_subprocess(config: pytest.Config) -> None:
    """
    Controller-side counterpart to :func:`_maybe_spawn_subprocess`.

    Tears down the subprocess Gate at session end.  Safe to call in any
    process; workers and externally-managed sessions hold no handle.
    """
    server: _SubprocessServer | None = getattr(config, "_inferencegate_subprocess", None)
    if server is not None:
        server.request_stop()
        config._inferencegate_subprocess = None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _resolve_gate_kwargs(pytest_config: pytest.Config) -> dict[str, Any]:
    """
    Resolve all InferenceGate session-level configuration from CLI flags,
    environment variables, and ini options.

    The returned dict carries only session-level fields (mode, cache_dir,
    fuzzy settings, record_timeout, max_replies).  Endpoint and routing
    config is *not* loaded here \u2014 consuming projects push it via
    ``POST /gate/config`` after the server is up.
    """
    mode_str = _resolve_option(pytest_config, "mode")
    cache_dir = _resolve_option(pytest_config, "cache_dir")
    config_path = _resolve_option(pytest_config, "config") or None
    port_str = _resolve_option(pytest_config, "port") or "0"
    port = int(port_str)

    fuzzy_raw = _resolve_option(pytest_config, "fuzzy_model")
    if isinstance(fuzzy_raw, bool):
        fuzzy_model = fuzzy_raw
    elif fuzzy_raw is None or fuzzy_raw == "":
        fuzzy_model = False
    else:
        fuzzy_model = str(fuzzy_raw).lower() in ("true", "1", "yes")

    fuzzy_sampling = str(_resolve_option(pytest_config, "fuzzy_sampling") or "off")
    max_replies_raw = _resolve_option(pytest_config, "max_non_greedy_replies")
    max_non_greedy_replies = int(max_replies_raw) if max_replies_raw not in (None, "") else 5
    max_live_raw = _resolve_option(pytest_config, "max_live_requests")
    max_live_requests = int(max_live_raw) if max_live_raw not in (None, "") else None
    record_timeout_raw = _resolve_option(pytest_config, "record_timeout")
    record_timeout = float(record_timeout_raw) if record_timeout_raw not in (None, "") else 600.0

    return {
        "host": "127.0.0.1",
        "port": port,
        "mode": mode_str,
        "cache_dir": cache_dir,
        "config": config_path,
        "fuzzy_model": fuzzy_model,
        "fuzzy_sampling": fuzzy_sampling,
        "max_non_greedy_replies": max_non_greedy_replies,
        "max_live_requests": max_live_requests,
        "record_timeout": record_timeout,
    }


@pytest.fixture(scope="session")
def inference_gate_url() -> str:
    """
    Session-scoped fixture returning the base URL of the InferenceGate proxy.

    Returns a string like ``"http://127.0.0.1:54321"``.  Point your AI
    client's ``base_url`` at ``f"{inference_gate_url}/v1"`` to route through
    the proxy.

    The Gate is spawned by :func:`pytest_configure`; this fixture just reads
    ``INFERENCEGATE_URL`` from the environment.
    """
    url = os.environ.get("INFERENCEGATE_URL")
    if not url:
        raise RuntimeError("INFERENCEGATE_URL is not set; the InferenceGate subprocess was not spawned. "
                           "Check pytest_configure for spawn errors.")
    return url


def _build_header_overrides(node: pytest.Item, worker_id: str) -> dict[str, str]:
    """
    Build per-test InferenceGate headers from pytest metadata and markers.
    """
    custom_marker = node.get_closest_marker("inferencegate")

    header_overrides: dict[str, str] = {}
    if custom_marker is not None:
        if "fuzzy_model" in custom_marker.kwargs:
            value = custom_marker.kwargs["fuzzy_model"]
            header_overrides["X-InferenceGate-Require-Fuzzy-Model"] = "on" if value else "off"
        if "fuzzy_sampling" in custom_marker.kwargs:
            header_overrides["X-InferenceGate-Require-Fuzzy-Sampling"] = str(custom_marker.kwargs["fuzzy_sampling"])

    header_overrides["X-InferenceGate-Metadata-Test-NodeID"] = node.nodeid
    header_overrides["X-InferenceGate-Metadata-Worker-ID"] = worker_id
    return header_overrides


@pytest.fixture(autouse=True)
def _inferencegate_match_policy(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """
    Apply per-test cassette matching overrides via Gate-owned header context.

    ``@pytest.mark.inferencegate(fuzzy_model=..., fuzzy_sampling=...)`` is
    translated into ``X-InferenceGate-Require-*`` headers, and every test also
    receives ``Metadata-Test-NodeID`` / ``Metadata-Worker-ID`` provenance.
    Downstream clients that know about :mod:`inference_gate.pytest_context`
    can attach these headers to their outbound HTTP requests.
    """
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
    header_overrides = _build_header_overrides(request.node, worker_id)
    context_token = pytest_context.update_headers(header_overrides)

    try:
        yield
    finally:
        pytest_context.reset_headers(context_token)
