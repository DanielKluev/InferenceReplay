"""
CLI for InferenceGate.

Provides command-line interface for starting the proxy server
in different modes and managing the inference cache.

Entry point: `main()` (registered as `inference-gate` console script).

Configuration is loaded from YAML files at `$USERDIR/.InferenceGate/config.yaml`
by default, or from a user-specified path via the `--config` option.
Command-line options override configuration file values.
"""

import asyncio
import json
import logging
from typing import Any

import aiohttp
import click

from inference_gate.cli_format import (format_index_rows_json, format_index_rows_table, format_reply_human, format_reply_json,
                                       format_stats_human, format_stats_json, format_tape_detail_human, format_tape_detail_json)
from inference_gate.config import Config, ConfigManager, EndpointDef, ModelRouteDef
from inference_gate.inference_gate import InferenceGate
from inference_gate.modes import Mode
from inference_gate.outflow.client import OutflowClient
from inference_gate.outflow.model_router import EndpointConfig, ModelRoute
from inference_gate.recording.hashing import compute_response_hash
from inference_gate.recording.storage import CachedRequest, CachedResponse, CacheEntry, CacheStorage

# Global config manager and config, set up via pass_context
pass_config = click.make_pass_decorator(Config, ensure=True)


def setup_logging(verbose: bool) -> None:
    """
    Set up logging configuration.

    `verbose` enables DEBUG level logging, otherwise INFO level is used.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def load_config(ctx: click.Context, config_path: str | None) -> Config:
    """
    Load configuration and store it in context.

    Loads from `config_path` if specified, otherwise from default location.
    """
    manager = ConfigManager(config_path)
    config = manager.load()
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["config_manager"] = manager
    return config


def _build_endpoints_and_models(config: Config) -> tuple[dict[str, EndpointConfig], list[ModelRoute]]:
    """
    Convert YAML ``endpoints`` / ``models`` definitions into the runtime types
    consumed by :class:`InferenceGate`.

    Per-endpoint ``timeout`` falls back to the top-level ``record_timeout``
    when not set explicitly.  ``models`` preserves declaration order so glob
    tie-breaking works as documented.
    """
    endpoints: dict[str, EndpointConfig] = {}
    for name, ep in config.endpoints.items():
        endpoints[name] = EndpointConfig(
            url=ep.url,
            api_key=ep.api_key,
            timeout=float(ep.timeout) if ep.timeout is not None else float(config.record_timeout),
            proxy=ep.proxy,
        )
    routes: list[ModelRoute] = [ModelRoute(pattern=r.pattern, endpoint_name=r.endpoint, order=idx) for idx, r in enumerate(config.models)]
    return endpoints, routes


def get_config(ctx: click.Context) -> Config:
    """
    Get config from context, or load default if not present.
    """
    if ctx.obj is None:
        ctx.obj = {}
    if "config" not in ctx.obj:
        manager = ConfigManager()
        ctx.obj["config"] = manager.load()
        ctx.obj["config_manager"] = manager
    return ctx.obj["config"]


@click.group()
@click.version_option(version="0.1.0", prog_name="inference-gate")
@click.option("--config", "-C", "config_path", type=click.Path(), default=None,
              help="Path to configuration file (default: $USERDIR/.InferenceGate/config.yaml)")
@click.pass_context
def main(ctx: click.Context, config_path: str | None) -> None:
    """
    InferenceGate - AI inference replay for testing, debugging and development.

    This tool provides an OpenAI-compatible API proxy that can record and replay
    AI inference calls, helping you save costs and time during development and testing.

    Configuration is loaded from $USERDIR/.InferenceGate/config.yaml by default.
    Use --config to specify a custom configuration file path.
    Command-line options override configuration file values.
    """
    load_config(ctx, config_path)


@main.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the server on (default: 8080)")
@click.option("--host", "-h", default=None, help="Host to bind the server to (default: 127.0.0.1)")
@click.option("--cache-dir", "-c", default=None, help="Directory to store cached responses (default: .inference_cache)")
@click.option("--web-ui", is_flag=True, default=False, help="Enable the web UI dashboard")
@click.option("--web-ui-port", default=8081, type=int, help="Port for the web UI server (default: 8081)")
@click.option("--fuzzy-model/--no-fuzzy-model", default=None,
              help="Enable or disable fuzzy model matching: on cache miss, reuse entries with the same prompt but a different model")
@click.option("--fuzzy-sampling", default=None, type=click.Choice(["off", "soft", "aggressive"]),
              help="Sampling parameter fuzzy matching: off (exact), soft (non-greedy matches non-greedy), aggressive (any match)")
@click.option("--max-non-greedy-replies", default=None, type=int,
              help="Max replies to collect per non-greedy cassette before cycling (default: 5)")
@click.option("--max-live-requests", default=None, type=int,
              help="Global limit on the number of non-cassette upstream requests to perform per session")
@click.option("--record-timeout", default=None, type=float,
              help="Default upstream HTTP timeout in seconds (used when an endpoint does not set its own timeout). Default: 600.0")
@click.option("--verbose", "-v", is_flag=True, default=None, help="Enable verbose logging")
@click.pass_context
def start(ctx: click.Context, port: int | None, host: str | None, cache_dir: str | None, web_ui: bool, web_ui_port: int,
          fuzzy_model: bool | None, fuzzy_sampling: str | None, max_non_greedy_replies: int | None, max_live_requests: int | None,
          record_timeout: float | None, verbose: bool | None) -> None:
    """
    Start in record-and-replay mode (default).

    Replays cached inferences when available. On cache miss, dispatches the
    request to the endpoint matched by the request's ``model`` field via the
    configured routing table, records the response, and stores it for future
    replays.

    Endpoints and the model→endpoint routing table are loaded from the YAML
    config.  Use ``POST /gate/config`` at runtime to push live config from
    test harnesses without restarting the server.
    """
    config = get_config(ctx)

    # Apply config defaults, command-line overrides
    actual_port = port if port is not None else config.port
    actual_host = host if host is not None else config.host
    actual_cache_dir = cache_dir if cache_dir is not None else config.cache_dir
    actual_verbose = verbose if verbose is not None else config.verbose
    actual_fuzzy_model = fuzzy_model if fuzzy_model is not None else config.fuzzy_model
    actual_fuzzy_sampling = fuzzy_sampling if fuzzy_sampling is not None else config.fuzzy_sampling
    actual_max_replies = max_non_greedy_replies if max_non_greedy_replies is not None else config.max_non_greedy_replies
    actual_max_live_requests = max_live_requests if max_live_requests is not None else config.max_live_requests
    actual_record_timeout = record_timeout if record_timeout is not None else config.record_timeout

    setup_logging(actual_verbose)

    endpoints, models = _build_endpoints_and_models(config)
    gate = InferenceGate(host=actual_host, port=actual_port, mode=Mode.RECORD_AND_REPLAY, cache_dir=actual_cache_dir,
                         web_ui=web_ui, web_ui_port=web_ui_port, fuzzy_model=actual_fuzzy_model, fuzzy_sampling=actual_fuzzy_sampling,
                         max_non_greedy_replies=actual_max_replies, max_live_requests=actual_max_live_requests,
                         record_timeout=actual_record_timeout, endpoints=endpoints, models=models)

    click.echo("Starting InferenceGate in record-and-replay mode")
    click.echo(f"  Proxy: http://{actual_host}:{actual_port}")
    click.echo(f"  Endpoints: {sorted(endpoints) if endpoints else '(none — push via /gate/config)'}")
    click.echo(f"  Routes: {len(models)} entr{'y' if len(models) == 1 else 'ies'}")
    click.echo(f"  Cache dir: {actual_cache_dir}")
    click.echo(f"  Record timeout: {actual_record_timeout}s")
    if actual_fuzzy_model:
        click.echo("  Fuzzy model matching: enabled")
    if actual_fuzzy_sampling != "off":
        click.echo(f"  Fuzzy sampling: {actual_fuzzy_sampling}")
    click.echo(f"  Max non-greedy replies: {actual_max_replies}")
    if actual_max_live_requests is not None:
        click.echo(f"  Max live requests: {actual_max_live_requests}")
    if web_ui:
        click.echo(f"  WebUI: http://127.0.0.1:{web_ui_port}")

    asyncio.run(gate.run_forever())


@main.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the server on (default: 8080)")
@click.option("--host", "-h", default=None, help="Host to bind the server to (default: 127.0.0.1)")
@click.option("--cache-dir", "-c", default=None, help="Directory to store cached responses (default: .inference_cache)")
@click.option("--web-ui", is_flag=True, default=False, help="Enable the web UI dashboard")
@click.option("--web-ui-port", default=8081, type=int, help="Port for the web UI server (default: 8081)")
@click.option("--fuzzy-model/--no-fuzzy-model", default=None,
              help="Enable or disable fuzzy model matching: on cache miss, reuse entries with the same prompt but a different model")
@click.option("--fuzzy-sampling", default=None, type=click.Choice(["off", "soft", "aggressive"]),
              help="Sampling parameter fuzzy matching: off (exact), soft (non-greedy matches non-greedy), aggressive (any match)")
@click.option("--max-non-greedy-replies", default=None, type=int,
              help="Max replies to collect per non-greedy cassette before cycling (default: 5)")
@click.option("--verbose", "-v", is_flag=True, default=None, help="Enable verbose logging")
@click.pass_context
def replay(ctx: click.Context, port: int | None, host: str | None, cache_dir: str | None, web_ui: bool, web_ui_port: int,
           fuzzy_model: bool | None, fuzzy_sampling: str | None, max_non_greedy_replies: int | None, verbose: bool | None) -> None:
    """
    Start in replay-only mode.

    Only returns cached responses. Returns an error if a matching inference
    is not found in the cache. Useful for unit tests and CI pipelines.

    Command-line options override configuration file values.
    """
    config = get_config(ctx)

    # Apply config defaults, command-line overrides
    actual_port = port if port is not None else config.port
    actual_host = host if host is not None else config.host
    actual_cache_dir = cache_dir if cache_dir is not None else config.cache_dir
    actual_verbose = verbose if verbose is not None else config.verbose
    actual_fuzzy_model = fuzzy_model if fuzzy_model is not None else config.fuzzy_model
    actual_fuzzy_sampling = fuzzy_sampling if fuzzy_sampling is not None else config.fuzzy_sampling
    actual_max_replies = max_non_greedy_replies if max_non_greedy_replies is not None else config.max_non_greedy_replies

    setup_logging(actual_verbose)

    gate = InferenceGate(host=actual_host, port=actual_port, mode=Mode.REPLAY_ONLY, cache_dir=actual_cache_dir, web_ui=web_ui,
                         web_ui_port=web_ui_port, fuzzy_model=actual_fuzzy_model, fuzzy_sampling=actual_fuzzy_sampling,
                         max_non_greedy_replies=actual_max_replies)

    click.echo("Starting InferenceGate in replay-only mode")
    click.echo(f"  Proxy: http://{actual_host}:{actual_port}")
    click.echo(f"  Cache dir: {actual_cache_dir}")
    if actual_fuzzy_model:
        click.echo("  Fuzzy model matching: enabled")
    if actual_fuzzy_sampling != "off":
        click.echo(f"  Fuzzy sampling: {actual_fuzzy_sampling}")
    click.echo(f"  Max non-greedy replies: {actual_max_replies}")
    if web_ui:
        click.echo(f"  WebUI: http://127.0.0.1:{web_ui_port}")

    asyncio.run(gate.run_forever())


@main.command()
@click.option("--mode", default="replay", type=click.Choice(["replay", "record"]),
              help="Operating mode: 'replay' (cached only, default) or 'record' (cache + upstream).")
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind the server to (default: 127.0.0.1).")
@click.option("--port", "-p", default=0, type=int, help="Port to bind to. 0 = OS-assigned ephemeral (default: 0).")
@click.option("--cache-dir", "-c", default=None, help="Directory for cached cassettes (default: from config).")
@click.option("--fuzzy-model/--no-fuzzy-model", default=None, help="Enable/disable fuzzy model matching.")
@click.option("--fuzzy-sampling", default=None, type=click.Choice(["off", "soft", "aggressive"]),
              help="Sampling parameter fuzzy matching level.")
@click.option("--max-non-greedy-replies", default=None, type=int, help="Max replies per non-greedy cassette before cycling.")
@click.option("--max-live-requests", default=None, type=int,
              help="Global limit on the number of non-cassette upstream requests to perform per session")
@click.option("--record-timeout", default=None, type=float,
              help="Default upstream HTTP timeout in seconds.  Falls back to config.record_timeout (default: 600.0).  "
                   "Test harnesses should pass the same value as the pytest timeout so a slow recording does not silently abort.")
@click.option("--print-url", is_flag=True, default=False,
              help="Print 'INFERENCEGATE_URL=http://...' on a single line after the server is ready, then continue running.  "
                   "Designed for parent processes that need to discover the OS-assigned port.")
@click.option("--parent-pid", default=None, type=int,
              help="If set, periodically check this PID; exit cleanly when the parent process disappears.  "
                   "Used by pytest-xdist controllers to ensure the subprocess Gate dies with the test session.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging.")
@click.pass_context
def serve(ctx: click.Context, mode: str, host: str, port: int, cache_dir: str | None, fuzzy_model: bool | None,
          fuzzy_sampling: str | None, max_non_greedy_replies: int | None, max_live_requests: int | None, record_timeout: float | None,
          print_url: bool, parent_pid: int | None, verbose: bool) -> None:
    """
    Start a long-running InferenceGate server suitable for use as a subprocess.

    Designed for parent-process orchestration (notably the pytest plugin's
    xdist-aware mode).  Differs from ``start`` / ``replay`` in three ways:

    1. ``--print-url`` emits a single ``INFERENCEGATE_URL=...`` line after the
       server is listening so the parent can read the OS-assigned port.
    2. ``--parent-pid`` enables a watchdog that exits the server when the
       parent process is gone, avoiding orphaned servers in CI.
    3. ``--mode`` is a single flag so the same command line covers replay
       and record sessions.

    Endpoints and the model→endpoint routing table are loaded from the YAML
    config; tests typically push them via ``POST /gate/config`` after the
    server is up.
    """
    setup_logging(verbose)

    config = get_config(ctx)
    actual_cache_dir = cache_dir if cache_dir is not None else config.cache_dir
    actual_fuzzy_model = fuzzy_model if fuzzy_model is not None else config.fuzzy_model
    actual_fuzzy_sampling = fuzzy_sampling if fuzzy_sampling is not None else config.fuzzy_sampling
    actual_max_replies = max_non_greedy_replies if max_non_greedy_replies is not None else config.max_non_greedy_replies
    actual_max_live_requests = max_live_requests if max_live_requests is not None else config.max_live_requests
    actual_record_timeout = record_timeout if record_timeout is not None else config.record_timeout

    gate_mode = Mode.RECORD_AND_REPLAY if mode == "record" else Mode.REPLAY_ONLY
    endpoints, models = _build_endpoints_and_models(config) if gate_mode == Mode.RECORD_AND_REPLAY else ({}, [])

    gate = InferenceGate(host=host, port=port, mode=gate_mode, cache_dir=actual_cache_dir, fuzzy_model=actual_fuzzy_model,
                         fuzzy_sampling=actual_fuzzy_sampling, max_non_greedy_replies=actual_max_replies,
                         max_live_requests=actual_max_live_requests, record_timeout=actual_record_timeout,
                         endpoints=endpoints, models=models)

    asyncio.run(_run_serve(gate, print_url=print_url, parent_pid=parent_pid))


async def _run_serve(gate: InferenceGate, print_url: bool, parent_pid: int | None) -> None:
    """
    Async driver for ``serve``: starts the gate, prints the URL, polls parent.

    Splits ``InferenceGate.run_forever`` into start + custom wait loop so we
    can announce the OS-assigned port and watch the parent PID.
    """
    import os
    import sys

    await gate.start()
    if print_url:
        # Single-line, machine-parseable announcement on stdout.
        sys.stdout.write(f"INFERENCEGATE_URL={gate.base_url}\n")
        sys.stdout.flush()

    try:
        while True:
            if parent_pid is not None and not _process_alive(parent_pid):
                # Parent died; exit cleanly so the subprocess does not orphan.
                break
            await asyncio.sleep(1.0 if parent_pid is not None else 3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await gate.stop()
    # Suppress the ``os`` import warning when not used in record mode.
    _ = os


def _process_alive(pid: int) -> bool:
    """
    Return True if the process with ``pid`` is alive.

    Cross-platform: uses ``os.kill(pid, 0)`` on POSIX and ``OpenProcess`` on
    Windows.  Safe to call frequently — does not raise on missing process.
    """
    import os
    if os.name == "nt":
        # On Windows os.kill(pid, 0) raises OSError for any condition; use ctypes for a definitive check.
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        if not ok:
            return False
        # 259 == STILL_ACTIVE
        return exit_code.value == 259
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # PID exists but we lack signal rights — process is still alive.
        return True
    except OSError:
        return False


@main.group()
@click.pass_context
def cassette(ctx: click.Context) -> None:
    """
    Cassette management commands.

    Inspect, search, read, and manage recorded inference cassettes.
    All read commands support --json for machine-parseable output
    (suitable for AI agentic consumption).
    """


def _resolve_cassette_id(storage: CacheStorage, cassette_id: str) -> str | None:
    """
    Resolve a cassette ID (full or prefix) to an exact content_hash.

    Returns the content_hash if exactly one match is found.
    Prints an error and returns None if zero or multiple matches found.
    """
    matches = storage.resolve_prefix(cassette_id)
    if len(matches) == 0:
        click.echo(f"Error: no cassette found matching '{cassette_id}'", err=True)
        return None
    if len(matches) > 1:
        click.echo(f"Error: ambiguous cassette ID '{cassette_id}', matches {len(matches)} cassettes:", err=True)
        for row in matches[:10]:
            click.echo(f"  {row.content_hash}  {row.model}  {row.slug}", err=True)
        if len(matches) > 10:
            click.echo(f"  ... and {len(matches) - 10} more", err=True)
        return None
    return matches[0].content_hash


@cassette.command(name="list")
@click.option("--cache-dir", "-c", default=None, help="Directory where cached responses are stored")
@click.option("--model", "-m", default=None, help="Filter by model name (substring match)")
@click.option("--greedy/--non-greedy", default=None, help="Filter by greedy / non-greedy sampling")
@click.option("--has-tools", is_flag=True, default=None, help="Filter cassettes that use tools")
@click.option("--has-logprobs", is_flag=True, default=None, help="Filter cassettes that have logprobs")
@click.option("--after", default=None, help="Only cassettes recorded after this ISO 8601 date")
@click.option("--before", default=None, help="Only cassettes recorded before this ISO 8601 date")
@click.option("--sort", "sort_by", default="recorded", type=click.Choice(["recorded", "model", "tokens_in", "tokens_out"]),
              help="Sort by field (default: recorded)")
@click.option("--limit", "-n", default=None, type=int, help="Maximum number of results")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON array")
@click.pass_context
def cassette_list(ctx: click.Context, cache_dir: str | None, model: str | None, greedy: bool | None, has_tools: bool | None,
                  has_logprobs: bool | None, after: str | None, before: str | None, sort_by: str, limit: int | None, as_json: bool) -> None:
    """List all cached cassettes with filtering and sorting."""
    config = get_config(ctx)
    actual_cache_dir = cache_dir if cache_dir is not None else config.cache_dir

    storage = CacheStorage(actual_cache_dir)
    rows = storage.filter_entries(model=model, greedy=greedy, has_tools=has_tools, has_logprobs=has_logprobs, after=after, before=before,
                                  sort_by=sort_by, limit=limit)

    if as_json:
        click.echo(format_index_rows_json(rows))
    else:
        click.echo(format_index_rows_table(rows))


@cassette.command(name="search")
@click.argument("query")
@click.option("--cache-dir", "-c", default=None, help="Directory where cached responses are stored")
@click.option("--model", "-m", default=None, help="Additional filter by model name (substring)")
@click.option("--limit", "-n", default=20, type=int, help="Maximum number of results (default: 20)")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON array")
@click.pass_context
def cassette_search(ctx: click.Context, query: str, cache_dir: str | None, model: str | None, limit: int, as_json: bool) -> None:
    """Search cassettes by prompt content.

    Performs case-insensitive substring matching on the first user message
    and slug fields. QUERY is the text to search for.
    """
    config = get_config(ctx)
    actual_cache_dir = cache_dir if cache_dir is not None else config.cache_dir

    storage = CacheStorage(actual_cache_dir)
    rows = storage.search_entries(query, model=model, limit=limit)

    if as_json:
        click.echo(format_index_rows_json(rows))
    else:
        click.echo(format_index_rows_table(rows))


@cassette.command(name="show")
@click.argument("cassette_id")
@click.option("--cache-dir", "-c", default=None, help="Directory where cached responses are stored")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON object")
@click.pass_context
def cassette_show(ctx: click.Context, cassette_id: str, cache_dir: str | None, as_json: bool) -> None:
    """Show cassette metadata, prompt, and reply summaries.

    CASSETTE_ID is the content hash (or unique prefix) of the cassette to show.
    Does not print full completion text — use 'read' for that.
    """
    config = get_config(ctx)
    actual_cache_dir = cache_dir if cache_dir is not None else config.cache_dir

    storage = CacheStorage(actual_cache_dir)
    resolved = _resolve_cassette_id(storage, cassette_id)
    if resolved is None:
        ctx.exit(1)
        return

    tape_data = storage.load_tape(resolved)
    if tape_data is None:
        click.echo(f"Error: tape file not found for cassette {resolved}", err=True)
        ctx.exit(1)
        return

    metadata, sections = tape_data
    index_row = storage.index.by_content_hash.get(resolved)

    if as_json:
        click.echo(format_tape_detail_json(resolved, metadata, sections, index_row))
    else:
        click.echo(format_tape_detail_human(resolved, metadata, sections, index_row))


@cassette.command(name="read")
@click.argument("cassette_id")
@click.option("--cache-dir", "-c", default=None, help="Directory where cached responses are stored")
@click.option("--reply", "-r", "reply_number", default=None, type=int, help="Show only a specific reply number")
@click.option("--prompt", "-p", "include_prompt", is_flag=True, default=False, help="Also include the prompt messages")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON")
@click.pass_context
def cassette_read(ctx: click.Context, cassette_id: str, cache_dir: str | None, reply_number: int | None, include_prompt: bool,
                  as_json: bool) -> None:
    """Read full completion text of a cassette's replies.

    CASSETTE_ID is the content hash (or unique prefix) of the cassette.
    By default shows all replies. Use --reply N to select a specific one.
    """
    config = get_config(ctx)
    actual_cache_dir = cache_dir if cache_dir is not None else config.cache_dir

    storage = CacheStorage(actual_cache_dir)
    resolved = _resolve_cassette_id(storage, cassette_id)
    if resolved is None:
        ctx.exit(1)
        return

    tape_data = storage.load_tape(resolved)
    if tape_data is None:
        click.echo(f"Error: tape file not found for cassette {resolved}", err=True)
        ctx.exit(1)
        return

    _, sections = tape_data

    if as_json:
        click.echo(format_reply_json(sections, reply_number=reply_number, include_prompt=include_prompt))
    else:
        click.echo(format_reply_human(sections, reply_number=reply_number, include_prompt=include_prompt))


@cassette.command(name="delete")
@click.argument("cassette_id", required=False)
@click.option("--model", "-m", default=None, help="Delete all cassettes matching this model name (substring match)")
@click.option("--non-200", is_flag=True, help="Delete all cassettes with a non-200 HTTP status code")
@click.option("--cache-dir", "-c", default=None, help="Directory where cached responses are stored")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def cassette_delete(ctx: click.Context, cassette_id: str | None, model: str | None, non_200: bool, cache_dir: str | None, yes: bool) -> None:
    """Delete a single cassette by ID, all cassettes for a model, or all non-200 cassettes.

    CASSETTE_ID is the content hash (or unique prefix) of the cassette to delete.
    Alternatively, use --model to delete all cassettes matching a model name, or
    use --non-200 to delete all cassettes that recorded an error response.
    Removes tape files, associated response files, and updates the index.
    """
    config = get_config(ctx)
    actual_cache_dir = cache_dir if cache_dir is not None else config.cache_dir

    storage = CacheStorage(actual_cache_dir)
    
    if not cassette_id and not model and not non_200:
        click.echo("Error: Must provide either CASSETTE_ID, --model, or --non-200.", err=True)
        ctx.exit(1)
        return
    if sum(bool(x) for x in [cassette_id, model, non_200]) > 1:
        click.echo("Error: Cannot combine CASSETTE_ID, --model, and --non-200.", err=True)
        ctx.exit(1)
        return

    if model or non_200:
        if model:
            rows = storage.filter_entries(model=model)
            if not rows:
                click.echo(f"No cassettes found matching model '{model}'.")
                return
            click.echo(f"Found {len(rows)} cassettes matching model '{model}':")
        else:
            rows = [r for r in storage.index._rows if r.status_code != 200]
            if not rows:
                click.echo("No cassettes found with non-200 status codes.")
                return
            click.echo(f"Found {len(rows)} cassettes with non-200 status codes:")
            
        models_found = {}
        for row in rows:
            models_found[row.model] = models_found.get(row.model, 0) + 1
        for m, count in models_found.items():
            click.echo(f"  - {m}: {count} cassette(s)")
            
        if not yes:
            if not click.confirm(f"Delete these {len(rows)} cassettes?"):
                click.echo("Aborted.")
                return
                
        deleted_count = 0
        for row in rows:
            if storage.delete_entry(row.content_hash):
                deleted_count += 1
                
        click.echo(f"Deleted {deleted_count} cassettes.")
        return

    # Delete by cassette_id
    resolved = _resolve_cassette_id(storage, cassette_id)
    if resolved is None:
        ctx.exit(1)
        return

    # Show what will be deleted
    index_row = storage.index.by_content_hash.get(resolved)
    if index_row:
        click.echo(f"Cassette: {resolved}")
        click.echo(f"  Model: {index_row.model}")
        click.echo(f"  Slug: {index_row.slug}")
        click.echo(f"  Replies: {index_row.replies}")

    if not yes:
        if not click.confirm("Delete this cassette?"):
            click.echo("Aborted.")
            return

    if storage.delete_entry(resolved):
        click.echo(f"Deleted cassette {resolved}")
    else:
        click.echo(f"Error: failed to delete cassette {resolved}", err=True)
        ctx.exit(1)


@cassette.command(name="stats")
@click.option("--cache-dir", "-c", default=None, help="Directory where cached responses are stored")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON")
@click.pass_context
def cassette_stats(ctx: click.Context, cache_dir: str | None, as_json: bool) -> None:
    """Show cache statistics."""
    config = get_config(ctx)
    actual_cache_dir = cache_dir if cache_dir is not None else config.cache_dir

    storage = CacheStorage(actual_cache_dir)
    rows = list(storage.index.by_content_hash.values())

    total_entries = len(rows)
    greedy_count = 0
    non_greedy_count = 0
    total_replies = 0
    total_tokens_in = 0
    total_tokens_out = 0
    entries_by_model: dict[str, int] = {}

    for row in rows:
        if row.is_greedy:
            greedy_count += 1
        else:
            non_greedy_count += 1
        total_replies += row.replies
        if row.tokens_in and row.tokens_in.isdigit():
            total_tokens_in += int(row.tokens_in)
        if row.tokens_out and row.tokens_out.isdigit():
            total_tokens_out += int(row.tokens_out)
        if row.model:
            entries_by_model[row.model] = entries_by_model.get(row.model, 0) + 1

    disk_size = storage.get_disk_size()

    if as_json:
        click.echo(
            format_stats_json(total_entries, total_replies, disk_size, greedy_count, non_greedy_count, entries_by_model, total_tokens_in,
                              total_tokens_out))
    else:
        click.echo(
            format_stats_human(total_entries, total_replies, disk_size, greedy_count, non_greedy_count, entries_by_model, total_tokens_in,
                               total_tokens_out, actual_cache_dir))


@cassette.command(name="reindex")
@click.option("--cache-dir", "-c", default=None, help="Directory where cached responses are stored")
@click.pass_context
def cassette_reindex(ctx: click.Context, cache_dir: str | None) -> None:
    """Rebuild the TSV index from tape files on disk."""
    config = get_config(ctx)
    actual_cache_dir = cache_dir if cache_dir is not None else config.cache_dir

    storage = CacheStorage(actual_cache_dir)
    count = storage.reindex()
    click.echo(f"Reindexed {count} tape files in {actual_cache_dir}")


@cassette.command(name="fill")
@click.argument("cassette_id")
@click.option("--count", "-n", default=None, type=int,
              help="Target number of unique completions (default: max_non_greedy_replies from config)")
@click.option("--cache-dir", "-c", default=None, help="Directory where cached responses are stored")
@click.option("--upstream", "-u", default=None, help="Upstream OpenAI API base URL")
@click.option("--api-key", "-k", envvar="OPENAI_API_KEY", default=None, help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
@click.option("--upstream-timeout", default=None, type=float, help="Timeout in seconds for upstream API requests (default: 120.0)")
@click.option("--proxy", default=None, help="HTTP proxy URL for upstream requests")
@click.option("--max-attempts", default=None, type=int,
              help="Maximum number of upstream requests before giving up (default: 3x target count)")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON")
@click.option("--verbose", "-v", is_flag=True, default=None, help="Enable verbose logging")
@click.pass_context
def cassette_fill(ctx: click.Context, cassette_id: str, count: int | None, cache_dir: str | None, upstream: str | None, api_key: str | None,
                  upstream_timeout: float | None, proxy: str | None, max_attempts: int | None, as_json: bool, verbose: bool | None) -> None:
    """Fill a non-greedy cassette with additional unique completions.

    Re-issues the original request to the upstream endpoint to collect more
    unique completions for a cassette until it has COUNT total replies.
    Responses are de-duplicated by content hash, so identical completions
    are not stored twice.

    CASSETTE_ID is the content hash (or unique prefix) of the cassette to fill.

    Only works with non-greedy cassettes (temperature > 0). Greedy cassettes
    always produce the same output, so filling them makes no sense.
    """
    config = get_config(ctx)

    actual_cache_dir = cache_dir if cache_dir is not None else config.cache_dir
    # ``cassette fill`` calls the upstream directly (not through Gate); the user must supply
    # connection details explicitly via flags or env vars.  No fallback to a single global
    # ``config.upstream`` exists in the multi-endpoint schema.
    actual_upstream = upstream
    actual_api_key = api_key
    actual_timeout = upstream_timeout if upstream_timeout is not None else config.record_timeout
    actual_proxy = proxy
    actual_verbose = verbose if verbose is not None else config.verbose
    target_count = count if count is not None else config.max_non_greedy_replies

    setup_logging(actual_verbose)

    if not actual_api_key:
        click.echo("Error: No API key provided. Set OPENAI_API_KEY environment variable, use --api-key, or configure in config file.",
                   err=True)
        ctx.exit(1)
        return

    storage = CacheStorage(actual_cache_dir)
    resolved = _resolve_cassette_id(storage, cassette_id)
    if resolved is None:
        ctx.exit(1)
        return

    # Load tape metadata to validate
    tape_data = storage.load_tape(resolved)
    if tape_data is None:
        click.echo(f"Error: tape file not found for cassette {resolved}", err=True)
        ctx.exit(1)
        return

    metadata, _ = tape_data
    index_row = storage.index.by_content_hash.get(resolved)

    # Validate: must be non-greedy
    if metadata.sampling.is_greedy:
        click.echo("Error: cassette uses greedy sampling (temperature=0). Filling only works with non-greedy cassettes.", err=True)
        ctx.exit(1)
        return

    existing_replies = index_row.replies if index_row else metadata.replies
    if existing_replies >= target_count:
        if as_json:
            click.echo(json.dumps({
                "content_hash": resolved,
                "added": 0,
                "duplicates": 0,
                "errors": 0,
                "attempts": 0,
                "total_replies": existing_replies,
                "target": target_count
            }, indent=2))
        else:
            click.echo(f"Cassette already has {existing_replies} replies (target: {target_count}). Nothing to do.")
        return

    # Reconstruct the request body from the tape
    request_body = storage.reconstruct_request_body(resolved)
    if request_body is None:
        click.echo(f"Error: could not reconstruct request body from cassette {resolved}", err=True)
        ctx.exit(1)
        return

    needed = target_count - existing_replies
    actual_max_attempts = max_attempts if max_attempts is not None else needed * 3

    if not as_json:
        click.echo(f"Cassette: {resolved}")
        click.echo(f"  Model: {metadata.model}")
        click.echo(f"  Existing replies: {existing_replies}")
        click.echo(f"  Target: {target_count}")
        click.echo(f"  Need: {needed} more unique completions")
        click.echo(f"  Max attempts: {actual_max_attempts}")
        click.echo(f"  Upstream: {actual_upstream}")
        click.echo()

    result = asyncio.run(
        _fill_cassette(storage, resolved, request_body, metadata.endpoint, actual_upstream, actual_api_key, actual_timeout, actual_proxy,
                       target_count, actual_max_attempts))

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"Done: {result['added']} new unique completions added ({result['duplicates']} duplicates, "
                   f"{result['errors']} errors in {result['attempts']} attempts)")
        click.echo(f"Cassette now has {result['total_replies']} replies")
        if result['errors'] > 0:
            click.echo("Warning: some upstream requests failed. Check --verbose for details.", err=True)


async def _fill_cassette(storage: CacheStorage, content_hash: str, request_body: dict[str, Any], endpoint: str, upstream_base_url: str,
                         api_key: str, timeout: float, proxy: str | None, target_count: int, max_attempts: int) -> dict[str, Any]:
    """
    Fill a cassette with unique completions by re-issuing requests to upstream.

    Sends the reconstructed request to the upstream endpoint up to `max_attempts`
    times, de-duplicating responses by content hash. Stops when the cassette has
    `target_count` total replies or `max_attempts` is exhausted.

    Returns a dict with fill statistics: added, duplicates, errors, attempts, total_replies.
    """
    log = logging.getLogger("FillCassette")

    # Collect existing response hashes for de-duplication
    existing_hashes = set(storage.get_reply_response_hashes(content_hash))
    index_row = storage.index.by_content_hash.get(content_hash)
    current_replies = index_row.replies if index_row else 0

    added = 0
    duplicates = 0
    errors = 0
    attempts = 0

    # Ensure request is non-streaming for fill (we just need the response body)
    fill_body = dict(request_body)
    fill_body.pop("stream", None)
    fill_body.pop("stream_options", None)

    client = OutflowClient(upstream_base_url, api_key=api_key, timeout=timeout, proxy=proxy)
    await client.start()

    try:
        while current_replies < target_count and attempts < max_attempts:
            attempts += 1
            log.info("Fill attempt %d/%d (have %d/%d replies)", attempts, max_attempts, current_replies, target_count)

            request = CachedRequest(method="POST", path=endpoint, headers={"Content-Type": "application/json"}, body=fill_body)

            try:
                response = await client.forward_request(request)
            except Exception as e:
                log.error("Upstream request failed: %s", e)
                errors += 1
                continue

            if response.status_code != 200:
                log.error("Upstream returned HTTP %d", response.status_code)
                errors += 1
                continue

            # Compute response hash for de-duplication
            resp_body = response.body
            if resp_body is None and response.is_streaming and response.chunks:
                from inference_gate.recording.reassembly import reassemble_streaming_response
                resp_body = reassemble_streaming_response(response.chunks, "")

            if resp_body is None:
                log.warning("Empty response body, skipping")
                errors += 1
                continue

            response_hash = compute_response_hash(resp_body)

            if response_hash in existing_hashes:
                log.info("Duplicate response (hash=%s), skipping", response_hash)
                duplicates += 1
                continue

            # Store the new unique completion
            entry = CacheEntry(request=request, response=response, model=fill_body.get("model"), temperature=fill_body.get("temperature"))
            storage._append_reply(content_hash, entry, index_row)

            existing_hashes.add(response_hash)
            current_replies += 1
            added += 1
            log.info("Added reply %d (hash=%s)", current_replies, response_hash)

            # Refresh index row for next iteration
            index_row = storage.index.by_content_hash.get(content_hash)
    finally:
        await client.stop()

    # Update max_replies if target_count is larger
    if target_count > (index_row.max_replies if index_row else 0):
        storage._update_max_replies(content_hash, target_count)

    return {
        "content_hash": content_hash,
        "added": added,
        "duplicates": duplicates,
        "errors": errors,
        "attempts": attempts,
        "total_replies": current_replies,
        "target": target_count,
    }


# =============================================================================
# Test commands
# =============================================================================


async def _send_test_prompt(url: str, headers: dict[str, str], model: str, prompt: str, verbose: bool, stream: bool = False,
                            show_thinking: bool = False) -> tuple[bool, str]:
    """
    Send a test prompt to a Chat Completions endpoint.

    `url` is the full URL to the chat completions endpoint.
    `headers` is a dict of HTTP headers to include.
    `model` is the model name to request.
    `prompt` is the user message to send.
    `stream` enables streaming mode (SSE) for the request.
    `show_thinking` controls whether reasoning/thinking tokens are displayed.

    Returns a tuple of (success, response_text).
    """
    setup_logging(verbose)
    log = logging.getLogger("TestCommand")

    payload: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 200}
    if stream:
        payload["stream"] = True

    log.debug("Sending test request to %s (stream=%s)", url, stream)
    log.debug("Using model: %s", model)

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return False, f"HTTP {resp.status}: {error_text}"

                if stream:
                    return await _read_streaming_response(resp, log, show_thinking=show_thinking)
                else:
                    return await _read_standard_response(resp, show_thinking=show_thinking)
        except aiohttp.ClientConnectorError as e:
            return False, f"Connection refused: {e}"
        except aiohttp.ClientError as e:
            return False, f"Connection error: {e}"
        except Exception as e:
            return False, f"Error: {e}"


async def _read_standard_response(resp: aiohttp.ClientResponse, show_thinking: bool = False) -> tuple[bool, str]:
    """
    Read a standard (non-streaming) Chat Completions response.

    If `show_thinking` is True, reasoning/thinking tokens are prepended with markers.

    Returns a tuple of (success, response_text).
    """
    data = await resp.json()
    if "choices" in data and len(data["choices"]) > 0:
        message = data["choices"][0].get("message", {})
        content = (message.get("content") or "").strip()
        reasoning = (message.get("reasoning_content") or message.get("reasoning") or "").strip()
        result_parts: list[str] = []
        if show_thinking and reasoning:
            result_parts.append(f"<thinking>{reasoning}</thinking>\n")
        result_parts.append(content)
        return True, "".join(result_parts)
    else:
        return False, f"Unexpected response format: {data}"


async def _read_streaming_response(resp: aiohttp.ClientResponse, log: logging.Logger, show_thinking: bool = False) -> tuple[bool, str]:
    """
    Read a streaming (SSE) Chat Completions response, printing tokens live as they arrive.

    Parses `data:` lines from the SSE stream and prints content deltas in real-time.
    If `show_thinking` is True, reasoning/thinking tokens are also displayed with markers.

    Returns a tuple of (success, assembled_content).
    """
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    chunk_count = 0
    in_thinking = False
    started_output = False

    async for raw_chunk in resp.content.iter_any():
        text = raw_chunk.decode("utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                continue
            try:
                event = json.loads(data_str)
                chunk_count += 1
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    # Handle thinking/reasoning tokens
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning is not None:
                        thinking_parts.append(reasoning)
                        if show_thinking:
                            if not started_output:
                                click.echo("\nResponse: ", nl=False)
                                started_output = True
                            if not in_thinking:
                                click.echo("<thinking>", nl=False)
                                in_thinking = True
                            click.echo(reasoning, nl=False)
                    # Handle content tokens (skip empty strings from reasoning-only chunks)
                    if delta.get("content"):
                        if not started_output:
                            click.echo("\nResponse: ", nl=False)
                            started_output = True
                        if in_thinking:
                            click.echo("</thinking>", nl=False)
                            in_thinking = False
                        content_parts.append(delta["content"])
                        click.echo(delta["content"], nl=False)
            except (json.JSONDecodeError, ValueError):
                log.debug("Skipping non-JSON SSE line: %s", data_str[:80])

    if in_thinking:
        click.echo("</thinking>", nl=False)
    if started_output:
        click.echo()  # Final newline after streamed output

    log.debug("Received %d streaming chunks", chunk_count)
    if content_parts:
        return True, "".join(content_parts).strip()
    elif thinking_parts:
        # Got thinking tokens but no content — still a success
        return True, "(thinking-only response, no text content)"
    elif chunk_count > 0:
        # Got chunks but no content (e.g. tool call only) — still a success
        return True, "(streaming response received, no text content)"
    else:
        return False, "No streaming chunks received"


def _print_test_result(success: bool, response: str, ctx: click.Context, streamed: bool = False, is_default_prompt: bool = True) -> None:
    """
    Print the result of a test prompt and exit with appropriate code.

    If `streamed` is True, the response was already printed live and is not re-printed.
    If `is_default_prompt` is True, the response is validated against the expected "OK" output.
    """
    if success:
        if not streamed:
            click.echo(f"\nResponse: {response}")
        if is_default_prompt:
            if response.strip().rstrip(".").upper() == "OK":
                click.echo("\n[SUCCESS] Test passed!")
            else:
                click.echo("\n[WARNING] Received a response, but with unexpected content.")
                click.echo("This may indicate the endpoint is working but the model did not follow the test prompt exactly.")
        else:
            click.echo("\n[SUCCESS] Response received.")
    else:
        click.echo(f"\n[FAILED] {response}", err=True)
        ctx.exit(1)


@main.command(name="test-gate")
@click.option("--host", "-h", default=None, help="Host of the running InferenceGate instance")
@click.option("--port", "-p", default=None, type=int, help="Port of the running InferenceGate instance")
@click.option("--model", "-m", default=None, help="Model to use for the test (default: gpt-4o-mini)")
@click.option("--prompt", default=None, help="Custom prompt to send")
@click.option("--stream/--no-stream", default=False, help="Enable or disable streaming mode (default: non-streaming)")
@click.option("--show-thinking/--hide-thinking", default=False, help="Show or hide model thinking/reasoning tokens (default: hide)")
@click.option("--verbose", "-v", is_flag=True, default=None, help="Enable verbose logging")
@click.pass_context
def test_gate(ctx: click.Context, host: str | None, port: int | None, model: str | None, prompt: str | None, stream: bool,
              show_thinking: bool, verbose: bool | None) -> None:
    """
    Test a running InferenceGate instance.

    Sends a test prompt to a running InferenceGate proxy to verify it is
    accepting and processing requests correctly. Uses the same host/port
    from the configuration, so you don't need to pass them explicitly.

    No API key is needed — the running instance already has it configured.

    Use --stream to test streaming (SSE) responses, --no-stream (default) for standard JSON responses.
    Use --show-thinking to display model reasoning/thinking tokens.

    Command-line options override configuration file values.
    """
    config = get_config(ctx)

    # Apply config defaults, command-line overrides
    actual_host = host if host is not None else config.host
    actual_port = port if port is not None else config.port
    actual_model = model if model is not None else config.test_model
    actual_prompt = prompt if prompt is not None else config.test_prompt
    actual_verbose = verbose if verbose is not None else config.verbose

    gate_url = f"http://{actual_host}:{actual_port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    click.echo(f"Testing InferenceGate at http://{actual_host}:{actual_port}...")
    click.echo(f"Using model: {actual_model}")
    click.echo(f"Streaming: {stream}")

    success, response = asyncio.run(
        _send_test_prompt(gate_url, headers, actual_model, actual_prompt, actual_verbose, stream=stream, show_thinking=show_thinking))
    _print_test_result(success, response, ctx, streamed=stream, is_default_prompt=(prompt is None))


@main.command(name="test-upstream")
@click.option("--upstream", "-u", default=None, help="Upstream OpenAI API base URL")
@click.option("--api-key", "-k", envvar="OPENAI_API_KEY", default=None, help="OpenAI API key")
@click.option("--model", "-m", default=None, help="Model to use for the test (default: gpt-4o-mini)")
@click.option("--prompt", default=None, help="Custom prompt to send")
@click.option("--stream/--no-stream", default=False, help="Enable or disable streaming mode (default: non-streaming)")
@click.option("--show-thinking/--hide-thinking", default=False, help="Show or hide model thinking/reasoning tokens (default: hide)")
@click.option("--verbose", "-v", is_flag=True, default=None, help="Enable verbose logging")
@click.pass_context
def test_upstream(ctx: click.Context, upstream: str | None, api_key: str | None, model: str | None, prompt: str | None, stream: bool,
                  show_thinking: bool, verbose: bool | None) -> None:
    """
    Test the connection to the upstream API directly.

    Sends a test prompt directly to the upstream API (bypassing InferenceGate)
    to verify that the API key and endpoint are working correctly.

    Use --stream to test streaming (SSE) responses, --no-stream (default) for standard JSON responses.
    Use --show-thinking to display model reasoning/thinking tokens.

    Command-line options override configuration file values.
    """
    config = get_config(ctx)

    # Apply config defaults, command-line overrides.  ``test`` calls upstream directly,
    # bypassing Gate; the user must supply connection details explicitly via flags or env vars.
    actual_upstream = upstream
    actual_api_key = api_key
    actual_model = model if model is not None else config.test_model
    actual_prompt = prompt if prompt is not None else config.test_prompt
    actual_verbose = verbose if verbose is not None else config.verbose

    if not actual_upstream:
        click.echo("Error: No upstream URL provided.  Pass --upstream <URL> explicitly.", err=True)
        ctx.exit(1)
        return
    if not actual_api_key:
        click.echo("Error: No API key provided. Set OPENAI_API_KEY environment variable or use --api-key.", err=True)
        ctx.exit(1)
        return

    url = f"{actual_upstream.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {actual_api_key}"}

    click.echo(f"Testing upstream API at {actual_upstream}...")
    click.echo(f"Using model: {actual_model}")
    click.echo(f"Streaming: {stream}")

    success, response = asyncio.run(
        _send_test_prompt(url, headers, actual_model, actual_prompt, actual_verbose, stream=stream, show_thinking=show_thinking))
    _print_test_result(success, response, ctx, streamed=stream, is_default_prompt=(prompt is None))


# =============================================================================
# Config command group
# =============================================================================


@main.group()
@click.pass_context
def config(ctx: click.Context) -> None:
    """Configuration management commands."""


@config.command(name="show")
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Show current configuration."""
    cfg = get_config(ctx)
    manager: ConfigManager = ctx.obj["config_manager"]

    click.echo(f"Configuration file: {manager.get_config_path()}")
    click.echo(f"File exists: {manager.exists()}")
    click.echo()
    click.echo("Current settings:")
    click.echo(f"  host: {cfg.host}")
    click.echo(f"  port: {cfg.port}")
    click.echo(f"  cache_dir: {cfg.cache_dir}")
    click.echo(f"  verbose: {cfg.verbose}")
    click.echo(f"  record_timeout: {cfg.record_timeout}")
    click.echo(f"  fuzzy_model: {cfg.fuzzy_model}")
    click.echo(f"  fuzzy_sampling: {cfg.fuzzy_sampling}")
    click.echo(f"  test_model: {cfg.test_model}")
    click.echo(f"  test_prompt: {cfg.test_prompt[:50]}..." if len(cfg.test_prompt) > 50 else f"  test_prompt: {cfg.test_prompt}")
    # New endpoints/models routing schema.
    click.echo(f"  endpoints: {len(cfg.endpoints)} configured")
    for name, ep in cfg.endpoints.items():
        masked = "***" if ep.api_key else "(none)"
        click.echo(f"    - {name}: url={ep.url} api_key={masked} proxy={ep.proxy or '(none)'} timeout={ep.timeout}")
    click.echo(f"  models: {len(cfg.models)} routes")
    for route in cfg.models:
        click.echo(f"    - pattern={route.pattern!r} -> endpoint={route.endpoint!r}")


@config.command(name="init")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing configuration file")
@click.pass_context
def config_init(ctx: click.Context, force: bool) -> None:
    """Initialize a default configuration file."""
    manager: ConfigManager = ctx.obj["config_manager"]

    if manager.exists() and not force:
        click.echo(f"Configuration file already exists at {manager.get_config_path()}")
        click.echo("Use --force to overwrite.")
        return

    cfg = manager.create_default()
    click.echo(f"Created default configuration file at {manager.get_config_path()}")
    click.echo()
    click.echo("Edit this file to customize your settings.")
    click.echo("You can also set OPENAI_API_KEY environment variable for API key.")


@config.command(name="path")
@click.pass_context
def config_path(ctx: click.Context) -> None:
    """Show the path to the configuration file."""
    manager: ConfigManager = ctx.obj["config_manager"]
    click.echo(manager.get_config_path())


if __name__ == "__main__":
    main()
