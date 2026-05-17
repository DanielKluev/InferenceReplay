"""
Configuration management for InferenceGate.

Handles loading and saving configuration from YAML files.
Default configuration file is stored at `$USERDIR/.InferenceGate/config.yaml`.

Key classes: `Config`, `ConfigManager`
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class EndpointDef(BaseModel):
    """
    YAML representation of a single named upstream endpoint.

    Mirrors :class:`inference_gate.outflow.model_router.EndpointConfig` but is a
    Pydantic model so it can be loaded directly from YAML.
    """

    url: str = Field(..., description="Base URL of the upstream API")
    api_key: str | None = Field(default=None, description="Optional Bearer token for authentication")
    proxy: str | None = Field(default=None, description="Optional HTTP proxy URL for upstream requests")
    timeout: float | None = Field(default=None, description="Per-endpoint upstream timeout in seconds; falls back to record_timeout")


class ModelRouteDef(BaseModel):
    """
    YAML representation of a single routing rule.

    ``endpoint=None`` registers the route as an *offline sentinel* — the model
    name is contracted but no live upstream is available; record-mode requests
    matching this rule receive HTTP 503 ``model_offline``.
    """

    pattern: str = Field(..., description="Exact model name or fnmatch glob (e.g. 'Gemma4:*', '*')")
    endpoint: str | None = Field(default=None, description="Endpoint name from the 'endpoints' map, or null for offline sentinel")


class Config(BaseModel):
    """
    Configuration model for InferenceGate.

    Stores all configurable options including server settings, the named
    endpoint pool, and the model→endpoint routing table.
    """

    # Server settings
    host: str = Field(default="127.0.0.1", description="Host to bind the server to")
    port: int = Field(default=8080, description="Port to run the server on")

    # Storage settings
    cache_dir: str = Field(default=".inference_cache", description="Directory to store cached responses")

    # Logging settings
    verbose: bool = Field(default=False, description="Enable verbose logging")

    # Streaming settings
    non_streaming_models: list[str] = Field(default_factory=list,
                                            description="Models that do not support streaming and should not be forced to stream")

    # Default upstream HTTP timeout used when an endpoint omits its own ``timeout``.
    record_timeout: float = Field(
        default=600.0, description="Default upstream HTTP timeout (seconds) used when an endpoint omits its own timeout. "
        "Also bounds how long forward_request may block during recording.")

    # Fuzzy matching toggles (off by default; opt-in per request via headers or per session via /gate/config)
    fuzzy_model: bool = Field(
        default=False, description="Enable fuzzy model matching: on cache miss, reuse entries with the same prompt "
        "but a different model")
    fuzzy_sampling: str = Field(
        default="off", description="Sampling parameter fuzzy matching level: "
        "'off' (exact only), 'soft' (non-greedy matches non-greedy), "
        "'aggressive' (greedy and non-greedy may match)")

    # Multi-reply settings for non-greedy sampling
    max_non_greedy_replies: int = Field(
        default=5, description="Maximum number of replies to collect per non-greedy cassette "
        "before switching to replay cycling")

    # Global limit on live requests
    max_live_requests: int | None = Field(
        default=None, description="Global limit on the number of non-cassette upstream requests "
        "InferenceGate will perform per session. If not provided, means infinite.")

    # Endpoint pool keyed by name.  Each endpoint becomes a pooled OutflowClient
    # (deduplicated by (url, api_key, proxy)).
    endpoints: dict[str, EndpointDef] = Field(
        default_factory=dict, description="Named upstream endpoints.  Example: "
        "{'local_spark': {'url': 'http://127.0.0.1:8001'}}")

    # Ordered routing table.  Each rule binds a model pattern to an endpoint name
    # (or to None for the offline sentinel).
    models: list[ModelRouteDef] = Field(
        default_factory=list, description="Ordered routing rules.  Each rule has 'pattern' (model name or fnmatch glob) "
        "and 'endpoint' (endpoint name from the 'endpoints' map, or null for offline sentinel).")

    # Test command settings
    test_prompt: str = Field(
        default='This is a test prompt. Reply with **ONLY** "OK." to confirm that everything is ok. DO NOT output anything else.',
        description="Default prompt for the test command")
    test_model: str = Field(default="gpt-4o-mini", description="Default model for the test command")

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_upstream_schema(cls, data: Any) -> Any:
        """
        Translate legacy single-upstream YAML fields into the new routing schema.

        Older config files used top-level ``upstream``, ``api_key``, ``proxy``,
        ``upstream_timeout`` and optional ``model_routes``.  Current runtime
        code expects named ``endpoints`` plus ordered ``models``.  Keeping this
        migration here lets existing developer configs continue to work without
        preserving the legacy fields as public ``Config`` attributes.
        """
        if not isinstance(data, dict):
            return data
        migrated = dict(data)
        if migrated.get("endpoints") or migrated.get("models"):
            return migrated

        default_upstream = migrated.get("upstream")
        default_api_key = migrated.get("api_key")
        default_proxy = migrated.get("proxy")
        default_timeout = migrated.get("upstream_timeout", migrated.get("record_timeout", 600.0))
        endpoints: dict[str, dict[str, Any]] = {}
        models: list[dict[str, Any]] = []

        legacy_routes = migrated.get("model_routes") or {}
        if isinstance(legacy_routes, dict):
            for idx, (pattern, route_cfg) in enumerate(legacy_routes.items()):
                if not isinstance(route_cfg, dict):
                    continue
                route_upstream = route_cfg.get("upstream") or route_cfg.get("url")
                if not route_upstream:
                    continue
                endpoint_name = f"legacy_route_{idx}"
                endpoints[endpoint_name] = {
                    "url": route_upstream,
                    "api_key": route_cfg.get("api_key", default_api_key),
                    "proxy": route_cfg.get("proxy", default_proxy),
                    "timeout": route_cfg.get("timeout", default_timeout),
                }
                models.append({"pattern": str(pattern), "endpoint": endpoint_name})

        if default_upstream:
            endpoints.setdefault("legacy_default", {
                "url": default_upstream,
                "api_key": default_api_key,
                "proxy": default_proxy,
                "timeout": default_timeout,
            })
            if not any(route.get("pattern") == "*" for route in models):
                models.append({"pattern": "*", "endpoint": "legacy_default"})

        if endpoints:
            migrated["endpoints"] = endpoints
        if models:
            migrated["models"] = models
        if "upstream_timeout" in migrated and "record_timeout" not in migrated:
            migrated["record_timeout"] = migrated["upstream_timeout"]
        return migrated


class ConfigManager:
    """
    Manages loading and saving InferenceGate configuration.

    Configuration is loaded from YAML files. The default location is
    `$USERDIR/.InferenceGate/config.yaml`, but a custom path can be specified.
    """

    DEFAULT_CONFIG_DIR = ".InferenceGate"
    DEFAULT_CONFIG_FILE = "config.yaml"

    def __init__(self, config_path: str | Path | None = None) -> None:
        """
        Initialize the configuration manager.

        `config_path` specifies a custom configuration file path.
        If None, the default path `$USERDIR/.InferenceGate/config.yaml` is used.
        """
        self.log = logging.getLogger("ConfigManager")
        if config_path is not None:
            self.config_path = Path(config_path)
        else:
            self.config_path = self._get_default_config_path()

    def _get_default_config_path(self) -> Path:
        """
        Get the default configuration file path.

        Returns `$USERDIR/.InferenceGate/config.yaml`.
        """
        # Use USERPROFILE on Windows, HOME on Unix
        user_dir = os.environ.get("USERPROFILE") or os.environ.get("HOME") or Path.home()
        return Path(user_dir) / self.DEFAULT_CONFIG_DIR / self.DEFAULT_CONFIG_FILE

    def load(self) -> Config:
        """
        Load configuration from the YAML file.

        If the file doesn't exist, returns default configuration.
        """
        config_dict: dict[str, Any] = {}

        if self.config_path.exists():
            self.log.debug("Loading configuration from %s", self.config_path)
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded is not None:
                    config_dict = loaded
        else:
            self.log.debug("Configuration file not found at %s, creating with defaults", self.config_path)

        config = Config(**config_dict)

        return config

    def save(self, config: Config) -> None:
        """
        Save configuration to the YAML file.

        Creates the directory structure if it doesn't exist.  ``api_key``
        fields inside endpoints are stripped before persistence so secrets
        never round-trip through the YAML file.
        """
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = config.model_dump(exclude_none=True, exclude_defaults=False)

        # Strip per-endpoint api_key entries for security; users should set them via env vars or
        # push them via /gate/config at runtime.
        for ep in config_dict.get("endpoints", {}).values():
            ep.pop("api_key", None)

        self.log.debug("Saving configuration to %s", self.config_path)
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def get_config_path(self) -> Path:
        """
        Get the current configuration file path.
        """
        return self.config_path

    def exists(self) -> bool:
        """
        Check if the configuration file exists.
        """
        return self.config_path.exists()

    def create_default(self) -> Config:
        """
        Create and save a default configuration file.

        Returns the created configuration.
        """
        config = Config()
        self.save(config)
        return config
