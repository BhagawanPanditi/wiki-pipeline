"""Shared CLI Configuration & Dynamic Per-Stage Settings.
------------------------------------------------------
Provides minimal, unified argument parsing and dynamic per-stage settings
resolution (servers, concurrencies, max_tokens, timeouts) directly from config.json.
"""

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import _read_config

DEFAULT_RUN_NAME = "default"


@dataclass(frozen=True)
class StageProfile:
    """Settings profile for one stage."""
    key: str
    concurrency_per_server: int
    request_timeout: float
    connect_timeout: float = 30.0
    max_tokens: Optional[int] = None
    attempts: int = 4


@dataclass
class StageSettings:
    """Resolved runtime settings for a pipeline stage."""
    profile: StageProfile
    servers: Dict[str, int]
    concurrency_per_server: int
    model_name: str
    max_tokens: Optional[int]
    attempts: int


class StageArgumentParser(argparse.ArgumentParser):
    """Argument parser that normalises the optional ``--domain`` flag."""

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        if hasattr(parsed, "domain"):
            parsed.domain_label = parsed.domain or None
            parsed.domain = parsed.domain or DEFAULT_RUN_NAME
        return parsed


def resolve_settings(args: argparse.Namespace, stage: str) -> StageSettings:
    """Resolve runtime settings directly from config.json.
    
    Servers can be specified globally by device type (e.g. dgx, a40, l40),
    and each stage configures concurrency per device type in config.json.
    """
    cfg = _read_config()
    if not cfg:
        raise ValueError("config.json is missing or empty! All pipeline settings must be defined in config.json.")

    raw_servers = cfg.get("servers")
    if not raw_servers:
        raise ValueError("Missing 'servers' in config.json!")

    model = cfg.get("model_name") or cfg.get("model")
    if not model:
        raise ValueError("Missing 'model_name' in config.json!")

    stage_concurrency_cfg = cfg.get("stage_concurrency", {})
    stage_cfg = stage_concurrency_cfg.get(stage, {})
    if not isinstance(stage_cfg, dict):
        stage_cfg = {"default": int(stage_cfg)}

    # Map each server URL to its stage concurrency
    server_map: Dict[str, int] = {}

    if isinstance(raw_servers, dict):
        for dev_type, urls in raw_servers.items():
            conc = int(stage_cfg.get(dev_type, stage_cfg.get("concurrency", 16)))
            url_list = [urls] if isinstance(urls, str) else urls
            for u in url_list:
                server_map[u.strip()] = conc
    elif isinstance(raw_servers, list):
        default_conc = int(stage_cfg.get("concurrency", 16))
        for u in raw_servers:
            u_str = u.strip()
            # Match device type substring if present (e.g. dgx, a40, l40)
            matched_conc = default_conc
            for dev_type, conc in stage_cfg.items():
                if dev_type in u_str:
                    matched_conc = int(conc)
                    break
            server_map[u_str] = matched_conc

    max_conc = max(server_map.values()) if server_map else 16
    max_tokens = int(stage_cfg.get("max_tokens", 4096))
    request_timeout = float(stage_cfg.get("request_timeout", 600.0))
    attempts = int(stage_cfg.get("attempts", cfg.get("attempts", 4)))

    profile = StageProfile(
        key=stage,
        concurrency_per_server=max_conc,
        request_timeout=request_timeout,
        connect_timeout=30.0,
        max_tokens=max_tokens,
        attempts=attempts,
    )

    return StageSettings(
        profile=profile,
        servers=server_map,
        concurrency_per_server=max_conc,
        model_name=model,
        max_tokens=max_tokens,
        attempts=attempts,
    )


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    stage: str,
    description: str,
    include_resume: bool = False,
    include_input_path: bool = False,
) -> argparse.ArgumentParser:
    """Attach the minimal shared argument block for pipeline stages."""
    parser.add_argument("--wiki_path", "--wiki_dir", dest="wiki_path", type=str, required=True,
                        help="Output folder where wiki data is stored.")
    if include_input_path:
        parser.add_argument("--input_path", type=str, default=None,
                            help="Path to input corpus file (.json or .jsonl).")
    parser.add_argument("--disable_thinking", action="store_true", default=False,
                        help="Disable Qwen3.8-27B thinking mode.")
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="Explicitly enable Qwen3.8-27B thinking mode.")
    if include_resume:
        parser.add_argument("--resume", action="store_true",
                            help="Clear incomplete marks and continue from checkpoint.")
    return parser


def build_parser(*, stage: str, description: str, **kwargs: Any) -> argparse.ArgumentParser:
    """Create a minimal parser preloaded with essential stage arguments."""
    return add_common_args(StageArgumentParser(description=description),
                           stage=stage, description=description, **kwargs)
