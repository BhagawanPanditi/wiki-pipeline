"""Pipeline Stage Manifest and Verification Gate.

Implements Audit v3 Section 3.4:
- write_stage_manifest: writes stages/<stage>.done.json containing expected and completed item counts.
- verify_stage_manifest: verifies that an upstream stage completed with zero failures before allowing the next stage to run.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from utils.atomic_io import atomic_write_json, read_json


def write_stage_manifest(wiki_path: Path, stage: str, expected: int, completed: int, failed: int = 0, **extra) -> Path:
    """Record a stage completion manifest atomically."""
    manifest_dir = wiki_path / "stages"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_dir / f"{stage}.done.json"

    data = {
        "stage": stage,
        "expected": expected,
        "completed": completed,
        "failed": failed,
        "finished_at": datetime.now().isoformat(),
        **extra,
    }
    atomic_write_json(manifest_file, data)
    return manifest_file


def verify_stage_manifest(wiki_path: Path, required_stage: str) -> Dict[str, Any]:
    """Check that required upstream stage completed cleanly. Raises RuntimeError if missing or failed."""
    manifest_file = wiki_path / "stages" / f"{required_stage}.done.json"
    if not manifest_file.is_file():
        raise RuntimeError(
            f"Required upstream stage manifest missing: {manifest_file}. "
            f"Run '{required_stage}' first."
        )

    data = read_json(manifest_file)
    if not isinstance(data, dict):
        raise RuntimeError(f"Corrupt manifest file: {manifest_file}")

    expected = data.get("expected", 0)
    completed = data.get("completed", 0)
    failed = data.get("failed", 0)

    if failed > 0 or (expected > 0 and completed < expected):
        raise RuntimeError(
            f"Upstream stage '{required_stage}' had incomplete or failed items "
            f"(expected={expected}, completed={completed}, failed={failed}). "
            f"Rerun with --resume before proceeding."
        )

    return data
