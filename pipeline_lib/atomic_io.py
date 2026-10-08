"""Atomic file writers for durable checkpoints, graphs and debug logs.

A Slurm job can be preempted between ``open()`` and the final ``write()``, so
every state file is written to a sibling temporary, fsynced, then renamed over
the target. ``_fsync_dir`` also flushes the rename itself: without it a power
loss can leave the directory pointing at the old inode even though the new
file's data reached disk.
"""

import json
import os
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


def _fsync_dir(directory: Path) -> None:
    """Persist a rename so it survives power loss, where the OS allows it."""
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        # Not supported on every filesystem; the rename is still atomic.
        pass
    finally:
        os.close(dir_fd)


def atomic_write_json(path: PathLike, data: Any, *, indent: int = 2,
                      fsync_dir: bool = True) -> None:
    """Write JSON beside its destination, then atomically replace the old file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        if fsync_dir:
            _fsync_dir(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_text(path: PathLike, text: str, *, fsync_dir: bool = True) -> None:
    """Atomically replace a text file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        if fsync_dir:
            _fsync_dir(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_json(path: PathLike, default: Any = None) -> Any:
    """Read JSON, returning ``default`` when the file is absent or corrupt.

    A truncated checkpoint must never be mistaken for a valid one, so parse
    failures degrade to the default rather than propagating.
    """
    path = Path(path)
    if not path.is_file():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return default
