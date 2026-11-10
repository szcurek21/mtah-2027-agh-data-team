"""Stage caching for intermediate pipeline results.

Provides deterministic caching of expensive computations (tracking signals,
refined segments, etc.) with hash-based keys to enable fast iteration without
recomputing everything.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any


def _compute_cache_key(config: dict[str, Any], code_version: str | None = None) -> str:
    """Compute a stable hash key from configuration and optional code version."""
    # Sort keys for deterministic JSON serialization
    config_str = json.dumps(config, sort_keys=True, default=str)
    h = hashlib.sha256()
    h.update(config_str.encode("utf-8"))
    if code_version:
        h.update(code_version.encode("utf-8"))
    return h.hexdigest()[:16]  # First 16 chars is enough for uniqueness


def get_cache_dir() -> Path:
    """Return the cache directory path, creating it if needed."""
    cache_dir = Path(".cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def cache_stage(
    stage_name: str,
    config: dict[str, Any],
    data: Any,
    code_version: str | None = None,
) -> None:
    """Cache stage output with a hash-based key.

    Args:
        stage_name: Name of the pipeline stage (e.g., "dead_ball_signals").
        config: Configuration dict that uniquely identifies this computation.
        data: The data to cache (must be pickle-serializable).
        code_version: Optional code version string (e.g., git commit SHA).
    """
    key = _compute_cache_key(config, code_version)
    cache_dir = get_cache_dir()
    cache_file = cache_dir / f"{stage_name}_{key}.pkl"

    with open(cache_file, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cached_stage(
    stage_name: str,
    config: dict[str, Any],
    code_version: str | None = None,
) -> Any | None:
    """Load cached stage output if it exists.

    Args:
        stage_name: Name of the pipeline stage.
        config: Configuration dict that uniquely identifies this computation.
        code_version: Optional code version string.

    Returns:
        Cached data if found, None otherwise.
    """
    key = _compute_cache_key(config, code_version)
    cache_dir = get_cache_dir()
    cache_file = cache_dir / f"{stage_name}_{key}.pkl"

    if not cache_file.exists():
        return None

    with open(cache_file, "rb") as f:
        return pickle.load(f)


def clear_all_cache() -> int:
    """Delete all cached files. Returns number of files deleted."""
    cache_dir = get_cache_dir()
    files = list(cache_dir.glob("*.pkl"))
    for f in files:
        f.unlink()
    return len(files)
