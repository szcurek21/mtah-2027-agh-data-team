"""Run manifest written beside a submission CSV for byte-level reproducibility."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_manifest(
    output_csv: Path,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write `<submission>.manifest.json` next to the CSV."""
    output_csv = Path(output_csv)
    payload: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "python": sys.version,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "git_head": _git_head(),
        "uv_lock_sha256": _file_sha256(Path("uv.lock")),
        "submission_sha256": _file_sha256(output_csv),
        "output_csv": str(output_csv),
    }
    if extra:
        payload.update(extra)

    manifest_path = output_csv.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path
