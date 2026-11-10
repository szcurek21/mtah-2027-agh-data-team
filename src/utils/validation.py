"""Submission validation module.

Implements the official competition sanity checks to ensure any generated
CSV meets all formatting and integrity rules before submission.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

ALLOWED_LABELS = frozenset({"H", "A", "D", "X"})
EXPECTED_FRAMES = 199_319


class ValidationError(Exception):
    """Raised when a submission file violates competition rules."""
    pass


def validate_submission_file(csv_path: str | Path) -> Tuple[bool, Dict[str, int]]:
    """Validates a submission CSV against the competition rules.

    Args:
        csv_path: Path to the submission CSV file.

    Returns:
        Tuple of (is_valid, label_counts_dict)

    Raises:
        ValidationError: If any rule is violated.
    """
    path = Path(csv_path)
    if not path.exists():
        raise ValidationError(f"File does not exist: {path}")

    counts: Dict[str, int] = {"H": 0, "A": 0, "D": 0, "X": 0}
    transitions = 0
    prev_label = None

    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValidationError("File is empty")

        if header != ["frame", "label"]:
            raise ValidationError(f"Invalid header: expected ['frame', 'label'], got {header}")

        expected_frame = 0
        for row_idx, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValidationError(f"Row {row_idx}: expected 2 columns, got {len(row)}")

            frame_str, label = row[0].strip(), row[1].strip()

            try:
                frame_num = int(frame_str)
            except ValueError:
                raise ValidationError(f"Row {row_idx}: frame '{frame_str}' is not an integer")

            if frame_num != expected_frame:
                raise ValidationError(
                    f"Row {row_idx}: non-sequential frame index (expected {expected_frame}, got {frame_num})"
                )

            if label not in ALLOWED_LABELS:
                raise ValidationError(f"Row {row_idx}: invalid label '{label}' (allowed: H, A, D, X)")

            counts[label] += 1

            if prev_label is not None and label != prev_label:
                transitions += 1
            prev_label = label

            expected_frame += 1

    if expected_frame != EXPECTED_FRAMES:
        raise ValidationError(
            f"Invalid row count: expected {EXPECTED_FRAMES} frames, got {expected_frame}"
        )

    print("=" * 60)
    print(f"✅ SUBMISSION VALIDATION PASSED: {path.name}")
    print(f"Total rows: {expected_frame:,}")
    print(
        f"Distribution: H={counts['H']} ({counts['H']/expected_frame*100:.1f}%), "
        f"A={counts['A']} ({counts['A']/expected_frame*100:.1f}%), "
        f"D={counts['D']} ({counts['D']/expected_frame*100:.1f}%), "
        f"X={counts['X']} ({counts['X']/expected_frame*100:.1f}%)"
    )
    print(f"Transitions: {transitions:,}")
    print("=" * 60)

    return True, counts
