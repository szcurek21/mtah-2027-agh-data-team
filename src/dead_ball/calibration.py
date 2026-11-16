"""Self-calibration of tag-to-kinematic offsets without ground truth.

Computes median offset between tag boundaries and kinematic changepoints
for each event type, enabling automatic correction of systematic tag timing errors.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.dead_ball.contract import DeadSegment
from src.dead_ball.cusum import cusum_changepoints
from src.dead_ball.hysteresis import apply_hysteresis
from src.dead_ball.kinematics import N_FRAMES, TrackingSignals


def _longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    """Find the longest contiguous run of True values."""
    best: tuple[int, int] | None = None
    best_len = 0
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            start = i
            while i < n and mask[i]:
                i += 1
            end = i - 1
            length = end - start + 1
            if length > best_len:
                best_len = length
                best = (start, end)
        else:
            i += 1
    return best


def compute_tag_offset(
    tag_start: int,
    tag_end: int,
    signals: TrackingSignals,
    delta_pre_frames: int = 25,
    delta_post_frames: int = 50,
) -> tuple[int, int] | None:
    """Compute offset between tag boundary and kinematic changepoint.

    Returns:
        (start_offset, end_offset) where offset = kinematic - tag
        Returns None if no reliable changepoint found.
    """
    search_start = max(0, tag_start - delta_pre_frames)
    search_end = min(N_FRAMES - 1, tag_end + delta_post_frames)
    sl = slice(search_start, search_end + 1)

    local = signals.evidence[sl]
    dead_local = apply_hysteresis(local, low_threshold=0.35, high_threshold=0.60)

    run = _longest_true_run(dead_local)
    if run is None:
        return None

    kinematic_start_local = run[0]
    kinematic_end_local = run[1]

    down, up = cusum_changepoints(signals.median_speed[sl], threshold=4.0, drift=0.2)

    start_offset = None
    end_offset = None

    if len(down):
        nearest = int(down[np.argmin(np.abs(down - kinematic_start_local))])
        kinematic_start = search_start + nearest
        if abs(kinematic_start - tag_start) <= 25:
            start_offset = kinematic_start - tag_start

    if len(up):
        nearest = int(up[np.argmin(np.abs(up - kinematic_end_local))])
        kinematic_end = search_start + nearest
        if abs(kinematic_end - tag_end) <= 25:
            end_offset = kinematic_end - tag_end

    if start_offset is None and end_offset is None:
        return None

    return (start_offset or 0, end_offset or 0)


def calibrate_tag_offsets(
    tag_segments: List[DeadSegment],
    signals: TrackingSignals,
    delta_pre_frames: int = 25,
    delta_post_frames: int = 50,
) -> Dict[str, tuple[float, float]]:
    """Compute median offset per event type.

    Args:
        tag_segments: Tag-based candidate segments
        signals: Tracking signals for kinematic analysis
        delta_pre_frames: Search window before tag
        delta_post_frames: Search window after tag

    Returns:
        Dict mapping event kind to (median_start_offset, median_end_offset)
    """
    offsets_by_kind: Dict[str, List[tuple[int, int]]] = {}

    for seg in tag_segments:
        offset = compute_tag_offset(
            seg.start, seg.end, signals, delta_pre_frames, delta_post_frames
        )
        if offset is not None:
            kind = seg.kind
            if kind not in offsets_by_kind:
                offsets_by_kind[kind] = []
            offsets_by_kind[kind].append(offset)

    # Compute median offsets per kind
    result: Dict[str, tuple[float, float]] = {}
    for kind, offsets in offsets_by_kind.items():
        if len(offsets) < 5:  # Need minimum samples for reliable median
            result[kind] = (0.0, 0.0)
            continue

        start_offsets = [o[0] for o in offsets]
        end_offsets = [o[1] for o in offsets]

        result[kind] = (float(np.median(start_offsets)), float(np.median(end_offsets)))

    return result


def apply_tag_corrections(
    tag_segments: List[DeadSegment],
    offsets: Dict[str, tuple[float, float]],
) -> List[DeadSegment]:
    """Apply calibrated offsets to tag segments.

    Args:
        tag_segments: Original tag segments
        offsets: Offset dict from calibrate_tag_offsets

    Returns:
        Corrected segments
    """
    corrected = []
    for seg in tag_segments:
        start_offset, end_offset = offsets.get(seg.kind, (0.0, 0.0))

        new_start = int(round(seg.start + start_offset))
        new_end = int(round(seg.end + end_offset))

        # Clamp to valid range
        new_start = max(0, min(N_FRAMES - 1, new_start))
        new_end = max(0, min(N_FRAMES - 1, new_end))

        corrected.append(
            DeadSegment(
                start=new_start,
                end=new_end,
                kind=seg.kind,
                confidence=seg.confidence,
                start_sigma=seg.start_sigma,
                end_sigma=seg.end_sigma,
                team=seg.team,
            )
        )

    return corrected
