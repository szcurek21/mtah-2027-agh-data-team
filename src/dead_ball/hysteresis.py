"""Hysteresis thresholding and duration-based anti-flicker filtering.

Protects the spellF1 ranking metric by preventing micro-stoppages (which would
split valid possessions) and closing short false gaps inside real dead-ball events.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.dead_ball.contract import DeadSegment, DeadSegmentKind


def apply_hysteresis(
    signal: npt.NDArray[np.floating],
    low_threshold: float,
    high_threshold: float,
    initial_state: bool = False,
) -> np.ndarray:
    """Applies dual-threshold hysteresis to a continuous 1D signal.

    Transitions to True when signal >= high_threshold.
    Transitions to False when signal <= low_threshold.
    Maintains previous state when low_threshold < signal < high_threshold.

    Args:
        signal: 1D array of float scores (e.g. dead probability or speed inverse).
        low_threshold: Value below which state switches to False.
        high_threshold: Value above which state switches to True.
        initial_state: State at frame 0.

    Returns:
        Boolean array of the same shape as signal.
    """
    if low_threshold > high_threshold:
        raise ValueError(
            f"low_threshold ({low_threshold}) cannot exceed high_threshold ({high_threshold})"
        )

    n = len(signal)
    state = np.zeros(n, dtype=bool)
    current = initial_state

    for i in range(n):
        val = signal[i]
        if val >= high_threshold:
            current = True
        elif val <= low_threshold:
            current = False
        state[i] = current

    return state


def filter_min_durations(
    mask: npt.NDArray[np.bool_],
    min_dead_frames: int = 50,  # 2.0 seconds at 25 fps
    min_live_gap_frames: int = 25,  # 1.0 second at 25 fps
    protected_mask: npt.NDArray[np.bool_] | None = None,
) -> np.ndarray:
    """Enforces minimum duration constraints to eliminate flickers and close micro-gaps.

    Args:
        mask: Boolean mask (True = dead ball, False = live ball).
        min_dead_frames: Dead stretches shorter than this are flipped to False.
        min_live_gap_frames: Live stretches shorter than this between dead stretches are flipped to True.
        protected_mask: If a short dead stretch overlaps this mask (e.g. a Smart Tag),
            it is kept. Throw-ins are often shorter than 2 s.

    Returns:
        Filtered boolean mask with identical shape.
    """
    out = np.array(mask, dtype=bool, copy=True)
    n = len(out)
    if n == 0:
        return out

    protected = (
        np.zeros(n, dtype=bool)
        if protected_mask is None
        else np.asarray(protected_mask, dtype=bool)
    )
    if len(protected) != n:
        raise ValueError("protected_mask must match mask length")

    # 1. Morphological closing: close live gaps shorter than min_live_gap_frames
    if min_live_gap_frames > 1:
        i = 0
        while i < n:
            if not out[i]:
                gap_start = i
                while i < n and not out[i]:
                    i += 1
                gap_end = i - 1
                gap_len = gap_end - gap_start + 1

                has_left_neighbor = gap_start > 0 and out[gap_start - 1]
                has_right_neighbor = gap_end < n - 1 and out[gap_end + 1]

                if has_left_neighbor and has_right_neighbor and gap_len < min_live_gap_frames:
                    out[gap_start : gap_end + 1] = True
            else:
                i += 1

    # 2. Minimum dead duration: remove unprotected dead flickers
    if min_dead_frames > 1:
        i = 0
        while i < n:
            if out[i]:
                dead_start = i
                while i < n and out[i]:
                    i += 1
                dead_end = i - 1
                dead_len = dead_end - dead_start + 1

                if dead_len < min_dead_frames and not protected[dead_start : dead_end + 1].any():
                    out[dead_start : dead_end + 1] = False
            else:
                i += 1

    return out


def mask_to_dead_segments(
    mask: npt.NDArray[np.bool_],
    kind: DeadSegmentKind = "other",
    confidence: float = 1.0,
) -> tuple[DeadSegment, ...]:
    """Converts a boolean dead-ball mask into a tuple of non-overlapping DeadSegment objects."""
    n = len(mask)
    segments = []
    i = 0

    while i < n:
        if mask[i]:
            start = i
            while i < n and mask[i]:
                i += 1
            end = i - 1
            segments.append(
                DeadSegment(
                    start=start,
                    end=end,
                    kind=kind,
                    confidence=confidence,
                    team=None,
                )
            )
        else:
            i += 1

    return tuple(segments)
