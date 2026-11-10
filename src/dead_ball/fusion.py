"""Layer C boundary refinement, Layer D untagged stoppages, Layer E p_dead fusion."""

from __future__ import annotations

import numpy as np

from src.dead_ball.contract import N_FRAMES, DeadSegment, DeadSegmentKind
from src.dead_ball.cusum import cusum_changepoints
from src.dead_ball.hysteresis import apply_hysteresis
from src.dead_ball.kinematics import TrackingSignals

KIND_PRIORITY: dict[DeadSegmentKind, int] = {
    "goal": 6,
    "kick_off": 5,
    "free_kick": 4,
    "corner": 3,
    "goal_kick": 2,
    "throw_in": 1,
    "other": 0,
    "non_play": -1,
}


def _longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
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


def refine_candidate_bounds(
    candidate: DeadSegment,
    signals: TrackingSignals,
    delta_pre_frames: int,
    delta_post_frames: int,
    cusum_thresholds: dict[str, dict[str, float]] | None = None,
) -> DeadSegment:
    """Snap tag windows to kinematic start (speed drop / ball out) and restart impulse."""
    search_start = max(0, candidate.start - delta_pre_frames)
    search_end = min(N_FRAMES - 1, candidate.end + delta_post_frames)
    sl = slice(search_start, search_end + 1)
    local = signals.evidence[sl]
    dead_local = apply_hysteresis(local, low_threshold=0.35, high_threshold=0.60)

    run = _longest_true_run(dead_local)
    if run is None:
        return candidate

    start = search_start + run[0]
    end = search_start + run[1]

    # Use per-event-type CUSUM thresholds if provided
    thresholds = cusum_thresholds or {}
    event_thresholds = thresholds.get(candidate.kind, {"threshold": 4.0, "drift": 0.2})
    threshold = event_thresholds.get("threshold", 4.0)
    drift = event_thresholds.get("drift", 0.2)

    down, up = cusum_changepoints(signals.median_speed[sl], threshold=threshold, drift=drift)
    if len(down):
        # nearest speed collapse to the hysteresis start
        rel = int(down[np.argmin(np.abs(down - run[0]))])
        snapped = search_start + rel
        if abs(snapped - start) <= 25:
            start = snapped
    if len(up):
        rel = int(up[np.argmin(np.abs(up - run[1]))])
        snapped = search_start + rel
        if abs(snapped - end) <= 25:
            end = snapped

    start = int(np.clip(start, search_start, search_end))
    end = int(np.clip(end, start, search_end))

    start_sigma = abs(start - candidate.start) / 2.0
    end_sigma = abs(end - candidate.end) / 2.0
    return DeadSegment(
        start=start,
        end=end,
        kind=candidate.kind,
        confidence=min(0.95, candidate.confidence + 0.05),
        start_sigma=float(start_sigma),
        end_sigma=float(end_sigma),
        team=candidate.team,
    )


def detect_untagged_stoppages(
    signals: TrackingSignals,
    non_play: np.ndarray,
    tag_search_mask: np.ndarray,
    min_frames: int = 200,
) -> np.ndarray:
    """Layer D: long kinematic silences with no tag (injury / VAR / offside)."""
    still = apply_hysteresis(signals.evidence, low_threshold=0.50, high_threshold=0.75)
    still &= ~non_play
    still &= ~tag_search_mask

    out = np.zeros(N_FRAMES, dtype=bool)
    i = 0
    while i < N_FRAMES:
        if still[i]:
            start = i
            while i < N_FRAMES and still[i]:
                i += 1
            end = i - 1
            length = end - start + 1
            if length < min_frames:
                continue
            quiet_ball = (
                (~signals.ball_visible[start : end + 1]).mean() > 0.4
                or float(np.median(signals.ball_speed[start : end + 1])) < 1.0
            )
            if quiet_ball:
                out[start : end + 1] = True
        else:
            i += 1
    return out


def fuse_p_dead(
    non_play: np.ndarray,
    tag_search_mask: np.ndarray,
    refined_mask: np.ndarray,
    untagged_mask: np.ndarray,
    evidence: np.ndarray | None,
) -> np.ndarray:
    """Layer E: combine structure, tags, refined bounds and kinematics into p_dead."""
    # Search windows and raw evidence stay below 0.5 so is_dead() is a hard
    # decision. The decoder can still read 0.3–0.45 as a soft prior.
    p = np.full(N_FRAMES, 0.05, dtype=np.float32)
    if evidence is not None:
        p = np.maximum(p, 0.10 + 0.35 * np.clip(evidence, 0.0, 1.0)).astype(np.float32)
    p = np.maximum(p, np.where(tag_search_mask, 0.40, 0.0).astype(np.float32))
    p = np.maximum(p, np.where(untagged_mask, 0.80, 0.0).astype(np.float32))
    p = np.maximum(p, np.where(refined_mask, 0.90, 0.0).astype(np.float32))
    p = np.where(non_play, np.float32(1.0), p).astype(np.float32)
    return np.clip(p, 0.0, 1.0).astype(np.float32)


def annotate_segments(
    mask: np.ndarray,
    tag_segments: list[DeadSegment],
    non_play: np.ndarray,
) -> tuple[DeadSegment, ...]:
    """Assign kind/team from overlapping tag candidates; non-play stays structural."""
    segments: list[DeadSegment] = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            start = i
            while i < n and mask[i]:
                i += 1
            end = i - 1
            if non_play[start : end + 1].all():
                segments.append(
                    DeadSegment(start=start, end=end, kind="non_play", confidence=1.0)
                )
            else:
                segments.append(_label_from_tags(start, end, tag_segments, non_play))
        else:
            i += 1
    return tuple(segments)


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _label_from_tags(
    start: int,
    end: int,
    tag_segments: list[DeadSegment],
    non_play: np.ndarray,
) -> DeadSegment:
    # Ignore the non-play portion when labelling in-game dead (kick-off vs HT).
    play_start = start
    while play_start <= end and non_play[play_start]:
        play_start += 1
    play_end = end
    while play_end >= play_start and non_play[play_end]:
        play_end -= 1
    if play_start > play_end:
        return DeadSegment(start=start, end=end, kind="non_play", confidence=1.0)

    best: DeadSegment | None = None
    best_key = (-1, -1)
    for cand in tag_segments:
        ov = _overlap(play_start, play_end, cand.start, cand.end)
        if ov <= 0:
            continue
        key = (ov, KIND_PRIORITY.get(cand.kind, 0))
        if key > best_key:
            best_key = key
            best = cand

    if best is None:
        return DeadSegment(
            start=start,
            end=end,
            kind="other",
            confidence=0.70,
        )
    return DeadSegment(
        start=start,
        end=end,
        kind=best.kind,
        confidence=best.confidence,
        start_sigma=best.start_sigma,
        end_sigma=best.end_sigma,
        team=best.team,
    )
