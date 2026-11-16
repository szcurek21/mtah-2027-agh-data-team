"""Dead Ball Detector: match structure, tags as search windows, kinematics, fusion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.dead_ball.calibration import (
    apply_tag_corrections,
    calibrate_tag_offsets,
)
from src.dead_ball.contract import (
    FIRST_HALF_END,
    FIRST_HALF_START,
    N_FRAMES,
    SECOND_HALF_END,
    SECOND_HALF_START,
    DeadBallOutput,
    DeadSegment,
)
from src.dead_ball.fusion import (
    annotate_segments,
    detect_untagged_stoppages,
    fuse_p_dead,
    refine_candidate_bounds,
)
from src.dead_ball.hysteresis import filter_min_durations
from src.dead_ball.kinematics import extract_tracking_signals
from src.dead_ball.smart_tagging import (
    compare_tag_clocks,
    extract_dead_candidates_from_tags,
    parse_smart_tagging,
)


def build_match_structure_mask() -> np.ndarray:
    """Builds the boolean mask of non-play frames based on official match half boundaries."""
    non_play = np.zeros(N_FRAMES, dtype=bool)
    non_play[:FIRST_HALF_START] = True
    non_play[FIRST_HALF_END + 1 : SECOND_HALF_START] = True
    non_play[SECOND_HALF_END + 1 :] = True
    return non_play


def get_match_structure_segments() -> tuple[DeadSegment, ...]:
    """Returns the non-play structural segments as DeadSegment objects."""
    return (
        DeadSegment(
            start=0,
            end=FIRST_HALF_START - 1,
            kind="non_play",
            confidence=1.0,
        ),
        DeadSegment(
            start=FIRST_HALF_END + 1,
            end=SECOND_HALF_START - 1,
            kind="non_play",
            confidence=1.0,
        ),
        DeadSegment(
            start=SECOND_HALF_END + 1,
            end=N_FRAMES - 1,
            kind="non_play",
            confidence=1.0,
        ),
    )


def resolve_smart_tagging_path(
    user_path: Optional[str | Path] = None,
) -> Optional[Path]:
    """Deterministic XML discovery. Prefer tactical smart tagging."""
    if user_path is not None:
        path = Path(user_path)
        if not path.exists():
            raise FileNotFoundError(f"Smart tagging XML not found: {path}")
        return path

    data_dir = Path(os.environ.get("MTAH_DATA_DIR", Path.home() / "more-than-a-hack-2027" / "data"))
    candidates = sorted(data_dir.rglob("*smart_tagging.xml")) + sorted(
        Path("data").rglob("*smart_tagging.xml")
    )
    if not candidates:
        return None
    tactical = [p for p in candidates if "tactical_smart_tagging" in p.name]
    return tactical[0] if tactical else candidates[0]


class DeadBallDetector:
    """Layers A–E: structure, tag search windows, kinematics, untagged stoppages, fusion."""

    # Per-event-type CUSUM thresholds based on typical event duration
    CUSUM_THRESHOLDS = {
        "throw_in": {"threshold": 3.0, "drift": 0.15},  # Fast events
        "corner": {"threshold": 4.0, "drift": 0.2},
        "goal_kick": {"threshold": 5.0, "drift": 0.25},  # Slower events
        "free_kick": {"threshold": 4.5, "drift": 0.2},
        "kick_off": {"threshold": 5.0, "drift": 0.25},
        "goal": {"threshold": 5.0, "drift": 0.25},
        "other": {"threshold": 4.0, "drift": 0.2},
    }

    def __init__(
        self,
        smart_tagging_path: Optional[str | Path] = None,
        min_dead_frames: int = 50,
        min_live_gap_frames: int = 25,
        delta_pre_frames: int = 25,
        delta_post_frames: int = 50,
        allow_missing_tags: bool = False,
        measure_clock: bool = True,
        skip_tag_discovery: bool = False,
        clock_drift_threshold: int = 2,
        auto_correct_clock: bool = True,
        calibrate_tag_offsets: bool = False,
    ) -> None:
        self.non_play_mask = build_match_structure_mask()
        self.allow_missing_tags = allow_missing_tags
        self.measure_clock = measure_clock
        self.min_dead_frames = min_dead_frames
        self.min_live_gap_frames = min_live_gap_frames
        self.delta_pre_frames = delta_pre_frames
        self.delta_post_frames = delta_post_frames
        self.clock_drift_threshold = clock_drift_threshold
        self.auto_correct_clock = auto_correct_clock
        self.calibrate_tag_offsets = calibrate_tag_offsets
        self.clock_drift_correction = 0.0
        if skip_tag_discovery:
            self.smart_tagging_path = None
        else:
            self.smart_tagging_path = resolve_smart_tagging_path(smart_tagging_path)

    def predict(
        self,
        frames: pd.DataFrame | None = None,
        features: dict | None = None,
    ) -> DeadBallOutput:
        """Run A–E. `frames` is Olga's tracking table (metres, centre origin)."""
        tag_search, tag_core_mask, tag_segments = self._layer_b()
        signals = None
        if frames is not None:
            signals = extract_tracking_signals(frames, features)
        else:
            print("Layer C/D skipped: no tracking frames passed to DeadBallDetector.predict()")

        refined_segments: list[DeadSegment] = []
        refined_mask = np.zeros(N_FRAMES, dtype=bool)
        if signals is not None and tag_segments:
            for cand in tag_segments:
                refined = refine_candidate_bounds(
                    cand,
                    signals,
                    self.delta_pre_frames,
                    self.delta_post_frames,
                    cusum_thresholds=self.CUSUM_THRESHOLDS,
                )
                refined_segments.append(refined)
                refined_mask[refined.start : refined.end + 1] = True

            # Self-calibration of tag offsets if enabled
            if self.calibrate_tag_offsets:
                offsets = calibrate_tag_offsets(
                    tag_segments, signals, self.delta_pre_frames, self.delta_post_frames
                )
                print("Tag offset calibration:")
                for kind, (start_off, end_off) in offsets.items():
                    if abs(start_off) > 0.5 or abs(end_off) > 0.5:
                        print(f"  {kind}: start_offset={start_off:+.2f}, end_offset={end_off:+.2f}")
                # Apply corrections to refined segments
                refined_segments = apply_tag_corrections(refined_segments, offsets)
                # Update mask after corrections
                refined_mask = np.zeros(N_FRAMES, dtype=bool)
                for seg in refined_segments:
                    refined_mask[seg.start : seg.end + 1] = True
        else:
            refined_segments = list(tag_segments)
            refined_mask = np.array(tag_core_mask, copy=True)

        untagged = np.zeros(N_FRAMES, dtype=bool)
        evidence = None
        if signals is not None:
            untagged = detect_untagged_stoppages(signals, self.non_play_mask, tag_search)
            evidence = signals.evidence

        p_dead = fuse_p_dead(
            non_play=self.non_play_mask,
            tag_search_mask=tag_search,
            refined_mask=refined_mask,
            untagged_mask=untagged,
            evidence=evidence,
        )

        # Hard D is refined tags + untagged stoppages, not the search prior.
        in_game = ~self.non_play_mask
        in_game_dead = (refined_mask | untagged) & in_game
        in_game_filtered = filter_min_durations(
            in_game_dead,
            min_dead_frames=self.min_dead_frames,
            min_live_gap_frames=self.min_live_gap_frames,
            protected_mask=tag_core_mask,
        )
        final_dead_mask = self.non_play_mask | in_game_filtered
        p_dead = np.where(
            final_dead_mask,
            np.maximum(p_dead, np.float32(0.90)),
            np.minimum(p_dead, np.float32(0.45)),
        ).astype(np.float32)
        p_dead[self.non_play_mask] = 1.0

        structure = get_match_structure_segments()
        in_game_segments = annotate_segments(
            in_game_filtered, refined_segments or tag_segments, self.non_play_mask
        )
        # Drop accidental non_play labels from the in-game mask
        in_game_segments = tuple(s for s in in_game_segments if s.kind != "non_play")
        segments = tuple(sorted(structure + in_game_segments, key=lambda s: s.start))

        restart_frames_list = []
        for seg in segments:
            next_frame = seg.end + 1
            if next_frame < N_FRAMES and not final_dead_mask[next_frame]:
                restart_frames_list.append(next_frame)

        return DeadBallOutput(
            p_dead=p_dead,
            non_play=self.non_play_mask,
            segments=segments,
            restart_frames=np.array(restart_frames_list, dtype=np.int32),
        )

    def _layer_b(self) -> tuple[np.ndarray, np.ndarray, list[DeadSegment]]:
        empty_search = np.zeros(N_FRAMES, dtype=bool)
        empty_core = np.zeros(N_FRAMES, dtype=bool)
        path = self.smart_tagging_path
        if path is None:
            message = (
                "⚠️  WARNING: Layer B skipped - no smart_tagging.xml found under MTAH_DATA_DIR or ./data. "
                "In-play dead balls will be MISSING. This submission will have POOR RECALL. "
                "Use --allow-missing-tags only for testing, NOT for final submission."
            )
            if not self.allow_missing_tags:
                raise FileNotFoundError(message)
            print(message)
            return empty_search, empty_core, []

        if self.measure_clock:
            clock = compare_tag_clocks(path)
            print(
                "Tag clock (ElementTree vs kloppy): "
                f"matched={clock['n_matched']} "
                f"max_delta={clock['max_abs_frame_delta']} "
                f"mean_delta={clock['mean_abs_frame_delta']:.2f} "
                f"median_delta={clock['median_frame_delta']:.2f} "
                f"status={clock['status']}"
            )

            # Act on clock drift
            if clock["max_abs_frame_delta"] > self.clock_drift_threshold:
                if self.auto_correct_clock:
                    correction = -clock["median_frame_delta"]
                    self.clock_drift_correction = correction
                    print(
                        f"⚠️  Clock drift {clock['max_abs_frame_delta']} frames exceeds threshold {self.clock_drift_threshold}. "
                        f"Auto-correcting by {correction:.2f} frames (median drift)."
                    )
                else:
                    print(
                        f"⚠️  WARNING: Clock drift {clock['max_abs_frame_delta']} frames exceeds threshold {self.clock_drift_threshold}. "
                        f"Boundaries may be misaligned with Olga's orientation. "
                        f"Set auto_correct_clock=True to auto-correct."
                    )

        instances = parse_smart_tagging(path, prefer_kloppy=True)

        # Apply clock drift correction if needed
        if self.clock_drift_correction != 0.0:
            for inst in instances:
                inst["start_frame"] = int(round(inst["start_frame"] + self.clock_drift_correction))
                inst["end_frame"] = int(round(inst["end_frame"] + self.clock_drift_correction))
                # Clamp to valid range
                inst["start_frame"] = max(0, min(N_FRAMES - 1, inst["start_frame"]))
                inst["end_frame"] = max(0, min(N_FRAMES - 1, inst["end_frame"]))

        search_mask, tag_segments, core_mask = extract_dead_candidates_from_tags(
            instances=instances,
            delta_pre_frames=self.delta_pre_frames,
            delta_post_frames=self.delta_post_frames,
        )
        print(f"Layer B: {len(tag_segments)} set-piece/goal candidates from {path.name}")
        return search_mask, core_mask, tag_segments


BaselineDeadBallDetector = DeadBallDetector
