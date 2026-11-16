"""Data contract for the Dead Ball detection module.

Defines the output interfaces and data structures exchanged between the
dead ball detector, feature engineering, live possession classifier,
and temporal decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import numpy.typing as npt

# Total number of video frames in the reference tactical recording (at 25 fps)
N_FRAMES: int = 199_319

# Match half boundaries from official competition constants
FIRST_HALF_START: int = 11_185
FIRST_HALF_END: int = 90_651
SECOND_HALF_START: int = 110_808
SECOND_HALF_END: int = 190_197

DeadSegmentKind = Literal[
    "throw_in",
    "corner",
    "goal_kick",
    "free_kick",
    "kick_off",
    "goal",
    "other",
    "non_play",
]


@dataclass(frozen=True)
class DeadSegment:
    """Represents a continuous time interval where the ball is dead."""

    start: int  # First dead-ball frame index (inclusive)
    end: int  # Last dead-ball frame index (inclusive)
    kind: DeadSegmentKind  # Event type
    confidence: float = 1.0  # Estimated confidence in [0.0, 1.0]
    start_sigma: float = 0.0  # Boundary uncertainty in frames (std dev)
    end_sigma: float = 0.0  # Boundary uncertainty in frames (std dev)
    team: str | None = None  # Smart-tagging team if present; throw-ins are often N/A

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"Invalid DeadSegment: start ({self.start}) > end ({self.end})")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    @property
    def duration_frames(self) -> int:
        return self.end - self.start + 1

    @property
    def duration_seconds(self) -> float:
        return self.duration_frames / 25.0


@dataclass(frozen=True)
class DeadBallOutput:
    """Complete output produced by the Dead Ball detection module for the entire match."""

    p_dead: npt.NDArray[np.float32]  # Shape: (N_FRAMES,), probability ball is dead [0, 1]
    non_play: npt.NDArray[np.bool_]  # Shape: (N_FRAMES,), True for pre-game, halftime, post-game
    segments: Tuple[DeadSegment, ...]  # Sorted, non-overlapping dead segments
    restart_frames: npt.NDArray[np.int32]  # Frames where play restarts (D -> live)

    def __post_init__(self) -> None:
        if len(self.p_dead) != N_FRAMES:
            raise ValueError(f"p_dead length mismatch: expected {N_FRAMES}, got {len(self.p_dead)}")
        if len(self.non_play) != N_FRAMES:
            raise ValueError(
                f"non_play length mismatch: expected {N_FRAMES}, got {len(self.non_play)}"
            )

    def is_dead(self, threshold: float = 0.5) -> npt.NDArray[np.bool_]:
        """Returns boolean mask of dead frames based on probability threshold or non_play."""
        return self.non_play | (self.p_dead >= threshold)
