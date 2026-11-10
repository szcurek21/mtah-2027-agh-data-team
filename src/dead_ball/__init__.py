"""Dead Ball module for More Than A Hack 2027."""

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
    DeadSegmentKind,
)
from src.dead_ball.cusum import cusum_changepoints
from src.dead_ball.detector import (
    BaselineDeadBallDetector,
    DeadBallDetector,
    get_match_structure_segments,
)
from src.dead_ball.filters import savitzky_golay
from src.dead_ball.hysteresis import (
    apply_hysteresis,
    filter_min_durations,
    mask_to_dead_segments,
)
from src.dead_ball.smart_tagging import (
    compare_tag_clocks,
    extract_dead_candidates_from_tags,
    parse_smart_tagging,
    parse_smart_tagging_xml,
    seconds_to_frame,
)

__all__ = [
    "N_FRAMES",
    "FIRST_HALF_START",
    "FIRST_HALF_END",
    "SECOND_HALF_START",
    "SECOND_HALF_END",
    "DeadSegment",
    "DeadSegmentKind",
    "DeadBallOutput",
    "DeadBallDetector",
    "BaselineDeadBallDetector",
    "get_match_structure_segments",
    "savitzky_golay",
    "apply_hysteresis",
    "filter_min_durations",
    "mask_to_dead_segments",
    "cusum_changepoints",
    "parse_smart_tagging",
    "parse_smart_tagging_xml",
    "extract_dead_candidates_from_tags",
    "compare_tag_clocks",
    "seconds_to_frame",
    "calibrate_tag_offsets",
    "apply_tag_corrections",
]
