"""Moduł Logiki Posiadania Piłki — specyfikacja v3 ("10/10").

Właściciel: mrudnicki6. Zakres: §4.1–4.8 finalnej specyfikacji
(possession segment: team assignment, emissions, restart re-init,
semi-Markov decoder, p_home/confidence/boundaries).

Granice modułu: etykiety 'D' należą do modułu martwej piłki
(PSzarkowicz), ostateczne etykiety i polityka 'X' — do dekodera
czasowego (adriankulik13). Tu emitujemy wyłącznie ewidencję.
"""

from .ball import BallTrack, clean_ball_track
from .config import FPS, PITCH_L_M, PITCH_W_M, PossessionParams
from .decoder import DecodeResult, decode_possession
from .emissions import EmissionResult, compute_emissions
from .geometry import (
    ball_xy_meters,
    detect_coord_scale,
    infer_slot_teams,
    team_arrays,
)
from .pipeline import PossessionResult, predict_possession
from .restarts import RestartEvent, detect_restarts, live_segments

__all__ = [
    "FPS",
    "PITCH_L_M",
    "PITCH_W_M",
    "PossessionParams",
    "BallTrack",
    "clean_ball_track",
    "DecodeResult",
    "decode_possession",
    "EmissionResult",
    "compute_emissions",
    "ball_xy_meters",
    "detect_coord_scale",
    "infer_slot_teams",
    "team_arrays",
    "PossessionResult",
    "predict_possession",
    "RestartEvent",
    "detect_restarts",
    "live_segments",
]
