"""Pipeline modułu posiadania — sklejka etapów 4.1–4.8 specyfikacji v3.

Wejście: `frames` (tabela z src.decoder.load.load_frames), opcjonalnie
`teams` (slot -> 'home'/'away'), `dead_mask` (bool, z modułu martwej piłki),
`segments` (wskazówki restartów), `smart_tags`.

Wyjście: PossessionResult — CZYSTO ewidencja:
    p_home[t]      prawdopodobieństwo posiadania przez H (sigmoid z s_t)
    confidence[t]  wiarygodność klatki (w_vis + wzmocnienie przy restartach)
    labels         'H'/'A' na segmentach żywych, 'X' poza nimi
    boundaries     kandydackie granice przejęć (frame, from, to)

Etykiety 'D' i ostateczna polityka 'X'/minimalnych czasów trwania należą do
modułu martwej piłki i dekodera czasowego zespołu — ten moduł ich nie emituje.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .ball import BallTrack, clean_ball_track
from .config import FPS, PossessionParams
from .decoder import DecodeResult, decode_possession
from .emissions import EmissionResult, compute_emissions
from .geometry import (
    CoordScale,
    apply_coord_scale,
    ball_xy_meters,
    detect_coord_scale,
    infer_slot_teams,
    team_arrays,
)
from .restarts import (
    RestartEvent,
    apply_executor_teams,
    detect_restarts,
    live_segments,
)


@dataclass(frozen=True)
class PossessionResult:
    """Wynik modułu posiadania dla pełnego meczu (N = liczba klatek)."""

    p_home: np.ndarray          # (N,) float w [0,1]; 0.5 = brak ewidencji
    confidence: np.ndarray      # (N,) float w [0,1]
    labels: np.ndarray          # (N,) '<U1': 'H'/'A' na żywych, 'X' poza nimi
    boundaries: list            # [(frame, from_label, to_label)]
    decode: DecodeResult
    emissions: EmissionResult
    restarts: list[RestartEvent]
    live_mask: np.ndarray       # True = klatka żywa (od ~dead_mask)

    def frame_labels(self, dead_mask: np.ndarray | None = None) -> np.ndarray:
        """Pełna tablica etykiet do zgłoszenia: 'H'/'A'/'D'/'X'.

        'D' wstawiane wyłącznie tam, gdzie `dead_mask` potwierdza martwą piłkę
        (właścicielem etykiety D jest moduł martwej piłki — to tylko widok).
        """
        out = self.labels.copy()
        if dead_mask is not None:
            dead = np.asarray(dead_mask, dtype=bool)
            out[dead] = "D"
        return out


def _resolve_teams(
    frames: pd.DataFrame, teams: dict[str, str] | None
) -> dict[str, str]:
    if teams:
        return teams
    return infer_slot_teams(frames)


def _restart_priors(
    s: np.ndarray,
    restarts: list[RestartEvent],
    params: PossessionParams,
) -> tuple[np.ndarray, dict[int, tuple[int, float]]]:
    """Dodaje priors wykonawcy do emisji i zwraca init_priors dla dekodera.

    prior[t] += restart_prior * weight * sign * (1 - dt/window) dla
    t ∈ [frame, frame+restart_window). sign=+1 dla H.
    """
    s = s.copy()
    init_priors: dict[int, tuple[int, float]] = {}
    for r in restarts:
        if r.team not in ("H", "A"):
            continue
        sign = 1.0 if r.team == "H" else -1.0
        w = max(0.0, min(1.0, r.weight))
        f0 = r.frame
        f1 = min(f0 + params.restart_window, len(s))
        for t in range(f0, f1):
            decay = 1.0 - (t - f0) / params.restart_window
            s[t] += params.restart_prior * w * sign * decay
        code = 0 if r.team == "H" else 1
        init_priors[f0] = (code, params.restart_prior * w)
    return s, init_priors


def predict_possession(
    frames: pd.DataFrame,
    teams: dict[str, str] | None = None,
    dead_mask: np.ndarray | None = None,
    segments: list | None = None,
    smart_tags: pd.DataFrame | None = None,
    params: PossessionParams | None = None,
) -> PossessionResult:
    """Główna funkcja modułu. Czysta i deterministyczna.

    Parametry:
        frames:    tabela z `load_frames` (musi mieć ball_x, ball_y i sloty *_x/*_y)
        teams:     {slot: 'home'/'away'}; None => fallback §4.1 (wolny)
        dead_mask: bool (N,); None => cała tabela traktowana jako żywa
        segments:  wskazówki z modułu martwej piłki (restart_xy, confidence, end)
        smart_tags: opcjonalnie — tagi do cross-checku (na razie nieużywane;
                   geometria wykonawcy jest źródłem prawdy, §4.6)
    """
    params = params or PossessionParams()
    frames = frames.sort_values("video_frame", kind="stable").reset_index(drop=True)
    n = len(frames)

    # --- 4.2 skala ---
    scale: CoordScale = detect_coord_scale(frames)
    ball_m = ball_xy_meters(frames, scale)

    # --- 4.1 drużyny ---
    teams_map = _resolve_teams(frames, teams)
    players = team_arrays(frames, teams_map, scale)

    # --- 4.3 czyszczenie piłki ---
    track: BallTrack = clean_ball_track(ball_m, params)

    # --- 4.4 emisje ---
    em = compute_emissions(track, players["H"], players["A"], params)

    # --- maska żywej gry i restarty ---
    if dead_mask is None:
        dead_mask = np.zeros(n, dtype=bool)
    live_mask = ~np.asarray(dead_mask, dtype=bool)

    restarts = detect_restarts(
        dead_mask, segments=segments, ball_xy=track.xy, params=params
    )
    restarts = apply_executor_teams(
        restarts, track.xy, players["H"], players["A"], params
    )

    # --- priory wykonawców w emisjach + init priory dekodera ---
    s_adj, init_priors = _restart_priors(em.s, restarts, params)

    # --- 4.7 dekoder ---
    dec = decode_possession(s_adj, live_mask, params, init_priors)

    # --- 4.8 wyjście ---
    p_home = 1.0 / (1.0 + np.exp(-np.clip(s_adj, -params.evidence_clip,
                                         params.evidence_clip)))
    # poza żywymi klatkami brak ewidencji -> 0.5
    p_home = np.where(live_mask, p_home, 0.5)

    labels = np.full(n, "X", dtype="<U1")
    code_to_label = {1: "H", 2: "A"}
    for code, lab in code_to_label.items():
        labels[dec.labels == code] = lab

    # confidence: w_vis wzmocnione priorem restartu w oknie restart_window
    conf = em.confidence.copy()
    for r in restarts:
        if r.team not in ("H", "A"):
            continue
        f1 = min(r.frame + params.restart_window, n)
        conf[r.frame : f1] = np.maximum(
            conf[r.frame : f1], 0.5 * min(1.0, r.weight)
        )

    return PossessionResult(
        p_home=p_home.astype(np.float64),
        confidence=conf.astype(np.float64),
        labels=labels,
        boundaries=dec.boundaries,
        decode=dec,
        emissions=em,
        restarts=restarts,
        live_mask=live_mask,
    )
