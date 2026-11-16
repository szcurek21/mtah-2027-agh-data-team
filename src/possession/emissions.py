"""4.4 Model emisji — rdzeń modułu: ewidencja s_t i pewność confidence_t.

Dla każdej klatki żywej gry, dla drużyn c ∈ {H, A}:
    d_c[t]  = odległość od piłki do najbliższego zawodnika drużyny c (metry)
    logP_c  = -d_c / τ          (proximity likelihood, τ kalibrowane)
    s_t     = w_vis[t] · clip(logP_H - logP_A, ±evidence_clip)
              = w_vis[t] · clip((d_A - d_H) / τ, ±clip)

Aksjomat lotu (§4.5): gdy piłka jest w locie (duża prędkość lub proxy
powietrzne), w_vis = 0 — stan się utrzymuje, przejęcie niemożliwe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ball import BallTrack
from .config import PossessionParams


@dataclass(frozen=True)
class EmissionResult:
    """Wynik etapu emisji dla pełnego meczu.

    - s: ewidencja na korzyść H (log-ll); 0 = brak ewidencji
    - confidence: w_vis (1 = pełna wiarygodność klatki)
    - d_home / d_away: odległości do najbliższego zawodnika (metry, NaN = brak)
    - flight: maska lotu piłki (aksjomat §4.5)
    """

    s: np.ndarray
    confidence: np.ndarray
    d_home: np.ndarray
    d_away: np.ndarray
    flight: np.ndarray


def _min_dist_chunk(ball: np.ndarray, players: np.ndarray) -> np.ndarray:
    """(B,2) x (B,M,2) -> (B,) min odległość; NaN gdy brak piłki/graczy."""
    diff = players - ball[:, None, :]
    with np.errstate(invalid="ignore"):
        dist = np.sqrt(np.sum(diff * diff, axis=2))
    dist = np.where(np.isnan(dist), np.inf, dist)
    has_player = np.isfinite(players).all(axis=2).any(axis=1)
    out = dist.min(axis=1)
    out[~has_player] = np.nan
    return out


def team_min_distances(
    ball_xy: np.ndarray,
    players_h: np.ndarray,
    players_a: np.ndarray,
    chunk: int = 20000,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimalne odległości piłka-gracz per drużyna, chunkowo (pamięć)."""
    n = len(ball_xy)
    d_h = np.full(n, np.nan)
    d_a = np.full(n, np.nan)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        b = ball_xy[s:e]
        d_h[s:e] = _min_dist_chunk(b, players_h[s:e])
        d_a[s:e] = _min_dist_chunk(b, players_a[s:e])
    return d_h, d_a


def flight_mask(track: BallTrack, d_min: np.ndarray, params: PossessionParams) -> np.ndarray:
    """Maska lotu piłki: duża prędkość LUB proxy powietrzne.

    Proxy (bez pomiaru z): szybka piłka (> scramble_speed) bez żadnego
    zawodnika w promieniu scramble_radius — „długo niekontrolowana".
    """
    speed = np.where(np.isfinite(track.speed), track.speed, 0.0)
    fast = speed > params.flight_speed
    # aerial proxy: średnio szybka piłka i nikt blisko (np. dryf po sparingu)
    nobody_near = np.isfinite(d_min) & (d_min > params.scramble_radius)
    aerial = (speed > params.scramble_speed) & nobody_near
    # długa luka NaN z konwergencją graczy — uproszczenie: po >1 s bez piłki
    # i szybkiej piłce na granicy luki traktujemy jako lot
    long_gap_fast = (track.frames_since_seen > 25) & (speed > params.scramble_speed)
    return fast | aerial | long_gap_fast


def compute_emissions(
    track: BallTrack,
    players_h: np.ndarray,
    players_a: np.ndarray,
    params: PossessionParams | None = None,
) -> EmissionResult:
    """Liczy s_t i confidence_t dla wszystkich klatek (§4.4)."""
    params = params or PossessionParams()
    ball = track.xy
    d_h, d_a = team_min_distances(ball, players_h, players_a)
    d_min = np.fmin(
        np.where(np.isfinite(d_h), d_h, np.inf), np.where(np.isfinite(d_a), d_a, np.inf)
    )
    d_min[~np.isfinite(d_min)] = np.nan

    flight = flight_mask(track, d_min, params)

    # w_vis: 0 gdy piłka niewidoczna (dluga luka) lub w locie
    w_vis = (track.visible & ~flight).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = (d_a - d_h) / params.tau  # logP_H - logP_A
    s = w_vis * np.clip(log_ratio, -params.evidence_clip, params.evidence_clip)
    s = np.where(np.isfinite(s), s, 0.0)

    return EmissionResult(
        s=s.astype(np.float64),
        confidence=w_vis,
        d_home=d_h,
        d_away=d_a,
        flight=flight,
    )
