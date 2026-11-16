"""4.3 Czyszczenie trajektorii piłki: SG(7,2), krótkie interpolacje, ball_visible."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FPS, PossessionParams


@dataclass(frozen=True)
class BallTrack:
    """Wynik czyszczenia trajektorii piłki (w metrach, jeśli wejście było w metrach).

    - xy: wygładzona pozycja (N, 2); NaN tam, gdzie brak danych po interpolacji
    - speed: prędkość m/s z pochodnych po wygładzonym torze; NaN przy brakach
    - visible: True iff surowe dane ALBO krótka (<= max_interp_gap) luka
    - frames_since_seen: ile klatek od ostatniej klatki z widzialną piłką
    """

    xy: np.ndarray
    speed: np.ndarray
    visible: np.ndarray
    frames_since_seen: np.ndarray


def sg_coeffs(window: int, poly: int) -> np.ndarray:
    """Współczynniki Savitzky–Golay dla środka okna (tryb 'same')."""
    if window % 2 == 0:
        raise ValueError("window musi być nieparzyste")
    half = window // 2
    x = np.arange(-half, half + 1, dtype="float64")
    V = np.vander(x, poly + 1, increasing=True)
    # wiersz odpowiadający ocenie wielomianu w x=0 (wyraz stały)
    G = np.linalg.pinv(V)
    return G[0]


def _smooth_segments(y: np.ndarray, valid: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """SG na każdym ciągłym fragmencie `valid` o długości >= len(coeffs)."""
    n = len(y)
    out = np.full_like(y, np.nan, dtype="float64")
    w = len(coeffs)
    half = w // 2
    if w > n:
        return out
    # granice segmentów valid
    change = np.flatnonzero(np.diff(valid.astype(np.int8))) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n]])
    for s, e in zip(starts, ends):
        if not valid[s]:
            continue
        seg = y[s:e]
        if len(seg) < w:
            out[s:e] = seg
            continue
        pad = np.pad(seg, half, mode="edge")
        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(pad, w)  # (L, w)
        out[s:e] = windows @ coeffs
    return out


def _interp_short_gaps(xy: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Wypełnia luki <= max_gap liniowo. Zwraca (xy_filled, interp_mask).

    xy_filled ma NaN tylko w długich lukach i poza nimi.
    """
    n = len(xy)
    valid = np.isfinite(xy).all(axis=1)
    filled = xy.copy()
    interp = np.zeros(n, dtype=bool)
    if valid.all():
        return filled, interp
    idx = np.arange(n)
    # granice nieciągłych segmentów valid
    change = np.flatnonzero(np.diff(valid.astype(np.int8))) + 1
    bounds = np.concatenate([[0], change, [n]])
    for s, e in zip(bounds[:-1], bounds[1:]):
        if valid[s]:
            continue  # to segment danych, nie luka
        gap = e - s
        if gap > max_gap:
            continue  # długa luka zostaje NaN
        # interpolacja między s-1 a e (s>=1, e<=n-1 gwarantowane, bo valid[0]/[-1]?
        # niekoniecznie — sprawdź brzeg)
        if s == 0 or e == n:
            # luka na brzegu: nie interpolujemy (brak obu końców)
            continue
        for c in range(2):
            filled[s:e, c] = np.interp(idx[s:e], [s - 1, e], [xy[s - 1, c], xy[e, c]])
        interp[s:e] = True
    return filled, interp


def clean_ball_track(ball_xy: np.ndarray, params: PossessionParams | None = None) -> BallTrack:
    """Pełne czyszczenie toru piłki wg §4.3.

    Wejście: (N, 2) pozycje piłki (dowolna jednostka, ale speed wtedy w tej
    samej jednostce/s). Zalecane podawać metry — wtedy speed jest w m/s.
    """
    params = params or PossessionParams()
    xy = np.asarray(ball_xy, dtype="float64")
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("ball_xy musi mieć kształt (N, 2)")
    n = len(xy)

    raw_visible = np.isfinite(xy).all(axis=1)
    filled, interp = _interp_short_gaps(xy, params.max_interp_gap)
    smooth_valid = np.isfinite(filled).all(axis=1)

    coeffs = sg_coeffs(params.sg_window, params.sg_poly)
    sm_x = _smooth_segments(filled[:, 0], smooth_valid, coeffs)
    sm_y = _smooth_segments(filled[:, 1], smooth_valid, coeffs)
    sm = np.stack([sm_x, sm_y], axis=1)
    # dla segmentów krótszych niż okno SG _smooth_segments już zostawiło
    # wartości wypełnione; dla długich luk NaN.
    # Uwaga: _smooth_segments zwraca NaN poza valid — wypełniamy je filled,
    # aby krótka luka bez pełnego kontekstu SG nie gubiła interpolacji.
    both_nan = ~np.isfinite(sm).all(axis=1) & smooth_valid
    sm[both_nan] = filled[both_nan]

    visible = raw_visible | interp

    # prędkość z pochodnych centralnych po wygładzonym torze [jednostka/s]
    speed = np.full(n, np.nan)
    if n > 2:
        dt = 1.0 / FPS
        dx = np.full(n, np.nan)
        dy = np.full(n, np.nan)
        dx[1:-1] = (sm[2:, 0] - sm[:-2, 0]) / (2 * dt)
        dy[1:-1] = (sm[2:, 1] - sm[:-2, 1]) / (2 * dt)
        speed = np.hypot(dx, dy)

    # frames_since_seen
    fss = np.full(n, np.nan)
    last = -1
    for i in range(n):
        if visible[i]:
            last = i
        if last >= 0:
            fss[i] = i - last

    return BallTrack(xy=sm, speed=speed, visible=visible, frames_since_seen=fss)
