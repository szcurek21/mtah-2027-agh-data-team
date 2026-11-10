"""Geometria boiska: konwersja na metry i macierze pozycji zawodników (§4.1–4.2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import NORM_COORD_LIMIT, PITCH_L_M, PITCH_W_M, PossessionParams

CoordScale = tuple[float, float, float, float]  # (x_scale, y_scale, x_shift, y_shift)


def detect_coord_scale(frames: pd.DataFrame) -> CoordScale:
    """Zwraca skalę (a, b, cx, cy) mapującą kolumny na metry: m = c * v + shift.

    Loader kloppy w trybie "secondspectrum" podaje już metry wyśrodkowane
    w (0,0). Jeśli dane są znormalizowane [0,1]^2 (|v| <= NORM_COORD_LIMIT),
    skalujemy o wymiary boiska i przesuwamy środek na (0,0).
    """
    cols = [c for c in frames.columns if c.endswith(("_x", "_y"))]
    if not cols:
        return (1.0, 1.0, 0.0, 0.0)
    vals = frames[cols].to_numpy(dtype="float64")
    finite = np.isfinite(vals)
    if not finite.any():
        return (1.0, 1.0, 0.0, 0.0)
    vmax = np.nanmax(np.abs(vals[finite]))
    if vmax <= NORM_COORD_LIMIT:
        # [0,1]^2 z narożnika -> metry ze środkiem w (0,0)
        return (PITCH_L_M, PITCH_W_M, -PITCH_L_M / 2.0, -PITCH_W_M / 2.0)
    return (1.0, 1.0, 0.0, 0.0)


def apply_coord_scale(values: np.ndarray, scale: CoordScale) -> np.ndarray:
    """Przekształca tablicę (..., 2) kolumn [x, y] na metry."""
    sx, sy, cx, cy = scale
    out = values.astype("float64", copy=True)
    out[..., 0] = out[..., 0] * sx + cx
    out[..., 1] = out[..., 1] * sy + cy
    return out


def ball_xy_meters(frames: pd.DataFrame, scale: CoordScale | None = None) -> np.ndarray:
    """Pozycja piłki (N, 2) w metrach; NaN zostaje NaN."""
    if scale is None:
        scale = detect_coord_scale(frames)
    xy = frames[["ball_x", "ball_y"]].to_numpy(dtype="float64")
    return apply_coord_scale(xy, scale)


def _slot_ids(frames: pd.DataFrame) -> list[str]:
    return [c[:-2] for c in frames.columns if c.endswith("_x") and c != "ball_x"]


def _all_player_xy(
    frames: pd.DataFrame, scale: CoordScale
) -> tuple[np.ndarray, list[str]]:
    """Pozycje wszystkich slotów: (N, S, 2) float32 w metrach + lista slotów."""
    slots = _slot_ids(frames)
    n = len(frames)
    xy = np.full((n, len(slots), 2), np.nan, dtype=np.float32)
    if not slots:
        return xy, slots
    cols = np.empty((n, len(slots) * 2), dtype="float64")
    for i, s in enumerate(slots):
        cols[:, 2 * i] = pd.to_numeric(frames[f"{s}_x"], errors="coerce")
        cols[:, 2 * i + 1] = pd.to_numeric(frames[f"{s}_y"], errors="coerce")
    xy = apply_coord_scale(cols.reshape(n, len(slots), 2), scale).astype(np.float32)
    return xy, slots


def team_arrays(
    frames: pd.DataFrame, teams: dict[str, str], scale: CoordScale | None = None
) -> dict[str, np.ndarray]:
    """Zwraca {'H': (N, M_h, 2), 'A': (N, M_a, 2)} — pozycje w metrach, NaN = brak.

    `teams` mapuje slot -> 'home'/'away' (jak w src.decoder.load.team_of).
    Slot z nieznaną drużyną jest pomijany.
    """
    if scale is None:
        scale = detect_coord_scale(frames)
    n = len(frames)
    out: dict[str, list[np.ndarray]] = {"H": [], "A": []}
    for slot in _slot_ids(frames):
        team = teams.get(slot)
        if team not in ("home", "away"):
            continue
        xy = frames[[f"{slot}_x", f"{slot}_y"]].to_numpy(dtype="float64")
        xy = apply_coord_scale(xy, scale)
        key = "H" if team == "home" else "A"
        out[key].append(xy)
    result = {}
    for key in ("H", "A"):
        if out[key]:
            result[key] = np.stack(out[key], axis=1)  # (M, N, 2) -> (N, M, 2)
        else:
            result[key] = np.full((n, 0, 2), np.nan)
    return result


# ---------------------------------------------------------------------------
# 4.1 Fallback: przypisanie drużyn do slotów, gdy nie ma mapowania z XML
# ---------------------------------------------------------------------------

def _kmeans2(points: np.ndarray, iters: int = 30) -> np.ndarray:
    """Deterministyczny k-means (k=2) na punktach (M,2). Zwraca etykiety 0/1."""
    valid = np.isfinite(points).all(axis=1)
    pts = points[valid]
    labels_full = np.zeros(len(points), dtype=np.int8)
    if len(pts) < 2:
        return labels_full
    # inicjalizacja: punkt o minimalnej i maksymalnej wartości x+y
    score = pts[:, 0] + pts[:, 1]
    centers = np.stack([pts[np.argmin(score)], pts[np.argmax(score)]])
    labels = np.zeros(len(pts), dtype=np.int8)
    for _ in range(iters):
        d = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d.argmin(axis=1).astype(np.int8)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in (0, 1):
            members = pts[labels == k]
            if len(members):
                centers[k] = members.mean(axis=0)
    labels_full[valid] = labels
    return labels_full


def infer_slot_teams(
    frames: pd.DataFrame,
    params: PossessionParams | None = None,
    scale: CoordScale | None = None,
) -> dict[str, str]:
    """Fallback §4.1: globalne przypisanie drużyn kotwiczone przy kopnięciu.

    Kotwice = pierwsze klatki połów (start gry). W każdej kotwicy klastrowanie
    k=2 rozdziela drużyny; drużyna "home" to klaster z mniejszym średnim x
    (Barcelona atakuje w prawo po orientacji loadera). Etykiety propagowane są
    klatka-po-klatce przez najbliższego sąsiada (w obie strony od kotwicy)
    i zbierane jako głosy per slot.

    Uwaga: to fallback — preferowane jest mapowanie slot->drużyna z metadanych
    (`src.decoder.load.team_of`). Zwraca {slot: 'home'/'away'/'unknown'}.
    """
    params = params or PossessionParams()
    if scale is None:
        scale = detect_coord_scale(frames)
    n = len(frames)
    xy_all, slots = _all_player_xy(frames, scale)
    if not slots:
        return {}

    # --- kotwice: początek każdej połowy + offset ---
    anchors: list[int] = []
    if "half" in frames.columns:
        for h in sorted(frames["half"].dropna().unique()):
            first = int(frames.index[frames["half"] == h][0])
            anchors.append(min(first + params.team_anchor_offset, n - 1))
    if not anchors:
        anchors = [min(params.team_anchor_offset, n - 1)]

    votes = np.zeros((len(slots), 2), dtype=np.int64)
    gate2 = np.float32(params.team_gate_m ** 2)

    for anchor in anchors:
        pts = xy_all[anchor]
        cl = _kmeans2(pts)
        valid = np.isfinite(pts).all(axis=1)
        if valid.sum() < 4:
            continue
        mean_x = [pts[valid & (cl == k), 0].mean() for k in (0, 1)]
        home_cluster = int(np.argmin(mean_x))  # home broni lewej bramki
        anchor_lab = np.full(len(slots), -1, dtype=np.int8)
        anchor_lab[valid] = np.where(cl[valid] == home_cluster, 1, 0)
        votes[np.arange(len(slots)), np.clip(anchor_lab, 0, 1)] += (anchor_lab >= 0)

        for direction in (+1, -1):
            prev_lab = anchor_lab.copy()          # etykiety klatki anchor
            prev_xy = pts.copy()
            idx = anchor + direction
            while 0 <= idx < n:
                cur_xy = xy_all[idx]
                cur_valid = np.isfinite(cur_xy).all(axis=1)
                prev_ok = np.isfinite(prev_xy).all(axis=1) & (prev_lab >= 0)
                lab = prev_lab.copy()
                if cur_valid.any() and prev_ok.any():
                    # (C, P) odległości kwadratowe
                    diff = cur_xy[cur_valid][:, None, :] - prev_xy[prev_ok][None, :, :]
                    d2 = (diff * diff).sum(axis=2)
                    nearest = d2.argmin(axis=1)
                    mind = d2[np.arange(len(nearest)), nearest]
                    src_idx = np.flatnonzero(prev_ok)[nearest]
                    take = mind <= gate2
                    cols = np.flatnonzero(cur_valid)
                    lab[cols[take]] = prev_lab[src_idx[take]]
                votes[np.arange(len(slots)), np.clip(lab, 0, 1)] += (lab >= 0)
                prev_lab, prev_xy = lab, cur_xy
                idx += direction

    result: dict[str, str] = {}
    for i, s in enumerate(slots):
        h, a = int(votes[i, 1]), int(votes[i, 0])
        if h == a == 0:
            result[s] = "unknown"
        else:
            result[s] = "home" if h >= a else "away"
    return result
