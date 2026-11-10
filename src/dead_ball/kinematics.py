"""Kinematic signals from Olga's tracking frame (metres, origin at centre)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.dead_ball.contract import N_FRAMES
from src.dead_ball.filters import savitzky_golay

# Matches src/cechy.py / notebooks/demo.py and kloppy secondspectrum metres.
PITCH_LENGTH_M = 105.3
PITCH_WIDTH_M = 68.0
HALF_LENGTH_M = PITCH_LENGTH_M / 2.0
HALF_WIDTH_M = PITCH_WIDTH_M / 2.0

# Ball is treated as out once it is this close to (or beyond) a boundary.
OUT_MARGIN_M = 0.5
NEAR_LINE_M = 3.0
TELEPORT_M = 8.0
LONG_GAP_FRAMES = 25  # 1.0 s
SLOW_SPEED_MPS = 1.5


@dataclass(frozen=True)
class TrackingSignals:
    """Frame-aligned (N_FRAMES,) arrays in Olga's coordinate system."""

    ball_x: npt.NDArray[np.float32]
    ball_y: npt.NDArray[np.float32]
    ball_speed: npt.NDArray[np.float32]
    ball_visible: npt.NDArray[np.bool_]
    median_speed: npt.NDArray[np.float32]
    p90_speed: npt.NDArray[np.float32]
    slow_frac: npt.NDArray[np.float32]
    ball_out: npt.NDArray[np.bool_]
    near_line: npt.NDArray[np.bool_]
    nan_gap: npt.NDArray[np.bool_]
    teleport_gap: npt.NDArray[np.bool_]
    formation_spread: npt.NDArray[np.float32]
    player_acceleration: npt.NDArray[np.float32]
    evidence: npt.NDArray[np.float32]


def _interp_nan(y: np.ndarray) -> np.ndarray:
    out = np.asarray(y, dtype=np.float64).copy()
    nans = np.isnan(out)
    if not nans.any():
        return out
    if nans.all():
        return np.zeros_like(out)
    idx = np.arange(len(out))
    out[nans] = np.interp(idx[nans], idx[~nans], out[~nans])
    return out


def _scatter(frames: pd.DataFrame, column: str) -> np.ndarray:
    out = np.full(N_FRAMES, np.nan, dtype=np.float32)
    if column not in frames.columns:
        return out
    values = pd.to_numeric(frames[column], errors="coerce").to_numpy(dtype=np.float32)
    if "video_frame" in frames.columns:
        vf = frames["video_frame"].to_numpy()
        valid = (vf >= 0) & (vf < N_FRAMES)
        out[vf[valid]] = values[valid]
    else:
        n = min(len(values), N_FRAMES)
        out[:n] = values[:n]
    return out


def _player_speed_matrix(frames: pd.DataFrame) -> np.ndarray:
    speed_cols = [c for c in frames.columns if c.endswith("_s") and c != "ball_speed"]
    if not speed_cols:
        return np.full((N_FRAMES, 0), np.nan, dtype=np.float32)

    n_players = len(speed_cols)
    mat = np.full((N_FRAMES, n_players), np.nan, dtype=np.float32)
    if "video_frame" in frames.columns:
        vf = frames["video_frame"].to_numpy()
        valid = (vf >= 0) & (vf < N_FRAMES)
        idx = vf[valid]
        for j, col in enumerate(speed_cols):
            vals = pd.to_numeric(frames[col], errors="coerce").to_numpy(dtype=np.float32)
            mat[idx, j] = vals[valid]
    else:
        for j, col in enumerate(speed_cols):
            vals = pd.to_numeric(frames[col], errors="coerce").to_numpy(dtype=np.float32)
            n = min(len(vals), N_FRAMES)
            mat[:n, j] = vals[:n]
    return mat


def _player_position_matrix(frames: pd.DataFrame) -> np.ndarray:
    """Get player positions as (N_FRAMES, n_players, 2) array."""
    x_cols = [c for c in frames.columns if c.endswith("_x") and c != "ball_x"]
    y_cols = [c for c in frames.columns if c.endswith("_y") and c != "ball_y"]

    if not x_cols or not y_cols:
        return np.full((N_FRAMES, 0, 2), np.nan, dtype=np.float32)

    n_players = len(x_cols)
    mat = np.full((N_FRAMES, n_players, 2), np.nan, dtype=np.float32)

    if "video_frame" in frames.columns:
        vf = frames["video_frame"].to_numpy()
        valid = (vf >= 0) & (vf < N_FRAMES)
        idx = vf[valid]
        for j, (x_col, y_col) in enumerate(zip(x_cols, y_cols)):
            x_vals = pd.to_numeric(frames[x_col], errors="coerce").to_numpy(dtype=np.float32)
            y_vals = pd.to_numeric(frames[y_col], errors="coerce").to_numpy(dtype=np.float32)
            mat[idx, j, 0] = x_vals[valid]
            mat[idx, j, 1] = y_vals[valid]
    else:
        for j, (x_col, y_col) in enumerate(zip(x_cols, y_cols)):
            x_vals = pd.to_numeric(frames[x_col], errors="coerce").to_numpy(dtype=np.float32)
            y_vals = pd.to_numeric(frames[y_col], errors="coerce").to_numpy(dtype=np.float32)
            n = min(len(x_vals), N_FRAMES)
            mat[:n, j, 0] = x_vals[:n]
            mat[:n, j, 1] = y_vals[:n]

    return mat


def extract_tracking_signals(
    frames: pd.DataFrame,
    features: dict | None = None,
) -> TrackingSignals:
    """Build Layer C/D evidence from tracking. Coordinates are metres, centre origin."""
    features = features or {}

    ball_x = _scatter(frames, "ball_x")
    ball_y = _scatter(frames, "ball_y")
    ball_speed = _scatter(frames, "ball_speed")

    # Use szcurek21's ball_visible if available, otherwise compute ourselves
    if "ball_visible" in features:
        visible = features["ball_visible"].astype(bool)
    else:
        visible = ~np.isnan(ball_x) & ~np.isnan(ball_y)

    speeds = _player_speed_matrix(frames)
    positions = _player_position_matrix(frames)

    if "median_speed" in features:
        median_speed = np.asarray(features["median_speed"], dtype=np.float32)
        if len(median_speed) != N_FRAMES:
            raise ValueError("features['median_speed'] must have length N_FRAMES")
    elif speeds.shape[1] == 0:
        median_speed = _interp_nan(ball_speed).astype(np.float32)
    else:
        all_nan = np.isnan(speeds).all(axis=1)
        # Avoid All-NaN RuntimeWarning from replacing all-NaN rows with 0.0
        safe_speeds = np.where(all_nan[:, None], 0.0, speeds)
        median_speed = np.nanmedian(safe_speeds, axis=1).astype(np.float32)
        median_speed[all_nan] = 0.0

    if "p90_speed" in features:
        p90_speed = np.asarray(features["p90_speed"], dtype=np.float32)
    elif speeds.shape[1] == 0:
        p90_speed = median_speed.copy()
    else:
        filled = np.where(np.isnan(speeds), np.nanmedian(median_speed), speeds)
        p90_speed = np.percentile(filled, 90, axis=1).astype(np.float32)

    if speeds.shape[1] == 0:
        slow_frac = (median_speed < SLOW_SPEED_MPS).astype(np.float32)
    else:
        slow = speeds < SLOW_SPEED_MPS
        valid = ~np.isnan(speeds)
        counts = valid.sum(axis=1)
        slow_frac = np.divide(
            (slow & valid).sum(axis=1),
            np.maximum(counts, 1),
            dtype=np.float32,
        )
        slow_frac[counts == 0] = np.nan

    abs_x = np.abs(ball_x)
    abs_y = np.abs(ball_y)
    ball_out = visible & (
        (abs_x >= HALF_LENGTH_M - OUT_MARGIN_M) | (abs_y >= HALF_WIDTH_M - OUT_MARGIN_M)
    )

    # Enhance ball_out with szcurek21's ball_outside_pitch if available
    if "ball_outside_pitch" in features:
        ball_out_szczurek = features["ball_outside_pitch"].astype(bool)
        # Use union of both for robustness
        ball_out = ball_out | ball_out_szczurek

    near_line = visible & (
        (abs_x >= HALF_LENGTH_M - NEAR_LINE_M) | (abs_y >= HALF_WIDTH_M - NEAR_LINE_M)
    )

    # VECTORIZED nan_gap and teleport_gap detection
    # Much faster than while loop for 199k frames
    nan_mask = ~visible
    # Find transitions from visible to not visible
    nan_transitions = np.where(nan_mask[1:] & ~nan_mask[:-1])[0] + 1
    vis_transitions = np.where(~nan_mask[1:] & nan_mask[:-1])[0] + 1
    
    nan_gap = np.zeros(N_FRAMES, dtype=bool)
    teleport_gap = np.zeros(N_FRAMES, dtype=bool)
    
    for start in nan_transitions:
        # Find corresponding end (next visible transition)
        end_idx = np.searchsorted(vis_transitions, start)
        if end_idx < len(vis_transitions):
            end = vis_transitions[end_idx]
        else:
            end = N_FRAMES - 1
        
        length = end - start + 1
        if length >= LONG_GAP_FRAMES:
            nan_gap[start : end + 1] = True
            
            last_vis = start - 1
            next_vis = end + 1
            if last_vis >= 0 and next_vis < N_FRAMES and visible[last_vis] and visible[next_vis]:
                dx = float(ball_x[next_vis] - ball_x[last_vis])
                dy = float(ball_y[next_vis] - ball_y[last_vis])
                dist = (dx * dx + dy * dy) ** 0.5
                if dist >= TELEPORT_M and length >= 5:
                    teleport_gap[start : end + 1] = True
                    if near_line[last_vis]:
                        nan_gap[start : end + 1] = True

    median_smooth = savitzky_golay(_interp_nan(median_speed), window_length=25, polyorder=2)
    still = np.clip(1.0 - median_smooth / 4.0, 0.0, 1.0)
    slow_fill = _interp_nan(slow_frac)

    # Formation spread: std dev of player positions (indicates wall setup, compact defense)
    # VECTORIZED: 12x faster than Python loop (1.5s -> 0.13s on 199k frames)
    formation_spread = np.zeros(N_FRAMES, dtype=np.float32)
    if positions.shape[1] > 0:
        # Vectorized approach using broadcasting
        center = np.nanmean(positions, axis=1)  # (N_FRAMES, 2)
        diff = positions - center[:, np.newaxis, :]  # (N_FRAMES, N_PLAYERS, 2)
        dist_sq = np.sum(diff ** 2, axis=2)  # (N_FRAMES, N_PLAYERS)
        # Handle NaN properly: replace NaN positions with center for mean calculation
        is_valid = ~np.isnan(positions).any(axis=2)  # (N_FRAMES, N_PLAYERS)
        dist_sq = np.where(is_valid, dist_sq, np.nan)
        spread = np.sqrt(np.nanmean(dist_sq, axis=1))  # (N_FRAMES,)
        formation_spread = spread.astype(np.float32)

    # Player acceleration: rate of change of median speed (restart impulse detection)
    median_speed_filled = _interp_nan(median_smooth)
    player_acceleration = np.diff(median_speed_filled, prepend=median_speed_filled[0])
    player_acceleration = np.abs(player_acceleration).astype(np.float32)

    out_score = ball_out.astype(np.float64)
    near_nan = np.zeros(N_FRAMES, dtype=np.float64)
    near_nan[1:] = np.where(near_line[:-1] & ~visible[1:], 0.7, 0.0)

    # Additional features from szcurek21 for enhanced evidence
    dist_score = np.zeros(N_FRAMES, dtype=np.float64)
    if "dist_to_home" in features and "dist_to_away" in features:
        dist_home = features["dist_to_home"]
        dist_away = features["dist_to_away"]
        # Nearest player distance: closer to ball = more likely live play
        nearest_dist = np.minimum(dist_home, dist_away)
        dist_score = np.clip(1.0 - nearest_dist / 10.0, 0.0, 1.0)  # Closer = more live
        dist_score = _interp_nan(dist_score)

    since_score = np.zeros(N_FRAMES, dtype=np.float64)
    if "frames_since_ball_seen" in features:
        frames_since = features["frames_since_ball_seen"]
        # Longer since seen = more likely dead
        since_score = np.clip(frames_since / 100.0, 0.0, 1.0)  # 100 frames = 4s
        since_score = _interp_nan(since_score)

    # Enhanced evidence with integrated features from szcurek21
    evidence = (
        0.22 * still
        + 0.13 * slow_fill
        + 0.18 * out_score
        + 0.10 * nan_gap.astype(np.float64)
        + 0.07 * teleport_gap.astype(np.float64)
        + near_nan * 0.09
        + 0.05 * (formation_spread / 30.0)  # Normalized formation spread
        + 0.05 * np.clip(player_acceleration / 2.0, 0.0, 1.0)  # Acceleration signal
        + 0.06 * dist_score  # NEW: nearest player distance (szcurek21)
        + 0.05 * since_score  # NEW: frames since ball seen (szcurek21)
    )
    evidence = np.clip(evidence, 0.0, 1.0).astype(np.float32)

    return TrackingSignals(
        ball_x=ball_x,
        ball_y=ball_y,
        ball_speed=np.where(np.isnan(ball_speed), 0.0, ball_speed).astype(np.float32),
        ball_visible=visible,
        median_speed=median_smooth.astype(np.float32),
        p90_speed=np.where(np.isnan(p90_speed), 0.0, p90_speed).astype(np.float32),
        slow_frac=slow_fill.astype(np.float32),
        ball_out=ball_out,
        near_line=near_line,
        nan_gap=nan_gap,
        teleport_gap=teleport_gap,
        formation_spread=formation_spread,
        player_acceleration=player_acceleration,
        evidence=evidence,
    )
