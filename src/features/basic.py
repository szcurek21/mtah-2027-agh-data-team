from __future__ import annotations

import numpy as np
import pandas as pd

SMOOTH_SEC = 0.4
MAX_PLAYER_SPEED = 12.0
MAX_PLAYER_ACC = 10.0
MAX_BALL_SPEED = 40.0
BALL_SPIKE_WIN = 7
BALL_SPIKE_M = 2.0


def slot_ids(frames: pd.DataFrame) -> list[str]:
    return [c[:-2] for c in frames.columns if c.endswith("_x") and c != "ball_x"]


def _fps(frames: pd.DataFrame) -> float:
    dt = np.diff(frames["t"].to_numpy())
    dt = dt[dt > 0]
    return float(round(1.0 / np.median(dt)))


def _segments(frames: pd.DataFrame) -> np.ndarray:
    vf = frames["video_frame"].to_numpy()
    half = frames["half"].fillna(-1).to_numpy()
    new = np.r_[True, (np.diff(vf) != 1) | (np.diff(half) != 0)]
    return np.cumsum(new)


def _smooth(df: pd.DataFrame, seg: np.ndarray, window: int) -> pd.DataFrame:
    return (
        df.groupby(seg, sort=False)
        .rolling(window, center=True, min_periods=max(1, window // 2))
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )


def _derivative(df: pd.DataFrame, seg: np.ndarray, fps: float) -> pd.DataFrame:
    g = df.groupby(seg, sort=False)
    return (g.shift(-1) - g.shift(1)) * (fps / 2.0)


def clean_ball(frames: pd.DataFrame) -> pd.DataFrame:
    out = frames.copy()
    seg = _segments(frames)
    ball = frames[["ball_x", "ball_y"]].astype("float64")
    med = (
        ball.groupby(seg, sort=False)
        .rolling(BALL_SPIKE_WIN, center=True, min_periods=3)
        .median()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    dev = np.hypot(ball["ball_x"] - med["ball_x"], ball["ball_y"] - med["ball_y"])
    spike = (dev > BALL_SPIKE_M).to_numpy()
    out.loc[spike, ["ball_x", "ball_y"]] = np.nan
    out.attrs["ball_spikes_removed"] = int(spike.sum())
    return out


def kinematics(frames: pd.DataFrame) -> pd.DataFrame:
    fps = _fps(frames)
    seg = _segments(frames)
    win = max(1, int(round(SMOOTH_SEC * fps)))
    slots = slot_ids(frames)

    out = {"video_frame": frames["video_frame"].to_numpy()}

    xs = frames[[f"{s}_x" for s in slots]].astype("float64")
    ys = frames[[f"{s}_y" for s in slots]].astype("float64")
    xs.columns = ys.columns = slots
    xs_s, ys_s = _smooth(xs, seg, win), _smooth(ys, seg, win)
    vx, vy = _derivative(xs_s, seg, fps), _derivative(ys_s, seg, fps)
    speed = np.hypot(vx, vy)
    bad = speed > MAX_PLAYER_SPEED
    vx, vy, speed = vx.mask(bad), vy.mask(bad), speed.mask(bad)
    acc = _derivative(_smooth(speed, seg, win), seg, fps)
    acc = acc.mask(acc.abs() > MAX_PLAYER_ACC)

    for s in slots:
        out[f"{s}_vx"] = vx[s].to_numpy(np.float32)
        out[f"{s}_vy"] = vy[s].to_numpy(np.float32)
        out[f"{s}_speed"] = speed[s].to_numpy(np.float32)
        out[f"{s}_acc"] = acc[s].to_numpy(np.float32)

    ball = frames[["ball_x", "ball_y"]].astype("float64")
    ball_s = _smooth(ball, seg, max(1, win // 2))
    bv = _derivative(ball_s, seg, fps)
    bspeed = np.hypot(bv["ball_x"], bv["ball_y"])
    bad = bspeed > MAX_BALL_SPEED
    out["ball_vx"] = bv["ball_x"].mask(bad).to_numpy(np.float32)
    out["ball_vy"] = bv["ball_y"].mask(bad).to_numpy(np.float32)
    out["ball_speed_calc"] = bspeed.mask(bad).to_numpy(np.float32)
    out["ball_dir"] = np.degrees(np.arctan2(out["ball_vy"], out["ball_vx"])).astype(np.float32)

    return pd.DataFrame(out, index=frames.index)


def _nanmin_arg(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    filled = np.where(np.isnan(d), np.inf, d)
    idx = filled.argmin(axis=1)
    val = filled[np.arange(len(d)), idx]
    empty = np.isinf(val)
    return np.where(empty, -1, idx), np.where(empty, np.nan, val)


def _team_shape(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    with np.errstate(all="ignore"), _quiet():
        cx, cy = np.nanmean(x, axis=1), np.nanmean(y, axis=1)
        return {
            "cx": cx,
            "cy": cy,
            "len": np.nanmax(x, axis=1) - np.nanmin(x, axis=1),
            "wid": np.nanmax(y, axis=1) - np.nanmin(y, axis=1),
            "spread": np.nanmean(np.hypot(x - cx[:, None], y - cy[:, None]), axis=1),
            "n": (~np.isnan(x)).sum(axis=1),
        }


class _quiet:
    def __enter__(self):
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        self._w.__exit__(*exc)


def frames_since_ball_seen(visible: np.ndarray, seg: np.ndarray) -> np.ndarray:
    idx = np.arange(len(visible), dtype=np.float64)
    last = pd.Series(np.where(visible, idx, np.nan)).groupby(seg).ffill().to_numpy()
    return (idx - last).astype(np.float32)


def frame_features(frames: pd.DataFrame, teams: dict[str, str]) -> pd.DataFrame:
    slots = [s for s in slot_ids(frames) if s in teams]
    x = frames[[f"{s}_x" for s in slots]].to_numpy(np.float64)
    y = frames[[f"{s}_y" for s in slots]].to_numpy(np.float64)
    bx = frames["ball_x"].to_numpy(np.float64)[:, None]
    by = frames["ball_y"].to_numpy(np.float64)[:, None]
    team = np.array([teams[s] for s in slots])
    home, away = team == "home", team == "away"

    dist = np.hypot(x - bx, y - by)
    idx, dmin = _nanmin_arg(dist)
    slot_arr = np.array(slots + [None], dtype=object)
    team_arr = np.array([str(t) for t in team] + [None], dtype=object)

    visible = ~np.isnan(bx[:, 0])
    out = {
        "video_frame": frames["video_frame"].to_numpy(),
        "ball_visible": visible,
        "frames_since_ball_seen": frames_since_ball_seen(visible, _segments(frames)),
        "nearest_slot": slot_arr[idx],
        "nearest_team": team_arr[idx],
        "nearest_dist": dmin.astype(np.float32),
        "nearest_home_dist": _nanmin_arg(dist[:, home])[1].astype(np.float32),
        "nearest_away_dist": _nanmin_arg(dist[:, away])[1].astype(np.float32),
    }
    for name, mask in (("home", home), ("away", away)):
        for k, v in _team_shape(x[:, mask], y[:, mask]).items():
            out[f"{name}_{k}"] = v if k == "n" else v.astype(np.float32)

    return pd.DataFrame(out, index=frames.index)


def build_features(
    frames: pd.DataFrame, teams: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:

    clean = clean_ball(frames)
    kin, feat = kinematics(clean), frame_features(clean, teams)
    feat.attrs["ball_spikes_removed"] = clean.attrs["ball_spikes_removed"]
    return kin, feat


if __name__ == "__main__":
    import sys

    from src.decoder.load import load_frames, team_of

    match = sys.argv[1] if len(sys.argv) > 1 else "BAR-ATH"
    frames = load_frames(match)
    kin, feat = build_features(frames, team_of(match))
    speeds = kin[[c for c in kin.columns if c.endswith("_speed")]].to_numpy().ravel()
    speeds = speeds[~np.isnan(speeds)]
    print(f"Klatki: {len(feat)}, kolumny kinematyki: {kin.shape[1]}, cech klatki: {feat.shape[1]}")
    print(
        f"Piłka widoczna: {feat['ball_visible'].mean():.1%} "
        f"(usunięte skoki piłki: {feat.attrs['ball_spikes_removed']} klatek)"
    )
    bs = kin["ball_speed_calc"].dropna()
    print(
        f"Prędkość piłki po czyszczeniu: mediana {bs.median():.2f} m/s, 99.9 pct {bs.quantile(0.999):.1f} m/s"
    )
    print(
        f"Prędkość zawodników: mediana {np.median(speeds):.2f} m/s, 99. percentyl {np.percentile(speeds, 99):.2f} m/s"
    )
    print(
        f"Najbliższy do piłki (gdy piłka widoczna): {feat['nearest_dist'].median():.2f} m (mediana)"
    )
