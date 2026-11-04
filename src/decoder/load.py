from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from kloppy import metrica
from kloppy.domain import TrackingDataset

DATA_DIR = Path(
    os.environ.get("MTAH_DATA_DIR", Path.home() / "more-than-a-hack-2027" / "data")
)
FPS = 25

HALF_BLOCKS: dict[str, list[tuple[int, int, int]]] = {
    "BAR-ATH": [(1, 11185, 90651), (2, 110808, 190197)],
}



def find_files(match: str, view: str, data_dir: Path = DATA_DIR) -> tuple[Path, Path]:
    """Zwraca (meta_xml, raw_txt) dla meczu i kamery ("tactical" / "broadcast")."""
    raw = list(data_dir.rglob(f"{match}_{view}_atd.txt"))
    meta = list(data_dir.rglob(f"{match}_{view}_atd.xml"))
    if len(raw) != 1 or len(meta) != 1:
        raise FileNotFoundError(
            f"Szukałam {match}_{view}_atd.txt/.xml w {data_dir}; "
            f"znalazłam txt={raw}, xml={meta}"
        )
    return meta[0], raw[0]


def load_dataset(
    match: str, view: str = "tactical", data_dir: Path = DATA_DIR, limit: int | None = None
) -> TrackingDataset:
    """Surowy dataset kloppy w metrach, (0,0) = środek boiska. Bez obracania połów."""
    meta, raw = find_files(match, view, data_dir)
    return metrica.load_tracking_epts(
        meta_data=meta, raw_data=raw, limit=limit, coordinates="secondspectrum"
    )


def team_of(match: str, view: str = "tactical", data_dir: Path = DATA_DIR) -> dict[str, str]:
    """{slot_id: "home" / "away"} na podstawie metadanych XML."""
    ds = load_dataset(match, view, data_dir, limit=1)
    return {
        str(p.player_id): ("home" if t.ground.name == "HOME" else "away")
        for t in ds.metadata.teams
        for p in t.players
    }



def _slot_ids(df: pd.DataFrame) -> list[str]:
    return [c[:-2] for c in df.columns if c.endswith("_x") and c != "ball_x"]


def _clean(wide: pd.DataFrame) -> pd.DataFrame:
    """Ujednolica nazwy i typy, dodaje video_frame i t."""
    slots = _slot_ids(wide)
    keep = ["frame_id", "ball_x", "ball_y", "ball_z", "ball_speed"]
    keep += [f"{s}_{k}" for s in slots for k in ("x", "y", "s")]
    video_frame = (wide["frame_id"].to_numpy() - 1).astype("int64")
    cols = {
        "video_frame": video_frame,
        "t": video_frame / FPS,  # sekundy od początku wideo
        "half": np.full(len(wide), np.nan),
        "graded": np.zeros(len(wide), dtype=bool),
    }
    for c in keep[1:]:
        if c in wide.columns:
            cols[c] = pd.to_numeric(wide[c], errors="coerce").to_numpy("float32")
    return pd.DataFrame(cols)


def _add_halves(frames: pd.DataFrame, blocks: list[tuple[int, int, int]]) -> None:
    for half, first, last in blocks:
        frames.loc[frames["video_frame"].between(first, last), "half"] = half
    frames["graded"] = frames["half"].notna()


HOME_TAG = {"BAR-ATH": "Barcelona"}  # nazwa gospodarzy w smart taggingu
ATTACK_CODES = {"BALL IN FINAL THIRD", "BALL IN THE BOX"}


def _home_attacks_right_by_tags(
    frames: pd.DataFrame, match: str, data_dir: Path
) -> dict[int, float] | None:
    
    from kloppy import sportscode

    paths = list(data_dir.rglob(f"{match}_tactical_smart_tagging.xml"))
    if len(paths) != 1 or match not in HOME_TAG:
        return None
    tags = sportscode.load(paths[0]).to_df()
    tags = tags[
        tags["code"].str.upper().isin(ATTACK_CODES)
        & (tags["Team"].astype(str).str.lower() == HOME_TAG[match].lower())
    ]
    ball_x = frames.set_index("video_frame")["ball_x"]
    result = {}
    for half in sorted(frames["half"].dropna().unique()):
        span = frames.loc[frames["half"] == half, "video_frame"]
        xs = []
        for _, r in tags.iterrows():
            a = round(r["timestamp"].total_seconds() * FPS)
            b = round(r["end_timestamp"].total_seconds() * FPS)
            if span.min() <= a <= span.max():
                xs.append(ball_x.loc[a:b].dropna().to_numpy())
        if xs and sum(map(len, xs)):
            result[int(half)] = float(np.median(np.concatenate(xs)))
    return result


def _home_defends_left_by_keepers(frames: pd.DataFrame, teams: dict[str, str]) -> dict[int, float]:
    slots = [s for s in teams if f"{s}_x" in frames]
    is_home = np.array([teams[s] == "home" for s in slots])
    result = {}
    for half in sorted(frames["half"].dropna().unique()):
        px = frames.loc[frames["half"] == half, [f"{s}_x" for s in slots]].to_numpy()
        px = px[~np.isnan(px).all(axis=1)]
        leftmost = np.argmin(np.where(np.isnan(px), np.inf, px), axis=1)
        result[int(half)] = float(is_home[leftmost].mean())
    return result


def _orient(
    frames: pd.DataFrame, match: str, teams: dict[str, str], data_dir: Path
) -> dict[int, dict]:
    tag_votes = _home_attacks_right_by_tags(frames, match, data_dir) or {}
    keeper_votes = _home_defends_left_by_keepers(frames, teams)
    xs = [c for c in frames.columns if c.endswith("_x")]
    ys = [c for c in frames.columns if c.endswith("_y")]
    info = {}
    for half in sorted(frames["half"].dropna().unique()):
        h = int(half)
        if h in tag_votes:
            flip, method, score = tag_votes[h] < 0, "piłka+smart tagging", round(tag_votes[h], 1)
        else:
            flip, method, score = keeper_votes[h] < 0.5, "bramkarze", round(keeper_votes[h], 2)
        if flip:
            in_half = frames["half"] == half
            frames.loc[in_half, xs] *= -1
            frames.loc[in_half, ys] *= -1
        info[h] = {"flipped": bool(flip), "method": method, "score": score}
    return info


def load_frames(
    match: str,
    view: str = "tactical",
    data_dir: Path = DATA_DIR,
    use_cache: bool = True,
) -> pd.DataFrame:
    cache = data_dir / "cache" / f"{match}_{view}_frames.pkl"
    if use_cache and cache.exists():
        return pd.read_pickle(cache)

    frames = _clean(load_dataset(match, view, data_dir).to_df())
    if view == "tactical" and match in HALF_BLOCKS:
        _add_halves(frames, HALF_BLOCKS[match])
        frames.attrs["flipped"] = _orient(
            frames, match, team_of(match, view, data_dir), data_dir
        )
    else:
        frames.attrs["flipped"] = {}

    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        frames.to_pickle(cache)
    return frames


def to_long(frames: pd.DataFrame, teams: dict[str, str]) -> pd.DataFrame:
    
    base = ["video_frame", "t", "half", "graded"]
    parts = [
        frames[base + ["ball_x", "ball_y", "ball_speed"]]
        .set_axis(base + ["x", "y", "speed"], axis=1)
        .assign(object_id="ball", team="ball")
    ]
    for slot in _slot_ids(frames):
        parts.append(
            frames[base + [f"{slot}_x", f"{slot}_y", f"{slot}_s"]]
            .set_axis(base + ["x", "y", "speed"], axis=1)
            .assign(object_id=slot, team=teams.get(slot, "unknown"))
        )
    long = pd.concat(parts, ignore_index=True).dropna(subset=["x", "y"])
    return long.sort_values(["video_frame", "team", "object_id"], ignore_index=True)



def describe(match: str, view: str) -> None:
    frames = load_frames(match, view)
    teams = team_of(match, view)
    print(f"{match} / {view}: {len(frames)} klatek, {len(_slot_ids(frames))} slotów")
    print("Slotów home/away:", sum(v == "home" for v in teams.values()),
          "/", sum(v == "away" for v in teams.values()))
    print("Klatek ocenianych:", int(frames["graded"].sum()))
    print("Obrócone połowy:", frames.attrs.get("flipped"))
    print(f"Piłka widoczna: {frames['ball_x'].notna().mean():.1%} wszystkich klatek, "
          f"{frames.loc[frames['graded'], 'ball_x'].notna().mean():.1%} ocenianych")
    print("Kolumny:")
    for c in frames.columns[:12]:
        print(f"  {c}: {frames[c].dtype}")
    print(f"  ... i dalej {{slot}}_x / _y / _s dla każdego slotu")


if __name__ == "__main__":
    describe(
        sys.argv[1] if len(sys.argv) > 1 else "BAR-ATH",
        sys.argv[2] if len(sys.argv) > 2 else "tactical",
    )