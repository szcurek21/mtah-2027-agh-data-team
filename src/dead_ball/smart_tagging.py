"""Smart Tagging loader and candidate window extractor for Dead Ball detection.

Extracts set pieces (throw-in, corner, goal kick, free kick, kick-off) and goals
from Metrica Sports Smart Tagging XML, converting them into candidate dead-ball
windows with configurable margins.

Frame conversion matches Olga's loader: round(timestamp_seconds * 25).
The kloppy SportsCode path is preferred so both modules share one clock;
ElementTree is the fallback and is used to measure parser drift.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

from src.dead_ball.contract import (
    N_FRAMES,
    DeadSegment,
    DeadSegmentKind,
)

FPS: int = 25

# Mapping from Smart Tagging 'Type' or 'code' to our DeadSegmentKind
TAG_TYPE_MAP: dict[str, DeadSegmentKind] = {
    "throw-in": "throw_in",
    "corner kick": "corner",
    "goal kick": "goal_kick",
    "free kick": "free_kick",
    "kick off (start)": "kick_off",
    "kick off (after goal)": "kick_off",
    "goals": "goal",
}

_MISSING_TEAM = frozenset({"n/a", "na", "none", "", "nan"})


def seconds_to_frame(seconds: float) -> int:
    """Official video-frame index used by Olga's loader and this module."""
    return int(max(0, min(N_FRAMES - 1, round(float(seconds) * FPS))))


def clean_team(team: str | None) -> str | None:
    if team is None:
        return None
    text = str(team).strip()
    if text.lower() in _MISSING_TEAM:
        return None
    return text


def _frame_from_timestamp(ts: Any) -> tuple[float, int]:
    if hasattr(ts, "total_seconds"):
        seconds = float(ts.total_seconds())
    else:
        seconds = float(ts)
    return seconds, seconds_to_frame(seconds)


def parse_smart_tagging_xml(xml_path: str | Path) -> list[dict]:
    """SportsCode XML parser (ElementTree). Same frame formula as kloppy path."""
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"Smart tagging XML not found: {path}")

    tree = ET.parse(path)
    root = tree.getroot()

    instances = []
    for inst in root.iter("instance"):
        start_el = inst.find("start")
        end_el = inst.find("end")
        code_el = inst.find("code")

        if start_el is None or end_el is None or code_el is None:
            continue

        try:
            start_sec = float(start_el.text or 0.0)
            end_sec = float(end_el.text or 0.0)
        except ValueError:
            continue

        code = (code_el.text or "").strip()
        team = None
        event_type = None

        for label in inst.iter("label"):
            group_el = label.find("group")
            text_el = label.find("text")
            if group_el is None or text_el is None:
                continue

            group = (group_el.text or "").strip().lower()
            text = (text_el.text or "").strip()

            if group == "team":
                team = text
            elif group == "type":
                event_type = text

        instances.append(
            {
                "code": code,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_frame": seconds_to_frame(start_sec),
                "end_frame": seconds_to_frame(end_sec),
                "team": clean_team(team),
                "type": event_type,
                "source": "elementtree",
            }
        )

    return instances


def parse_smart_tagging_kloppy(xml_path: str | Path) -> list[dict]:
    """Same events via kloppy.sportscode — the clock Olga uses for orientation."""
    from kloppy import sportscode

    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"Smart tagging XML not found: {path}")

    with open(path, "rb") as f:
        tags = sportscode.load(f).to_df()
    instances: list[dict] = []

    team_col = next((c for c in tags.columns if c.lower() == "team"), None)
    type_col = next((c for c in tags.columns if c.lower() == "type"), None)
    start_col = "timestamp" if "timestamp" in tags.columns else "start"
    end_col = "end_timestamp" if "end_timestamp" in tags.columns else "end"

    for _, row in tags.iterrows():
        start_sec, start_frame = _frame_from_timestamp(row[start_col])
        end_sec, end_frame = _frame_from_timestamp(row[end_col])
        code = str(row["code"]).strip() if "code" in tags.columns else ""
        team = clean_team(row[team_col] if team_col else None)
        event_type = row[type_col] if type_col else None
        if event_type is not None and (event_type != event_type):  # NaN
            event_type = None
        elif event_type is not None:
            event_type = str(event_type).strip()

        instances.append(
            {
                "code": code,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "team": team,
                "type": event_type,
                "source": "kloppy",
            }
        )

    return instances


def parse_smart_tagging(xml_path: str | Path, prefer_kloppy: bool = True) -> list[dict]:
    """Load tags, preferring kloppy so dead-ball and data-loader share a clock."""
    if prefer_kloppy:
        try:
            rows = parse_smart_tagging_kloppy(xml_path)
            if rows:
                return rows
        except Exception:
            pass
    return parse_smart_tagging_xml(xml_path)


def compare_tag_clocks(xml_path: str | Path) -> dict[str, float | int | str]:
    """Measure ElementTree vs kloppy frame disagreement for the same XML."""
    et_rows = parse_smart_tagging_xml(xml_path)
    try:
        kl_rows = parse_smart_tagging_kloppy(xml_path)
    except Exception as exc:
        return {
            "n_elementtree": len(et_rows),
            "n_kloppy": 0,
            "n_matched": 0,
            "max_abs_frame_delta": -1,
            "mean_abs_frame_delta": -1,
            "median_frame_delta": -1,
            "status": f"kloppy_failed: {exc}",
        }

    et_index = {(r["code"], r["start_frame"]): r for r in et_rows}
    deltas: list[int] = []
    matched = 0
    for row in kl_rows:
        key = (row["code"], row["start_frame"])
        # Match on code + nearest start if exact frame key misses
        if key in et_index:
            matched += 1
            deltas.append(0)
            continue
        same_code = [r for r in et_rows if r["code"] == row["code"]]
        if not same_code:
            continue
        nearest = min(same_code, key=lambda r: abs(r["start_frame"] - row["start_frame"]))
        if abs(nearest["start_frame"] - row["start_frame"]) <= 5:
            matched += 1
            deltas.append(abs(nearest["start_frame"] - row["start_frame"]))

    arr = np.array(deltas, dtype=np.float64) if deltas else np.array([0.0])
    return {
        "n_elementtree": len(et_rows),
        "n_kloppy": len(kl_rows),
        "n_matched": matched,
        "max_abs_frame_delta": int(arr.max()) if len(arr) else -1,
        "mean_abs_frame_delta": float(arr.mean()) if len(arr) else -1,
        "median_frame_delta": float(np.median(arr)) if len(arr) else -1,
        "status": "ok",
    }


def extract_dead_candidates_from_tags(
    instances: list[dict],
    delta_pre_frames: int = 25,
    delta_post_frames: int = 50,
) -> Tuple[np.ndarray, list[DeadSegment], np.ndarray]:
    """Tag events as *search windows*, not as dead-ball truth.

    Returns:
      - search_mask: expanded windows where Layer C may look for boundaries
      - core_segments: raw tag intervals (kind + team preserved)
      - core_mask: raw tag frames, used to protect short confirmed events
    """
    search_mask = np.zeros(N_FRAMES, dtype=bool)
    core_mask = np.zeros(N_FRAMES, dtype=bool)
    core_segments: List[DeadSegment] = []

    for inst in instances:
        code_upper = inst["code"].upper()
        event_type = (inst["type"] or "").lower()

        if code_upper not in ("SET PIECES", "GOALS") and event_type not in TAG_TYPE_MAP:
            continue

        kind: DeadSegmentKind = TAG_TYPE_MAP.get(event_type, "other")
        if code_upper == "GOALS":
            kind = "goal"

        raw_start = int(inst["start_frame"])
        raw_end = int(inst["end_frame"])
        if raw_end < raw_start:
            raw_start, raw_end = raw_end, raw_start

        core_start = max(0, raw_start)
        core_end = min(N_FRAMES - 1, raw_end)
        search_start = max(0, core_start - delta_pre_frames)
        search_end = min(N_FRAMES - 1, max(core_end + delta_post_frames, search_start))

        core_mask[core_start : core_end + 1] = True
        search_mask[search_start : search_end + 1] = True

        core_segments.append(
            DeadSegment(
                start=core_start,
                end=core_end,
                kind=kind,
                confidence=0.85,
                start_sigma=float(delta_pre_frames) / 2.0,
                end_sigma=float(delta_post_frames) / 2.0,
                team=clean_team(inst.get("team")),
            )
        )

    core_segments.sort(key=lambda seg: (seg.start, seg.end))
    return search_mask, core_segments, core_mask
