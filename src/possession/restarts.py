"""4.5–4.7: restart re-inicjalizacja, segmenty żywej gry i dekoder semi-Markov.

Restart = pierwsza klatka po `dead_mask` (wniosek: moduł martwej piłki
dostarcza `dead_mask` / `segments`; my tylko konsumujemy). Wykonawca jest
inferowany geometrycznie (§4.6): drużyna, której zawodnik jest średnio
najbliżej punktu wznowienia w pierwszych klatkach po restarcie.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FPS, PossessionParams


@dataclass(frozen=True)
class RestartEvent:
    """Pojedyncze wznowienie gry.

    - frame: pierwsza klatka żywej gry po przerwie
    - team: 'H'/'A' — inferowany wykonawca; None gdy brak danych
    - weight: waga zaufania (np. segments[i].confidence)
    - xy: punkt wznowienia w metrach (hint restart_xy albo pozycja piłki)
    """

    frame: int
    team: str | None
    weight: float
    xy: np.ndarray | None = None
    source: str = "dead_mask"


def _get(obj, name: str, default=None):
    """Dostęp do pola z dict albo obiektu (kontrakt dead_ball może być różny)."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def live_segments(dead_mask: np.ndarray) -> list[tuple[int, int]]:
    """Ciągłe odcinki żywej gry (indeksy włącznie) z maski martwej piłki."""
    live = ~np.asarray(dead_mask, dtype=bool)
    n = len(live)
    change = np.flatnonzero(np.diff(live.astype(np.int8))) + 1
    bounds = np.concatenate([[0], change, [n]])
    segs = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        if live[s]:
            segs.append((int(s), int(e - 1)))
    return segs


def detect_restarts(
    dead_mask: np.ndarray,
    segments: list | None = None,
    ball_xy: np.ndarray | None = None,
    params: PossessionParams | None = None,
) -> list[RestartEvent]:
    """Wykrywa wznowienia: przejścia martwa->żywa, opcjonalnie z hintów segments.

    `segments` (wyjście modułu martwej piłki): elementy z polami
    `end`/`restart_frames`, `restart_xy`, `confidence`, `end_sigma`.
    """
    params = params or PossessionParams()
    dead = np.asarray(dead_mask, dtype=bool)
    segs = live_segments(dead)
    if not segs:
        return []

    # mapowanie: początek segmentu żywego -> hint z modułu martwej piłki
    hints: dict[int, dict] = {}
    if segments:
        for seg in segments:
            end = _get(seg, "end", _get(seg, "restart_frame", None))
            if end is None:
                continue
            # restart_frame bywa jawnie podane; inaczej end+1
            rf = _get(seg, "restart_frame", None)
            frame = int(rf) if rf is not None else int(end) + 1
            conf = _get(seg, "confidence", 0.5)
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = 0.5
            end_sigma = _get(seg, "end_sigma", None)
            weight = conf
            if end_sigma is not None:
                try:
                    weight = conf / (1.0 + float(end_sigma))
                except (TypeError, ValueError):
                    pass
            hints[frame] = {
                "xy": _get(seg, "restart_xy", None),
                "weight": float(np.clip(weight, 0.0, 1.0)),
            }

    events = []
    for start, _end in segs:
        # hint z segments może być przesunięty o ±2 klatki względem granicy
        # dead_mask — bierzemy najbliższy
        hint = {}
        for delta in (0, -1, 1, -2, 2):
            if start + delta in hints:
                hint = hints[start + delta]
                break
        xy = hint.get("xy")
        if xy is None and ball_xy is not None:
            # punkt wznowienia = pierwsza widoczna pozycja piłki w segmencie
            probe = ball_xy[start : start + params.executor_window * 5]
            ok = np.isfinite(probe).all(axis=1)
            if ok.any():
                xy = probe[np.flatnonzero(ok)[0]]
        events.append(
            RestartEvent(
                frame=start,
                team=None,  # uzupełniane przez infer_executor
                weight=hint.get("weight", 0.7),
                xy=np.asarray(xy, dtype="float64") if xy is not None else None,
                source="segments" if start in hints else "dead_mask",
            )
        )
    return events


def infer_executor(
    ball_xy: np.ndarray,
    players_h: np.ndarray,
    players_a: np.ndarray,
    restart: RestartEvent,
    params: PossessionParams,
) -> str | None:
    """§4.6: drużyna wykonawcy = bliżej do punktu wznowienia w pierwszych klatkach.

    Dla każdego t w oknie [frame, frame+executor_window): punkt odniesienia to
    restart.xy (hint) albo pozycja piłki; liczymy min. dystans do zawodnika
    każdej drużyny i uśredniamy. Zwraca 'H'/'A'/None.
    """
    n = len(ball_xy)
    f0 = restart.frame

    # Punkt odniesienia (§4.6): gdy piłka jest NaN na starcie, bierzemy pierwszą
    # widoczną pozycję w ciągu 5 s i propagujemy stan wstecz — tu tylko
    # geometrię: kotwica = pierwsza widoczna klatka, fallback = hint restart_xy.
    probe_end = min(f0 + 5 * FPS, n)
    ball_seg = ball_xy[f0:probe_end]
    ball_ok = np.isfinite(ball_seg).all(axis=1)
    anchor = int(np.flatnonzero(ball_ok)[0]) + f0 if ball_ok.any() else None

    if anchor is not None:
        ref_xy = ball_xy[anchor]
        w0 = min(anchor, n - 1)
    elif restart.xy is not None:
        ref_xy = np.asarray(restart.xy, dtype="float64")
        w0 = f0
    else:
        return None

    f1 = min(w0 + params.executor_window, n)
    if f1 <= w0:
        return None

    d_h = np.full(f1 - w0, np.nan)
    d_a = np.full(f1 - w0, np.nan)
    ph = players_h[w0:f1]
    pa = players_a[w0:f1]
    for i in range(f1 - w0):
        for arr, out in ((ph, d_h), (pa, d_a)):
            frame_players = arr[i]
            ok = np.isfinite(frame_players).all(axis=1)
            if ok.any():
                d = np.sqrt(((frame_players[ok] - ref_xy) ** 2).sum(axis=1))
                out[i] = d.min()

    valid = np.isfinite(d_h) & np.isfinite(d_a)
    if valid.sum() < params.executor_min_frames:
        return None
    return "H" if np.nanmean(d_h[valid]) < np.nanmean(d_a[valid]) else "A"


def apply_executor_teams(
    restarts: list[RestartEvent],
    ball_xy: np.ndarray,
    players_h: np.ndarray,
    players_a: np.ndarray,
    params: PossessionParams,
) -> list[RestartEvent]:
    """Uzupełnia restart.team przez inferencję geometryczną (zastępuje None)."""
    out = []
    for r in restarts:
        team = r.team or infer_executor(ball_xy, players_h, players_a, r, params)
        out.append(
            RestartEvent(frame=r.frame, team=team, weight=r.weight, xy=r.xy, source=r.source)
        )
    return out
