"""4.7 Dekoder: semi-Markov, dwa przejścia, per segment żywej gry.

Stany: {H, A} (+ wewnętrzny `contested` rozwiązany do Inkumbenta — w praktyce
egzekwowany przez koszt przełączenia λ i minimalny czas trwania d_min).

- Przejście 1 (forward): Viterbi na emisjach s_t z karą przełączenia λ.
- Wymuszenie minimalnego trwania: krótkie spelle (< d_min) scalane z tańszym
  sąsiadem — semi-Markov przez ograniczenie na długość, nie geometryczny
  rozkład przebywania.
- Przejście 2 (backward): ten sam dekoder na odwróconej sekwencji; granice
  przejęć dopasowywane parami i stawiane na min(fwd, bwd) — takeover ląduje
  na pierwszej klatce kontaktu (wsteczna korekta, nie łatka).

Czysta funkcja: deterministyczna, bez stochastyki.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import PossessionParams
from .restarts import RestartEvent

H, A = 1, 2  # kody etykiet wewnętrznych


@dataclass(frozen=True)
class DecodeResult:
    """Wynik dekodowania.

    - labels: (N,) int8; H/A na klatkach żywych segmentów, 0 poza nimi
    - boundaries: lista (frame, from_label, to_label) — kandydackie granice
      przejęć do dyspozycji dekodera czasowego zespołu
    """

    labels: np.ndarray
    boundaries: list[tuple[int, int, int]] = field(default_factory=list)


def _viterbi(
    e_h: np.ndarray, e_a: np.ndarray, switch_cost: float,
    init_team: int | None = None, init_weight: float = 0.0,
) -> np.ndarray:
    """Viterbi dla dwóch stanów. e_h/e_a = emisja (log-ll) na korzyść stanu.

    Zwraca etykiety 0/1 (0=H, 1=A) w postaci int8.
    """
    n = len(e_h)
    labels = np.zeros(n, dtype=np.int8)
    if n == 0:
        return labels
    NEG = -1e18
    dp_h = np.empty(n)
    dp_a = np.empty(n)
    bp_h = np.zeros(n, dtype=np.int8)  # 0=poprzedni H, 1=poprzedni A
    bp_a = np.zeros(n, dtype=np.int8)

    dp_h[0] = e_h[0] + (init_weight if init_team == 0 else 0.0)
    dp_a[0] = e_a[0] + (init_weight if init_team == 1 else 0.0)

    for t in range(1, n):
        # stan H
        stay = dp_h[t - 1]
        switch = dp_a[t - 1] - switch_cost
        if switch > stay:
            dp_h[t] = e_h[t] + switch
            bp_h[t] = 1
        else:
            dp_h[t] = e_h[t] + stay
            bp_h[t] = 0
        # stan A; bp_a[t] = indeks stanu poprzedniego (0=H, 1=A)
        stay = dp_a[t - 1]
        switch = dp_h[t - 1] - switch_cost
        if switch > stay:
            dp_a[t] = e_a[t] + switch
            bp_a[t] = 0  # poprzedni stan: H
        else:
            dp_a[t] = e_a[t] + stay
            bp_a[t] = 1  # poprzedni stan: A

    # backtrack od lepszego stanu końcowego
    state = 0 if dp_h[-1] >= dp_a[-1] else 1
    for t in range(n - 1, -1, -1):
        labels[t] = state
        if t > 0:
            state = bp_h[t] if state == 0 else bp_a[t]
    return labels


def _enforce_min_dwell(
    labels: np.ndarray, e_h: np.ndarray, e_a: np.ndarray, min_dwell: int
) -> np.ndarray:
    """Scala spelle krótsze niż min_dwell z tańszym sąsiadem.

    Koszt scalenia spella do drużyny c = suma (e_c - e_obecna) na spellu.
    Remisy rozstrzyga inertia: preferuj poprzednika (lewy sąsiad), potem
    przy braku sąsiada — następnik. Deterministyczne, do zbieżności.
    """
    lab = labels.copy()
    n = len(lab)
    if n == 0:
        return lab
    emit = np.stack([e_h, e_a], axis=1)  # (N, 2)

    for _ in range(n + 1):  # górny limit iteracji = liczba spellów
        # granice spellów
        change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [n]])
        viol = [
            (s, e) for s, e in zip(starts, ends) if e - s < min_dwell
        ]
        if not viol:
            break
        merged_any = False
        for s, e in viol:
            cur = int(lab[s])
            # kandydaci: lewy sąsiad (lab[s-1]) i prawy (lab[e])
            cands = []
            if s > 0:
                cands.append(int(lab[s - 1]))
            if e < n:
                cands.append(int(lab[e]))
            cands = sorted(set(cands))
            if not cands:
                continue
            # koszt przejścia na kandydata c: emit_c - emit_cur na spellu
            costs = []
            for c in cands:
                costs.append(float((emit[s:e, c] - emit[s:e, cur]).sum()))
            # remisy -> lewy sąsiad (pierwszy w liście); przy remisie kosztów
            # preferujemy też lewy (inercja)
            best = int(cands[int(np.argmin(costs))])
            lab[s:e] = best
            merged_any = True
        if not merged_any:
            break
    return lab


def _switch_frames(labels: np.ndarray) -> list[int]:
    """Indeksy pierwszych klatek nowych spellów (granice)."""
    if len(labels) < 2:
        return []
    return [int(i) for i in np.flatnonzero(labels[1:] != labels[:-1]) + 1]


def _decode_segment(
    s: np.ndarray,
    params: PossessionParams,
    init_team: int | None = None,
    init_weight: float = 0.0,
) -> tuple[np.ndarray, list[int]]:
    """Dwa przejścia na jednym segmencie. Zwraca (labels 0/1, granice)."""
    e_h = s
    e_a = -s

    # --- przejście 1: forward ---
    fwd = _viterbi(e_h, e_a, params.switch_cost, init_team, init_weight)
    fwd = _enforce_min_dwell(fwd, e_h, e_a, params.min_dwell)
    fwd_bounds = _switch_frames(fwd)

    # --- przejście 2: backward (odwrócona sekwencja) ---
    bwd_rev = _viterbi(e_h[::-1], e_a[::-1], params.switch_cost,
                       init_team=None, init_weight=0.0)
    bwd_rev = _enforce_min_dwell(bwd_rev, e_h[::-1], e_a[::-1], params.min_dwell)
    bwd = bwd_rev[::-1]
    bwd_bounds = _switch_frames(bwd)

    # --- fuzja granic: dla każdego forward-przejęcia szukamy odpowiadającej
    # granicy backward w oknie; finalna granica = min(fwd, bwd) => takeover
    # ląduje na pierwszej klatce kontaktu. Niezdopasowane forward-granice
    # zostają (backward nie ma prawa dodawać nowych podziałów).
    win = params.backward_match_window
    fused: list[int] = []
    used_b: set[int] = set()
    for fb in fwd_bounds:
        # szukaj najbliższej bwd granicy w oknie
        cand = [b for b in bwd_bounds
                if b not in used_b and abs(b - fb) <= win]
        if cand:
            b = min(cand, key=lambda x: abs(x - fb))
            used_b.add(b)
            fused.append(min(fb, b))
        else:
            fused.append(fb)
    fused = sorted(set(fused))

    # --- odbudowa etykiet od granic; etykieta startowa = forward ---
    lab = np.full(len(s), fwd[0], dtype=np.int8)
    for b in fused:
        lab[b:] = 1 - lab[b - 1]
    lab = _enforce_min_dwell(lab, e_h, e_a, params.min_dwell)
    return lab, _switch_frames(lab)


def decode_possession(
    s: np.ndarray,
    live_mask: np.ndarray,
    params: PossessionParams | None = None,
    init_priors: dict[int, tuple[int, float]] | None = None,
) -> DecodeResult:
    """Dekoduje cały mecz: osobno każdy ciągły segment żywej gry.

    - s: ewidencja (N,) na korzyść H
    - live_mask: True = klatka żywa (nie z dead_mask)
    - init_priors: {start_frame: (team_code, weight)} — priors po restarcie
      (team_code: 0=H, 1=A)

    Zwraca DecodeResult z etykietami H/A na segmentach i granicami.
    """
    params = params or PossessionParams()
    s = np.asarray(s, dtype="float64")
    n = len(s)
    labels = np.zeros(n, dtype=np.int8)  # 0 = poza segmentami
    boundaries: list[tuple[int, int, int]] = []

    live = np.asarray(live_mask, dtype=bool)
    change = np.flatnonzero(np.diff(live.astype(np.int8))) + 1
    bounds = np.concatenate([[0], change, [n]])
    init_priors = init_priors or {}

    for st, en in zip(bounds[:-1], bounds[1:]):
        if not live[st]:
            continue
        e = en  # en-1 = ostatnia klatka segmentu; slice [st:en)
        seg_s = np.nan_to_num(s[st:e], nan=0.0)
        init = init_priors.get(st)
        init_team = init[0] if init else None
        init_w = init[1] if init else 0.0
        lab, bounds_local = _decode_segment(seg_s, params, init_team, init_w)
        labels[st:e] = np.where(lab == 0, H, A)
        boundaries.extend(st + b for b in bounds_local)

    # granice: (frame, from, to) z etykiet po obu stronach
    fixed = [
        (frame, int(labels[frame - 1]), int(labels[frame]))
        for frame in boundaries
        if 0 < frame < n and labels[frame] != labels[frame - 1]
    ]
    return DecodeResult(labels=labels, boundaries=fixed)
