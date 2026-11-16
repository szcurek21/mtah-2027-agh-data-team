"""Parametry modułu posiadania — specyfikacja v3 ("10/10").

Wszystkie wartości domyślne są punktem startowym kalibracji (§7 specyfikacji);
żaden próg nie jest stroiony pod publicznym GT.
"""
from __future__ import annotations

from dataclasses import dataclass

FPS = 25

# Wymiary boiska w metrach — zgodne z src/cechy.py (PITCH_L, PITCH_W).
PITCH_L_M = 105.3
PITCH_W_M = 68.0

# Jeśli maksymalna |współrzędna| jest <= tej wartości, traktujemy dane jako
# znormalizowane [0,1]^2 i skalujemy do metrów. Loader (kloppy "secondspectrum")
# podaje już metry wyśrodkowane w (0,0), więc próg 1.5 bezpiecznie je rozróżnia.
NORM_COORD_LIMIT = 1.5


@dataclass(frozen=True)
class PossessionParams:
    # --- 4.3 Czyszczenie trajektorii piłki ---
    sg_window: int = 7        # Savitzky–Golay, nieparzyste; 7 klatek ~ ±0.28 s @25 fps
    sg_poly: int = 2          # rząd wielomianu SG
    max_interp_gap: int = 10  # maks. luka interpolowana (~0.4 s); dłuższe zostają NaN

    # --- 4.4 Model emisji ---
    tau: float = 1.5          # skala odległości (m); kalibrowana na klatkach restartów
    evidence_clip: float = 10.0  # clip |s_t| — ogranicza wpływ outlierów
    flight_speed: float = 12.0  # m/s; powyżej = piłka w locie (progi kalibrowane)
    scramble_speed: float = 7.0  # m/s; dolny próg „szybkiej" piłki dla proxy lotu
    scramble_radius: float = 2.0  # m; brak gracza bliżej + szybka piłka => brak kontroli

    # --- 4.6 Reinicjalizacja po wznowieniach ---
    restart_prior: float = 2.0    # siła priorsu wykonawcy (jednostki log-ll)
    restart_window: int = 25      # klatki (~1 s) po wznowieniu, w których działa prior
    executor_window: int = 10     # klatek do inferencji drużyny wykonawcy
    executor_min_frames: int = 3  # min. liczba klatek z danymi do decyzji

    # --- 4.7 Dekoder semi-Markov, dwa przejścia ---
    switch_cost: float = 30.0     # λ: koszt zmiany drużyny (log-ll); ~1–2 s dowodów
    min_dwell: int = 30           # min. czas trwania spella (klatki); 30 = 1.2 s
    backward_match_window: int = 50  # max. odległość dopasowania granic fwd/bwd

    # --- 4.1 Przypisanie drużyn (fallback, gdy brak mapowania ze slotów) ---
    team_gate_m: float = 3.0      # max. dystans propagacji etykiety drużyny między klatkami
    team_anchor_offset: int = 50  # klatki po starcie połów używane jako kotwice
