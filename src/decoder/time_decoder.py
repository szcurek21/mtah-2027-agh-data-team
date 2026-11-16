# src/decoder/smooth.py
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_FRAMES = 199_319
FPS = 25
ALLOWED = ("H", "A", "D", "X")

UNK, H, A, D = 0, 1, 2, 3
TO_CODE = {"X": UNK, "H": H, "A": A, "D": D}
TO_LABEL = np.array(["X", "H", "A", "D"])

# Domyślne granice połów dla BAR-ATH (fallback, gdy brak plików XML/TXT na dysku)
DEFAULT_HALF_BLOCKS: list[tuple[int, int, int]] = [
    (1, 11185, 90651),
    (2, 110808, 190197),
]


@dataclass(frozen=True)
class SmoothConfig:
    """Parametry wynikające z priorytetów domenowych (25 FPS), bez strojenia na teście."""

    min_spell: int = 25  # H/A krótsze niż 1.0 s (25 klatek) traktujemy jako szum
    min_dead: int = 25  # D krótsze niż 1.0 s (25 klatek) traktujemy jako szum
    mode_win: int = 7  # okno głosowania większościowego (~0.28 s) na pojedyncze przeskoki H<->A
    bridge_gap: int = (
        18  # luki X (~0.72 s) sklejane przed KROKIEM 2, jeśli suma sąsiadów >= min_len
    )
    micro_gap: int = 6  # mikro-zaniki trackingu (~0.24 s) sklejane przed filtracją szumu
    max_fill_gap: int = (
        125  # luki (X) do 5.0 s (125 klatek) między tą samą stabilną akcją wypełniamy
    )
    max_start_backfill: int = (
        750  # maks. luka X na starcie połowy (30 s) uzupełniana pierwszą akcją
    )

    def min_len(self, lab: int) -> int:
        return self.min_dead if lab == D else self.min_spell


def _rle(x: np.ndarray) -> tuple[list[int], list[int]]:
    """Kompresja Run-Length Encoding: zwraca (etykiety_bloków, długości_bloków)."""
    if len(x) == 0:
        return [], []
    cut = np.flatnonzero(x[1:] != x[:-1]) + 1
    starts = np.r_[0, cut]
    lens = np.diff(np.r_[starts, len(x)])
    return x[starts].tolist(), lens.tolist()


def _merge(labs: list[int], lens: list[int]) -> tuple[list[int], list[int]]:
    """Scala sąsiadujące bloki o tej samej etykiecie."""
    if not labs:
        return [], []
    out_l, out_n = [labs[0]], [lens[0]]
    for lab, n in zip(labs[1:], lens[1:]):
        if lab == out_l[-1]:
            out_n[-1] += n
        else:
            out_l.append(lab)
            out_n.append(n)
    return out_l, out_n


def mode_filter(codes: np.ndarray, win: int = 7) -> np.ndarray:
    """Lokalny filtr większościowy usuwający krótkie migotanie (np. H(12)-A(1)-H(12)).
    Klatki UNK ('X') nie głosują i same nigdy nie są nadpisywane.
    """
    # np.convolve(mode="same") zwraca max(len(a), len(kernel)) -> dla wejścia krótszego niż okno
    # długość wyjścia byłaby inna niż wejścia, więc takiego wejścia nie filtrujemy.
    if win <= 1 or len(codes) < win:
        return codes.astype(np.int8, copy=True)
    onehot = np.stack([(codes == k) for k in (H, A, D)], axis=1).astype(np.float32)
    kernel = np.ones(win, dtype=np.float32)
    votes = np.stack([np.convolve(onehot[:, j], kernel, mode="same") for j in range(3)], axis=1)
    top = votes.max(axis=1, keepdims=True)
    tie = (votes == top).sum(axis=1) > 1
    out = votes.argmax(axis=1) + 1
    out[(top[:, 0] == 0) | tie | (codes == UNK)] = UNK
    return out.astype(np.int8)


def smooth_half(codes: np.ndarray, cfg: SmoothConfig = SmoothConfig()) -> np.ndarray:
    """Wygładza kody (0=X, 1=H, 2=A, 3=D) pojedynczej połowy meczu."""
    if len(codes) == 0:
        return codes.astype(np.int8, copy=True)

    # KROK 0: Lokalny filtr większościowy na pojedyncze przeskoki klasyfikatora (H <-> A/D)
    cleaned = mode_filter(codes, win=cfg.mode_win)
    labs, lens = _rle(cleaned)

    # KROK 1: Zlep mikro-luki (<= micro_gap klatek zgubionego trackingu) między blokami tej samej
    # etykiety. Bez warunku na sumę sąsiadów: sklejenie nie tworzy akcji z niczego, a o tym, czy
    # wynik jest realną akcją (>= min_len), decyduje dopiero krok 2. Dzięki temu kaskada
    # H(10) X(3) H(10) X(3) H(10) skleja się w jedno H(36).
    changed = True
    while changed:
        changed = False
        for i in range(1, len(labs) - 1):
            if labs[i - 1] == labs[i + 1] != UNK:
                target = labs[i - 1]
                neighbors_len = lens[i - 1] + lens[i + 1]
                is_micro_x = labs[i] == UNK and lens[i] <= cfg.micro_gap
                is_bridge_x = (
                    labs[i] == UNK
                    and lens[i] <= cfg.bridge_gap
                    and neighbors_len >= cfg.min_len(target)
                )
                is_glitch = (
                    labs[i] != UNK
                    and lens[i] <= cfg.micro_gap
                    and neighbors_len >= cfg.min_len(target)
                )
                if is_micro_x or is_bridge_x or is_glitch:
                    labs[i] = target
                    changed = True
        if changed:
            labs, lens = _merge(labs, lens)

    # KROK 2: Usuń krótkie przebiegi będące szumem (< 1 s)
    labs_backup = labs.copy()
    known_indices = [i for i, lab in enumerate(labs) if lab != UNK]
    longest_known_idx = max(known_indices, key=lambda i: lens[i]) if known_indices else None

    for i in range(len(labs)):
        if labs[i] != UNK and lens[i] < cfg.min_len(labs[i]):
            labs[i] = UNK

    # Zabezpieczenie: jeśli wycięto absolutnie wszystkie znane etykiety w połowie, zachowaj najdłuższą
    rescued = False
    if all(lab == UNK for lab in labs) and longest_known_idx is not None:
        rescued = True
        warnings.warn(
            "Wszystkie przebiegi w połowie były krótsze niż min_len; zachowano najdłuższy przebieg.",
            RuntimeWarning,
            stacklevel=2,
        )
        labs[longest_known_idx] = labs_backup[longest_known_idx]

    labs, lens = _merge(labs, lens)

    # KROK 3: Wypełnij luki (X) do max_fill_gap (5 s) między tymi samymi stabilnymi etykietami
    for i in range(1, len(labs) - 1):
        if labs[i] == UNK and labs[i - 1] == labs[i + 1] != UNK and lens[i] <= cfg.max_fill_gap:
            labs[i] = labs[i - 1]
    labs, lens = _merge(labs, lens)

    # KROK 4: Backfill na samym starcie połowy (z limitem max_start_backfill).
    # Pomijamy go po awaryjnym fallbacku: uratowany, krótki przebieg jest niepewny,
    # więc nie rozszerzamy go zgadywaniem na początek połowy.
    if (
        not rescued
        and len(labs) >= 2
        and labs[0] == UNK
        and labs[1] != UNK
        and lens[0] <= cfg.max_start_backfill
    ):
        labs[0] = labs[1]
        labs, lens = _merge(labs, lens)

    return np.repeat(labs, lens).astype(np.int8)


def smooth_labels(raw, half, cfg: SmoothConfig = SmoothConfig()) -> np.ndarray:
    """Wygładza etykiety ('H'/'A'/'D'/'X'/NaN) dla całego meczu z podziałem na połowy."""
    half_arr = np.asarray(half, dtype=float)
    raw_ser = pd.Series(raw, dtype=object)

    if len(raw_ser) != len(half_arr):
        raise ValueError(
            f"Niezgodna długość wejścia: len(raw)={len(raw_ser)} != len(half)={len(half_arr)}."
        )

    norm = raw_ser.astype(str).str.strip().str.upper()
    norm = norm.where(~raw_ser.isna(), other="X")
    norm = norm.replace({"NAN": "X", "NONE": "X", "": "X"})

    unknown = set(norm.unique()) - set(ALLOWED)
    if unknown:
        raise ValueError(f"Niedozwolone etykiety wejściowe w smooth_labels: {sorted(unknown)}")

    codes = norm.map(TO_CODE).to_numpy(dtype=np.int8)
    out = np.full(len(codes), UNK, dtype=np.int8)  # klatki nieoceniane poza połowami -> X

    for h in np.unique(half_arr[~np.isnan(half_arr)]):
        m = half_arr == h
        idx = np.flatnonzero(m)
        if len(idx) > 1 and (idx[-1] - idx[0] + 1 != len(idx)):
            raise ValueError(
                f"Połowa {h} nie jest ciągłym blokiem klatek (zawiera dziury NaN w środku)!"
            )
        out[m] = smooth_half(codes[m], cfg)

    return TO_LABEL[out]


def spell_stats(
    labels: np.ndarray, half: np.ndarray, raw: np.ndarray | None = None
) -> pd.DataFrame:
    """Kontrola poprawności bez ground truth (z uwzględnieniem dziedziczenia 'X' w spellF1)."""
    half_arr = np.asarray(half, dtype=float)
    rows = []
    for h in np.unique(half_arr[~np.isnan(half_arr)]):
        m = half_arr == h
        sub_sm = pd.Series(labels[m])
        x_pct_sm = float((sub_sm == "X").mean() * 100)
        x_pct_raw = (
            float((pd.Series(raw[m]).isna() | (pd.Series(raw[m]) == "X")).mean() * 100)
            if raw is not None
            else np.nan
        )

        # Symulacja zachowania ewaluatora spellF1: 'X' dziedziczy ostatnią znaną etykietę (ffill)
        inherited = sub_sm.replace("X", np.nan).ffill().dropna()
        if inherited.empty:
            rows.append(
                {
                    "half": int(h),
                    "spells": 0,
                    "switches": 0,
                    "median_s": np.nan,
                    "p10_s": np.nan,
                    "raw_X_%": round(x_pct_raw, 1),
                    "smooth_X_%": round(x_pct_sm, 1),
                }
            )
            continue

        runs = (inherited != inherited.shift()).cumsum()
        lens = inherited.groupby(runs).size()
        rows.append(
            {
                "half": int(h),
                "spells": int(len(lens)),
                "switches": int(len(lens) - 1),
                "median_s": float(lens.median() / FPS),
                "p10_s": float(lens.quantile(0.1) / FPS),
                "raw_X_%": round(x_pct_raw, 1),
                "smooth_X_%": round(x_pct_sm, 1),
            }
        )
    return pd.DataFrame(rows)


def write_submission(labels_array, output_path, n_frames: int = N_FRAMES) -> pd.DataFrame:
    """Waliduje etykiety i zapisuje CSV: nagłówek `frame,label`, frame = 0..n_frames-1."""
    labels = pd.Series(np.asarray(labels_array, dtype=object))

    if len(labels) != n_frames:
        raise ValueError(f"Oczekiwano {n_frames} etykiet, otrzymano {len(labels)}.")
    if labels.isna().any():
        raise ValueError(f"W etykietach znajduje się {int(labels.isna().sum())} wartości NaN.")
    bad = set(labels.unique()) - set(ALLOWED)
    if bad:
        raise ValueError(f"Niedozwolone etykiety: {sorted(map(str, bad))}. Dozwolone: {ALLOWED}.")

    sub = pd.DataFrame(
        {
            "frame": np.arange(n_frames, dtype=np.int64),
            "label": labels.astype(str).to_numpy(),
        }
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)

    # Odczyt kontrolny pliku bezpośrednio z dysku
    chk = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    if list(chk.columns) != ["frame", "label"]:
        raise ValueError(f"Zły nagłówek pliku: {list(chk.columns)}")
    if len(chk) != n_frames:
        raise ValueError(f"Plik ma {len(chk)} wierszy zamiast {n_frames}.")
    if (chk["frame"] == "").any() or (chk["label"] == "").any():
        raise ValueError("Plik zawiera puste komórki.")
    if not (chk["frame"].astype(np.int64).to_numpy() == np.arange(n_frames)).all():
        raise ValueError("Kolumna frame nie jest ciągiem 0..n_frames-1.")
    if not chk["label"].isin(ALLOWED).all():
        raise ValueError("Plik zawiera niedozwolone etykiety.")

    print(f"Zapisano {output_path} ({len(chk)} wierszy) – walidacja OK.")
    return sub


def load_raw_labels(path, half: np.ndarray | None = None, n_frames: int = N_FRAMES) -> np.ndarray:
    """Wczytuje surowe etykiety i bezpiecznie wyrównuje do video_frame 0..n_frames-1."""
    raw = pd.read_csv(path, dtype={"label": str})

    # Obsługa 'video_frame' jako aliasu dla 'frame' (zgodność z load.py i cechy.py)
    if "video_frame" in raw.columns and "frame" not in raw.columns:
        raw = raw.rename(columns={"video_frame": "frame"})

    if not {"frame", "label"} <= set(raw.columns):
        raise ValueError(
            f"{path} musi mieć kolumny 'frame' (lub 'video_frame') i 'label', ma {list(raw.columns)}"
        )
    if raw["frame"].isna().any():
        raise ValueError("Kolumna 'frame' zawiera wartości NaN.")

    raw["frame"] = raw["frame"].astype(np.int64)
    if raw["frame"].duplicated().any():
        raise ValueError("Zduplikowane wartości w kolumnie frame.")

    # Strażnik 1: pomylenie frame_id (1..N) z video_frame (0..N-1)
    if raw["frame"].min() == 1 and raw["frame"].max() == n_frames:
        raise ValueError(
            "Wykryto indeksowanie klatek od 1 do N_FRAMES (prawdopodobnie frame_id zamiast video_frame 0..N-1)!"
        )

    # Strażnik 2: zgubienie indeksu (reset_index) lub przesunięcie o +1 na samych klatkach 'graded'
    if half is not None:
        half_arr = np.asarray(half, dtype=float)
        graded_idx = np.flatnonzero(~np.isnan(half_arr))
        n_graded = len(graded_idx)
        if len(raw) == n_graded and raw["frame"].min() == 0 and raw["frame"].max() == n_graded - 1:
            raise ValueError(
                f"Plik ma dokładnie {n_graded} wierszy (tyle co graded), ale indeksy 0..{n_graded - 1}! "
                "Prawdopodobnie zresetowano indeks (reset_index) przed zapisem CSV."
            )
        if len(raw) == n_graded and np.array_equal(
            np.sort(raw["frame"].to_numpy()) - 1, graded_idx
        ):
            raise ValueError(
                "Wykryto przesunięcie indeksów o +1 na klatkach graded (użyto 1-bazowego frame_id zamiast video_frame)!"
            )

    # Strażnik 3: wyjście poza dozwolony zakres klatek 0..n_frames-1
    if raw["frame"].min() < 0 or raw["frame"].max() >= n_frames:
        raise ValueError(
            f"Indeksy klatek wykraczają poza zakres 0..{n_frames - 1}: "
            f"min={raw['frame'].min()}, max={raw['frame'].max()}."
        )

    s = raw.set_index("frame")["label"].str.strip().str.upper()
    s = s.replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    s = s.reindex(range(n_frames))

    unknown = set(s.dropna().unique()) - set(ALLOWED)
    if unknown:
        raise ValueError(f"Nieznane etykiety w {path}: {sorted(unknown)}")

    # Strażnik 4: kontrola obecności realnych predykcji (H/A/D) w klatkach ocenianych
    if half is not None:
        graded_mask = ~np.isnan(np.asarray(half, dtype=float))
        if graded_mask.any() and not s[graded_mask].isin(["H", "A", "D"]).any():
            warnings.warn(
                "Brak jakichkolwiek predykcji H/A/D w klatkach ocenianych (same 'X' lub NaN)!",
                RuntimeWarning,
                stacklevel=2,
            )

    return s.to_numpy(dtype=object)


def load_half(n_frames: int = N_FRAMES) -> np.ndarray:
    """Kolumna `half` z load_frames, wyrównana do video_frame 0..n_frames-1 (poza połowami = NaN)."""
    try:
        try:
            from src.decoder.load import load_frames
        except ModuleNotFoundError:
            from load import load_frames

        frames = load_frames("BAR-ATH", "tactical")
        if len(frames) != n_frames:
            raise ValueError(
                f"load_frames zwróciło {len(frames)} klatek, oczekiwano dokładnie {n_frames}."
            )
        return (
            frames.set_index("video_frame")["half"].reindex(range(n_frames)).to_numpy(dtype=float)
        )
    except (FileNotFoundError, ImportError) as exc:
        warnings.warn(
            f"Nie udało się wczytać połówek przez load_frames ({type(exc).__name__}: {exc}); "
            "używam domyślnych granic DEFAULT_HALF_BLOCKS.",
            RuntimeWarning,
            stacklevel=2,
        )
        half = np.full(n_frames, np.nan, dtype=float)
        for h, first, last in DEFAULT_HALF_BLOCKS:
            half[first : last + 1] = float(h)
        return half


def make_demo_raw(half: np.ndarray, seed: int = 0) -> np.ndarray:
    """Syntetyczne, migoczące etykiety do testu całego potoku (NIE do submisji)."""
    rng = np.random.default_rng(seed)
    n = len(half)
    raw = np.full(n, "X", dtype=object)
    i = 0
    while i < n:
        length = int(rng.integers(100, 600))
        raw[i : i + length] = rng.choice(["H", "A", "D"], p=[0.45, 0.45, 0.10])
        i += length
    flip = rng.random(n) < 0.08  # migotanie 8%
    raw[flip] = rng.choice(["H", "A"], size=int(flip.sum()))
    hidden = np.zeros(n, dtype=bool)  # piłka znika w blokach ~41%
    j = 0
    while j < n:
        length = int(rng.integers(20, 200))
        hidden[j : j + length] = rng.random() < 0.41
        j += length
    raw[hidden] = "X"
    return raw


def run_tests() -> None:
    """Wbudowane testy niezmienników i przypadków brzegowych."""
    cfg = SmoothConfig()

    # 1. Test niezmienników i idempotencji
    half = np.r_[np.full(5000, 1.0), np.full(300, np.nan), np.full(5000, 2.0)]
    raw = make_demo_raw(half, seed=42)
    out = smooth_labels(raw, half, cfg)
    assert len(out) == len(raw), "Długość wyjścia musi zgadzać się z wejściem."
    assert (out[np.isnan(half)] == "X").all(), "Poza połowami zawsze musi być 'X'."
    assert set(out) <= set(ALLOWED), "Wyjście zawiera niedozwolone znaki."
    assert (smooth_labels(out, half, cfg) == out).all(), "Brak idempotencji smooth_labels!"

    # 2. Test migotania: H(12) A(1) H(12) A(1) H(12) -> powinno dać ciągłe H(38)
    flicker = np.array([H] * 12 + [A] + [H] * 12 + [A] + [H] * 12, dtype=np.int8)
    res_flicker = smooth_half(flicker, cfg)
    assert (res_flicker == H).all(), f"Oczekiwano samych H, otrzymano: {res_flicker}"

    # 3. Test długiego braku danych na starcie połowy: [X*3000, H*500] -> nie wypełnia początku
    long_start = np.array([UNK] * 3000 + [H] * 500, dtype=np.int8)
    res_start = smooth_half(long_start, cfg)
    assert (res_start[:3000] == UNK).all() and (res_start[3000:] == H).all(), (
        "Krok 4 nie powinien wypełniać 3000 klatek X na starcie połowy!"
    )

    # 4. Test kaskadowych mikro-luk: H(10) X(3) H(10) X(3) H(10) -> ciągłe H(36)
    cascade = np.array([H] * 10 + [UNK] * 3 + [H] * 10 + [UNK] * 3 + [H] * 10, dtype=np.int8)
    res_cascade = smooth_half(cascade, cfg)
    assert (res_cascade == H).all(), "Kaskadowe mikro-luki powinny skleić się w jedno H(36)."

    # 5. Test połowy z samymi krótkimi przebiegami (fallback zachowuje najdłuższy)
    short_only = np.array([UNK] * 20 + [H] * 5 + [UNK] * 20 + [A] * 10 + [UNK] * 20, dtype=np.int8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res_short = smooth_half(short_only, cfg)
    assert (res_short == A).sum() == 10 and (res_short == H).sum() == 0, (
        "Fallback powinien zachować wyłącznie najdłuższy krótki przebieg (A*10)."
    )

    # 6. Wejście krótsze niż okno filtra: długość wyjścia musi zgadzać się z wejściem
    for n in (1, 3, 6):
        arr = np.array([H, A, H, A, H, A][:n], dtype=np.int8)
        assert mode_filter(arr, cfg.mode_win).shape == arr.shape, (
            f"mode_filter zmienia długość (n={n})"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert smooth_half(arr, cfg).shape == arr.shape, f"smooth_half zmienia długość (n={n})"

    # 7. Mikro-luki, których sąsiedzi parami są krótsi niż min_len (12+9=21), nie mogą kasować
    #    akcji, jeśli po sklejeniu całość ma >= min_len (12+2+9+2+12 = 37)
    joined = np.array([H] * 12 + [UNK] * 2 + [H] * 9 + [UNK] * 2 + [H] * 12, dtype=np.int8)
    assert (smooth_half(joined, cfg) == H).all(), "H(12) X(2) H(9) X(2) H(12) powinno dać H(37)."

    # 8. Po fallbacku nie wolno doklejać backfillu (uratowany przebieg zostaje dokładnie 10 klatek)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert (smooth_half(short_only, cfg) != UNK).sum() == 10, (
            "Backfill po fallbacku rozszerzył przebieg."
        )

    # 9. Długa luka między różnymi etykietami zostaje X (nie zgadujemy punktu zmiany)
    gap = np.array([A] * 100 + [UNK] * 100 + [H] * 100, dtype=np.int8)
    res_gap = smooth_half(gap, cfg)
    assert (
        (res_gap[100:200] == UNK).all()
        and (res_gap[:100] == A).all()
        and (res_gap[200:] == H).all()
    )

    print("Wszystkie testy jednostkowe i niezmienników zakończone sukcesem (OK)!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dekoder czasowy: raw -> wygładzone -> submission.csv")
    ap.add_argument("--input", default="baseline_raw.csv", help="ścieżka do surowych etykiet CSV")
    ap.add_argument("--output", default=None, help="ścieżka do pliku wyjściowego CSV")
    ap.add_argument(
        "--demo",
        action="store_true",
        help="zamiast pliku użyj syntetycznych etykiet (tylko test potoku)",
    )
    ap.add_argument(
        "--test",
        action="store_true",
        help="uruchom wbudowane testy jednostkowe i zakończ",
    )
    args = ap.parse_args()

    if args.test:
        run_tests()
        raise SystemExit(0)

    # Zabezpieczenie przed przypadkowym wysłaniem pliku demo jako finalnej submisji
    if args.output is None:
        output_path = "demo_NIE_SUBMITOWAC.csv" if args.demo else "submission.csv"
    else:
        output_path = args.output

    half = load_half()
    raw = make_demo_raw(half) if args.demo else load_raw_labels(args.input, half=half)

    cfg = SmoothConfig()
    smoothed = smooth_labels(raw, half, cfg)

    print("\nRozkład etykiet (surowe -> wygładzone):")
    print(
        pd.DataFrame(
            {
                "raw": pd.Series(raw).fillna("X").value_counts(),
                "smoothed": pd.Series(smoothed).value_counts(),
            }
        )
        .fillna(0)
        .astype(int)
        .to_string()
    )

    print("\nStatystyki spelli po wygładzeniu:")
    print(spell_stats(smoothed, half, raw=raw).round(2).to_string(index=False))

    write_submission(smoothed, output_path)
