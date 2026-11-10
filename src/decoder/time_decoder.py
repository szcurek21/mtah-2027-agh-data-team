# src/decoder/smooth.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
# ── dodaj na górze src/decoder/smooth.py ─────────────────────────────
import argparse
from pathlib import Path

N_FRAMES = 199_319
ALLOWED = ("H", "A", "D", "X")

UNK, H, A, D = 0, 1, 2, 3
TO_CODE = {"X": UNK, "H": H, "A": A, "D": D}
TO_LABEL = np.array(["X", "H", "A", "D"])


@dataclass(frozen=True)
class SmoothConfig:
    # wartości z priorytetów domenowych, NIE dostrajane na zestawie testowym
    min_spell: int = 25        # H/A krótsze niż 1 s traktujemy jako szum
    min_dead: int = 25         # D krótsze niż 1 s też
    micro_gap: int = 6         # mikro-zaniki trackingu (~0.24 s) sklejamy przed filtracją szumu
    max_fill_gap: int = 125    # luki (X) do 5 s między stabilnymi etykietami wypełniamy

    def min_len(self, lab: int) -> int:
        return self.min_dead if lab == D else self.min_spell

def _rle(x: np.ndarray) -> tuple[list[int], list[int]]:
    cut = np.flatnonzero(x[1:] != x[:-1]) + 1
    starts = np.r_[0, cut]
    lens = np.diff(np.r_[starts, len(x)])
    return x[starts].tolist(), lens.tolist()


def _merge(labs: list[int], lens: list[int]) -> tuple[list[int], list[int]]:
    out_l, out_n = [labs[0]], [lens[0]]
    for l, n in zip(labs[1:], lens[1:]):
        if l == out_l[-1]:
            out_n[-1] += n
        else:
            out_l.append(l)
            out_n.append(n)
    return out_l, out_n


# def smooth_half(codes: np.ndarray, cfg: SmoothConfig) -> np.ndarray:
#     """Wygładza kody (0=X,1=H,2=A,3=D) jednej połowy."""
#     if len(codes) == 0:
#         return codes
#     labs, lens = _rle(codes)
#     while True:
#         changed = False
#         # 1) wypełnij luki ograniczone tą samą etykietą
#         for i in range(1, len(labs) - 1):
#             if (labs[i] == UNK and labs[i - 1] == labs[i + 1] != UNK
#                     and lens[i] <= cfg.max_fill_gap):
#                 labs[i] = labs[i - 1]
#                 changed = True
#         labs, lens = _merge(labs, lens)
#         # 2) krótkie znane przebiegi (poza krawędziami połowy) -> X
#         for i in range(1, len(labs) - 1):
#             if labs[i] != UNK and lens[i] < cfg.min_len(labs[i]):
#                 labs[i] = UNK
#                 changed = True
#         labs, lens = _merge(labs, lens)
#         if not changed:
#             break
#     return np.repeat(labs, lens).astype(np.int8)
#lepsza wersja
def smooth_half(codes: np.ndarray, cfg: SmoothConfig) -> np.ndarray:
    """Wygładza kody (0=X, 1=H, 2=A, 3=D) jednej połowy."""
    if len(codes) == 0:
        return codes
    labs, lens = _rle(codes)

    # KROK 1: Zlep mikro-luki (np. 1-5 klatek zgubionego trackingu wewnątrz akcji),
    # ale tylko wtedy, gdy sąsiednie bloki razem dają już realną akcję (>= min_len).
    for i in range(1, len(labs) - 1):
        if (labs[i] == UNK and labs[i - 1] == labs[i + 1] != UNK
                and lens[i] <= cfg.micro_gap
                and (lens[i - 1] + lens[i + 1]) >= cfg.min_len(labs[i - 1])):
            labs[i] = labs[i - 1]
    labs, lens = _merge(labs, lens)

    # KROK 2: Usuń krótkie przebiegi będące szumem (< 1 s)
    # Zabezpieczenie: zachowaj najdłuższy przebieg, gdyby wszystkie były < min_len
    known_indices = [i for i, l in enumerate(labs) if l != UNK]
    longest_known_idx = max(known_indices, key=lambda i: lens[i]) if known_indices else None

    for i in range(len(labs)):
        if labs[i] != UNK and lens[i] < cfg.min_len(labs[i]):
            labs[i] = UNK

    # Jeśli wycięliśmy absolutnie wszystkie znane etykiety w połowie, przywróć najdłuższą
    if all(l == UNK for l in labs) and longest_known_idx is not None:
        labs, lens = _rle(codes)
        for i in range(len(labs)):
            if i != longest_known_idx:
                labs[i] = UNK

    labs, lens = _merge(labs, lens)

    # KROK 3: Wypełnij luki (X) do max_fill_gap (5 s) między tymi samymi stabilnymi etykietami
    for i in range(1, len(labs) - 1):
        if (labs[i] == UNK and labs[i - 1] == labs[i + 1] != UNK
                and lens[i] <= cfg.max_fill_gap):
            labs[i] = labs[i - 1]
    labs, lens = _merge(labs, lens)

    # KROK 4: Backfill na samym starcie połowy (eliminacja "osieroconych" X bez historii do dziedziczenia)
    if len(labs) >= 2 and labs[0] == UNK and labs[1] != UNK:
        labs[0] = labs[1]
        labs, lens = _merge(labs, lens)

    return np.repeat(labs, lens).astype(np.int8)

def smooth_labels(raw, half: np.ndarray, cfg: SmoothConfig = SmoothConfig()) -> np.ndarray:
    """raw: etykiety 'H'/'A'/'D'/'X'/NaN dla każdej klatki; half: kolumna z load_frames."""
    codes = pd.Series(raw).map(TO_CODE).fillna(UNK).astype(np.int8).to_numpy()
    out = np.full(len(codes), UNK, dtype=np.int8)       # klatki nieoceniane -> X
    for h in np.unique(half[~np.isnan(half)]):
        m = half == h
        out[m] = smooth_half(codes[m], cfg)
    return TO_LABEL[out]


def spell_stats(labels: np.ndarray, half: np.ndarray) -> pd.DataFrame:
    """Kontrola poprawności bez ground truth (z uwzględnieniem dziedziczenia 'X' w spellF1)."""
    rows = []
    for h in np.unique(half[~np.isnan(half)]):
        # Symulacja zachowania ewaluatora: 'X' dziedziczy ostatnią znaną etykietę (ffill)
        l = pd.Series(labels[half == h]).replace("X", np.nan).ffill().dropna()
        if l.empty:
            rows.append({"half": int(h), "spells": 0, "median_s": np.nan, "p10_s": np.nan})
            continue
        runs = (l != l.shift()).cumsum()
        lens = l.groupby(runs).size()
        rows.append({
            "half": int(h),
            "spells": len(lens),
            "median_s": lens.median() / 25,
            "p10_s": lens.quantile(0.1) / 25,
        })
    return pd.DataFrame(rows)

# ── dopisz na końcu src/decoder/smooth.py ────────────────────────────
def write_submission(labels_array, output_path, n_frames: int = N_FRAMES) -> pd.DataFrame:
    """Waliduje etykiety i zapisuje CSV: nagłówek `frame,label`, frame = 0..n_frames-1."""
    labels = pd.Series(np.asarray(labels_array, dtype=object))

    if len(labels) != n_frames:
        raise ValueError(f"Oczekiwano {n_frames} etykiet, dostałem {len(labels)}.")
    if labels.isna().any():
        raise ValueError(f"W etykietach jest {int(labels.isna().sum())} wartości NaN.")
    bad = set(labels.unique()) - set(ALLOWED)
    if bad:
        raise ValueError(f"Niedozwolone etykiety: {sorted(map(str, bad))}. Dozwolone: {ALLOWED}.")

    sub = pd.DataFrame({
        "frame": np.arange(n_frames, dtype=np.int64),
        "label": labels.astype(str).to_numpy(),
    })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)

    # odczyt kontrolny pliku z dysku (to on jest oceniany, nie DataFrame w pamięci)
    chk = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    if list(chk.columns) != ["frame", "label"]:
        raise ValueError(f"Zły nagłówek: {list(chk.columns)}")
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


# def load_raw_labels(path, n_frames: int = N_FRAMES) -> np.ndarray:
#     """Wczytuje surowe etykiety (frame,label) i wyrównuje do 0..n_frames-1.
#     Brakujące klatki i puste komórki zostają NaN (smooth_labels zamieni je na X)."""
#     raw = pd.read_csv(path, dtype={"label": str})
#     if not {"frame", "label"} <= set(raw.columns):
#         raise ValueError(f"{path} musi mieć kolumny 'frame' i 'label', ma {list(raw.columns)}")
#     if raw["frame"].duplicated().any():
#         raise ValueError("Zduplikowane wartości w kolumnie frame.")
#     s = raw.set_index("frame")["label"].reindex(range(n_frames))
#     unknown = set(s.dropna().unique()) - set(ALLOWED)
#     if unknown:
#         raise ValueError(f"Nieznane etykiety w {path}: {sorted(unknown)}")
#     return s.to_numpy(dtype=object)
#lepsza wersja
def load_raw_labels(path, half: np.ndarray | None = None, n_frames: int = N_FRAMES) -> np.ndarray:
    """Wczytuje surowe etykiety i bezpiecznie wyrównuje do video_frame 0..n_frames-1."""
    raw = pd.read_csv(path, dtype={"label": str})

    # Obsługa 'video_frame' jako aliasu dla 'frame' (zgodność z load.py i cechy.py)
    if "video_frame" in raw.columns and "frame" not in raw.columns:
        raw = raw.rename(columns={"video_frame": "frame"})

    if not {"frame", "label"} <= set(raw.columns):
        raise ValueError(f"{path} musi mieć kolumny 'frame' (lub 'video_frame') i 'label', ma {list(raw.columns)}")
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

    # Strażnik 2: zgubienie indeksu przy zapisie samych klatek 'graded' (reset_index)
    if half is not None:
        n_graded = int((~np.isnan(half)).sum())
        if len(raw) == n_graded and raw["frame"].min() == 0 and raw["frame"].max() == n_graded - 1:
            raise ValueError(
                f"Plik ma dokładnie {n_graded} wierszy (tyle co graded), ale indeksy 0..{n_graded-1}! "
                "Prawdopodobnie zresetowano indeks (reset_index) przed zapisem CSV."
            )

    s = raw.set_index("frame")["label"].str.strip().str.upper()
    s = s.replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    s = s.reindex(range(n_frames))

    unknown = set(s.dropna().unique()) - set(ALLOWED)
    if unknown:
        raise ValueError(f"Nieznane etykiety w {path}: {sorted(unknown)}")
    return s.to_numpy(dtype=object)

def load_half(n_frames: int = N_FRAMES) -> np.ndarray:
    """Kolumna `half` z load_frames, wyrównana do video_frame 0..n_frames-1 (poza połowami = NaN)."""
    from src.decoder.load import load_frames

    frames = load_frames("BAR-ATH", "tactical")
    if len(frames) != n_frames:
        print(f"UWAGA: load_frames zwróciło {len(frames)} klatek, oczekiwano {n_frames}.")
    return (frames.set_index("video_frame")["half"]
            .reindex(range(n_frames)).to_numpy(dtype=float))


def make_demo_raw(half: np.ndarray, seed: int = 0) -> np.ndarray:
    """Syntetyczne, migoczące etykiety do testu całego potoku (NIE do submisji)."""
    rng = np.random.default_rng(seed)
    n = len(half)
    raw = np.full(n, "X", dtype=object)
    i = 0
    while i < n:
        L = int(rng.integers(100, 600))
        raw[i:i + L] = rng.choice(["H", "A", "D"], p=[0.45, 0.45, 0.10])
        i += L
    flip = rng.random(n) < 0.08                       # migotanie
    raw[flip] = rng.choice(["H", "A"], size=flip.sum())
    hidden = np.zeros(n, dtype=bool)                  # piłka znika w blokach ~41%
    j = 0
    while j < n:
        L = int(rng.integers(20, 200))
        hidden[j:j + L] = rng.random() < 0.41
        j += L
    raw[hidden] = "X"
    return raw


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dekoder czasowy: raw -> wygładzone -> submission.csv")
    ap.add_argument("--input", default="baseline_raw.csv")
    ap.add_argument("--output", default="submission.csv")
    ap.add_argument("--demo", action="store_true",
                    help="zamiast pliku użyj syntetycznych etykiet (tylko test potoku)")
    args = ap.parse_args()

    half = load_half()
    raw = make_demo_raw(half) if args.demo else load_raw_labels(args.input)

    cfg = SmoothConfig()                              # parametry domenowe, bez strojenia
    smoothed = smooth_labels(raw, half, cfg)

    print("\nRozkład etykiet (surowe -> wygładzone):")
    print(pd.DataFrame({
        "raw": pd.Series(raw).fillna("X").value_counts(),
        "smoothed": pd.Series(smoothed).value_counts(),
    }).fillna(0).astype(int).to_string())
    print("\nStatystyki spelli po wygładzeniu:")
    print(spell_stats(smoothed, half).round(2).to_string(index=False))

    write_submission(smoothed, args.output)