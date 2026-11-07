# ===IMPORTY===
import pandas as pd
import numpy as np
from src.decoder.load import load_frames, team_of

# ===STAŁE===
FPS = 25.0
HALF_L=105.3/2
HALF_W=68.0/2

# ===FUNKCJE==
def extract_team_xy(df, slots):
    # pobieramy na przemian kolumny X i Y dla każdego zawodnika
    cols = [f"{s}_{axis}" for s in slots for axis in ["x", "y"]]
    # przekształcamy na kształt (klatki, zawodnicy, 2 współrzędne)
    return df[cols].to_numpy().reshape(len(df), len(slots), 2)

# funkcja do sprawdzenia czy wiemy gdzie jest piłka
def ball_visible(ball_xy: np.ndarray) -> np.ndarray:
    return ~np.isnan(ball_xy).any(axis=1)

# funkcja określająca liczbę klatek od ostatniej na której było widać piłkę, 0 jeśli widzimy w bieżącej klatce, NaN jeśli jeszcze nie widzieliśmy
def frames_since_ball_seen(visible: np.ndarray) ->np.ndarray:
    out=np.full(len(visible), np.nan)
    last= None
    for i, v in enumerate(visible):
        if v:
            last=i
        if last is not None:
            out[i]= i-last
    return out

def min_dist_to_players(ball_xy: np.ndarray, players_xy: np.ndarray) -> np.ndarray:
    """
    Oblicza dystans od piłki do najbliższego gracza z macierzy (N, M, 2).
    Zwraca wektor (N,) z odległościami lub NaN przy braku piłki/graczy.
    """
    if players_xy.shape[1] == 0:
        return np.full(len(ball_xy), np.nan)
        

    dists = np.linalg.norm(players_xy - ball_xy[:, None, :], axis=2)
    
    # Maska na klatki, gdzie wszyscy gracze danej drużyny to NaN
    all_nan = np.isnan(dists).all(axis=1)
    
    # Zamiana NaN na np.inf przed wyznaczeniem minimum
    min_dists = np.where(np.isnan(dists), np.inf, dists).min(axis=1)
    min_dists[all_nan] = np.nan
    return min_dists

def nearest_dist_by_team(ball_xy: np.ndarray, home_xy: np.ndarray, away_xy: np.ndarray) -> dict[str, np.ndarray]:
    """Zwraca słownik {'H': (N,), 'A': (N,)} z dystansami do najbliższego zawodnika."""
    return {
        "H": min_dist_to_players(ball_xy, home_xy),
        "A": min_dist_to_players(ball_xy, away_xy),
    }

def ball_speed(ball_xy: np.ndarray,max_speed: float = 60.0)-> np.ndarray:
    zmiana_xy=np.diff(ball_xy, axis=0)
    dystans=np.linalg.norm(zmiana_xy, axis=1)
    predkosc=dystans*FPS
    predkosc[predkosc > max_speed] = np.nan
    out= np.full(len(ball_xy), np.nan)
    out[1:]=predkosc
    return out

def per_half(func, values: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Uruchamia func osobno w każdej połowie i skleja wyniki."""
    out = None
    for h in np.unique(half[~np.isnan(half)]):
        m = half == h
        res = func(values[m])
        if out is None:
            out = np.full((len(values),) + res.shape[1:], np.nan)
        out[m] = res
    return out

def rolling_median(x: np.ndarray, window: int, min_periods: int = 1) -> np.ndarray:
    """Ruchoma mediana wokół każdego punktu (okno wyśrodkowane), działa na 1D i 2D."""
    out = (pd.DataFrame(x).rolling(window, center=True, min_periods=min_periods)
           .median().to_numpy().copy())
    return out.reshape(x.shape)

def smooth_ball(ball_xy: np.ndarray, window: int = 5) -> np.ndarray:
    """Wygładza pozycję piłki medianą, ale nie wymyśla jej tam, gdzie jej nie widać."""
    sm = rolling_median(ball_xy, window)
    sm[np.isnan(ball_xy)] = np.nan
    return sm


def count_within(ball_xy: np.ndarray, players_xy: np.ndarray, radius: float) -> np.ndarray:
    """Liczba zawodników w promieniu radius od piłki; NaN, gdy piłki nie widać."""
    dists = np.linalg.norm(players_xy - ball_xy[:, None, :], axis=2)
    counts = (dists < radius).sum(axis=1).astype(float)   # NaN < r to False, więc brak gracza nie liczy się
    counts[np.isnan(ball_xy).any(axis=1)] = np.nan
    return counts


def n_tracked(players_xy: np.ndarray) -> np.ndarray:
    """Ilu zawodników drużyny ma w danej klatce pozycję (jakość trackingu)."""
    return (~np.isnan(players_xy).any(axis=2)).sum(axis=1).astype(float)


def ball_outside_pitch(ball_xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """1.0, gdy piłka jest poza boiskiem (podpowiedź dla piłki martwej), 0.0 w grze, NaN bez piłki."""
    out = ((np.abs(ball_xy[:, 0]) > HALF_L + margin) |
           (np.abs(ball_xy[:, 1]) > HALF_W + margin)).astype(float)
    out[np.isnan(ball_xy).any(axis=1)] = np.nan
    return out


def compute_features(frames: pd.DataFrame, teams: dict) -> pd.DataFrame:
    """Jeden wiersz na klatkę (indeks = video_frame), jedna kolumna na cechę."""
    half = frames["half"].to_numpy()
    ball_xy = frames[["ball_x", "ball_y"]].to_numpy()

    home_slots = [s for s, t in teams.items() if t == "home"]
    away_slots = [s for s, t in teams.items() if t == "away"]
    home_xy = extract_team_xy(frames, home_slots)
    away_xy = extract_team_xy(frames, away_slots)

    vis = ball_visible(ball_xy)
    dist = nearest_dist_by_team(ball_xy, home_xy, away_xy)
    speed = per_half(ball_speed, ball_xy, half)
    speed_smooth = per_half(lambda b: ball_speed(smooth_ball(b)), ball_xy, half)

    feats = pd.DataFrame({
        "ball_x": ball_xy[:, 0],
        "ball_y": ball_xy[:, 1],
        "ball_visible": vis,
        "frames_since_ball_seen": per_half(frames_since_ball_seen, vis, half),
        "ball_outside_pitch": ball_outside_pitch(ball_xy),
        "dist_H": dist["H"],
        "dist_A": dist["A"],
        "dist_diff": dist["A"] - dist["H"],          # > 0: bliżej gospodarze
        "n_within3_H": count_within(ball_xy, home_xy, 3.0),
        "n_within3_A": count_within(ball_xy, away_xy, 3.0),
        "n_within5_H": count_within(ball_xy, home_xy, 5.0),
        "n_within5_A": count_within(ball_xy, away_xy, 5.0),
        "n_tracked_H": n_tracked(home_xy),
        "n_tracked_A": n_tracked(away_xy),
        "ball_speed": speed,
        "ball_speed_smooth": speed_smooth,
        "ball_speed_1s": per_half(lambda s: rolling_median(s, 25, 5), speed_smooth, half),
    }, index=frames["video_frame"].to_numpy())
    feats.index.name = "video_frame"
    return feats

# ===MAIN===

if __name__ == "__main__":
    print("Wczytywanie danych meczowych...")
    frames = load_frames("BAR-ATH", "tactical")
    teams = team_of("BAR-ATH", "tactical")

    # 1) ciągłość klatek 
    frame_diffs = frames["video_frame"].diff().dropna()
    print("\n=== TEST CIĄGŁOŚCI KLATEK ===")
    print("video_frame rośnie zawsze o 1:", (frame_diffs == 1).all())

    # 2) wszystkie cechy jednym wywołaniem
    feats = compute_features(frames, teams)
    graded = frames["graded"].to_numpy(dtype=bool)
    g = feats[graded]                      # tylko klatki oceniane

    print("\n=== ODSETEK NaN W KLATKACH OCENIANYCH (%) ===")
    print((g.isna().mean() * 100).round(1).to_string())

    print("\n=== ZAKRESY CECH (min / mediana / max) ===")
    print(g.describe().loc[["min", "50%", "max"]].T.round(2).to_string())

    # 3) porównanie prędkości surowej i wygładzonej (bez limitu)
    ball_xy = frames[["ball_x", "ball_y"]].to_numpy()
    half = frames["half"].to_numpy()
    raw = per_half(lambda b: ball_speed(b, max_speed=np.inf), ball_xy, half)
    smooth = per_half(lambda b: ball_speed(smooth_ball(b), max_speed=np.inf), ball_xy, half)

    def pct_above(v, thr):
        v = v[~np.isnan(v)]
        return (v > thr).mean() * 100

    print("\n=== KLATKI POWYŻEJ PROGU PRĘDKOŚCI (% klatek z prędkością) ===")
    for thr in (35, 40, 60):
        print(f"> {thr} m/s: surowa {pct_above(raw, thr):.3f}%, "
              f"wygładzona {pct_above(smooth, thr):.3f}%")

    # 4) kontrole logiczne: wszystkie muszą wyjść True
    checks = {
        "dystanse nieujemne": bool((g[["dist_H", "dist_A"]].min() >= 0).all()),
        "n_within5 >= n_within3 (H)": bool(((g["n_within5_H"] >= g["n_within3_H"])
                                            | g["n_within3_H"].isna()).all()),
        "n_within5 >= n_within3 (A)": bool(((g["n_within5_A"] >= g["n_within3_A"])
                                            | g["n_within3_A"].isna()).all()),
        "ball_visible zgodne z ball_x": bool((g["ball_visible"].astype(bool)
                                              == g["ball_x"].notna()).all()),
        "dist_diff == dist_A - dist_H": bool(np.allclose(
            g["dist_diff"], g["dist_A"] - g["dist_H"], equal_nan=True)),
    }
    print("\n=== KONTROLE LOGICZNE ===")
    for name, ok in checks.items():
        print(("OK   " if ok else "BŁĄD"), name)