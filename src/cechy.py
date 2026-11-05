import numpy as np
from src.decoder.load import load_frames, team_of


FPS = 25

def extract_team_xy(df, slots):
    # pobieramy na przemian kolumny X i Y dla każdego zawodnika
    cols = [f"{s}_{axis}" for s in slots for axis in ["x", "y"]]
    # przekształcamy na kształt (klatki, zawodnicy, 2 współrzędne)
    return df[cols].to_numpy().reshape(len(df), len(slots), 2)

# funkcja do sprawdzenia czy wiemy gdzie jest piłka
def ball_visible(ball_xy: np.ndarray) -> np.array:
    return ~np.isnan(ball_xy).any(axis=1)

# funkcja określająca liczbę klatek od ostatniej na której było widać piłkę, 0 jeśli widzimy w bieżącej klatce, NaN jeśli jeszcze nie widzieliśmy
def frames_since_ball_seen(visible: np.ndarray) ->np.array:
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

if __name__ == "__main__":
    from src.decoder.load import load_frames, team_of

    print("Wczytywanie danych meczowych...")
    frames = load_frames("BAR-ATH", "tactical")
    teams = team_of("BAR-ATH", "tactical")

    # ---prawdzenie ciągłości video_frame---
    frame_diffs = frames["video_frame"].diff().dropna()
    is_continuous = (frame_diffs == 1).all()
    
    print("\n===TEST CIĄGŁOŚCI KLATEK===")
    print(f"Czy video_frame rośnie zawsze dokładnie o 1? -> {is_continuous}")
    if not is_continuous:
        nierowne = frame_diffs[frame_diffs != 1].unique()
        print(f"Wykryto skoki w numeracji klatek! Różnice w klatkach: {nierowne}")

    # ---Wywołanie funkcji cech na prawdziwych danych---
    ball_xy = frames[["ball_x", "ball_y"]].to_numpy()

    home_slots = [s for s, t in teams.items() if t == "home"]
    away_slots = [s for s, t in teams.items() if t == "away"]

    home_players_xy = extract_team_xy(frames, home_slots)
    away_players_xy = extract_team_xy(frames, away_slots)

    vis = ball_visible(ball_xy)
    since_seen = frames_since_ball_seen(vis)
    dists = nearest_dist_by_team(ball_xy, home_players_xy, away_players_xy)

    # Filtracja klatek ocenianych (graded)
    if "graded" in frames.columns:
        graded_mask = frames["graded"].to_numpy()
        vis_graded = vis[graded_mask]
        
        print("\n===ANALIZA DANYCH GRADED===")
        print(f"Łączna liczba klatek ocenianych (graded): {len(vis_graded)}")
        print(f"Odsetek widocznej piłki w klatkach graded: {vis_graded.mean() * 100:.2f}%")
        print(f"Średni dystans H do piłki: {np.nanmean(dists['H']):.2f} m")
        print(f"Średni dystans A do piłki: {np.nanmean(dists['A']):.2f} m")