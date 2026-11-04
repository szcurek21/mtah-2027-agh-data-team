import numpy as np
import pandas as pd

PITCH_L, PITCH_W=105.3, 68.0
FPS = 25

# zmiana podziału współrzędnych Olgi na parę współrzędnych
ball_xy = frames[["ball_x", "ball_y"]].to_numpy()

# wyciągnięcie zawodników do macierzy (N, M, 2)
# gdzie N to liczba klatek, a M to liczba zawodników
home_slots = [s for s, t in teams.items() if t == "home"]
away_slots = [s for s, t in teams.items() if t == "away"]

def extract_team_xy(df, slots):
    # pobieramy na przemian kolumny X i Y dla każdego zawodnika
    cols = [f"{s}_{axis}" for s in slots for axis in ["x", "y"]]
    # przekształcamy na kształt (klatki, zawodnicy, 2 współrzędne)
    return df[cols].to_numpy().reshape(len(df), len(slots), 2)

home_players_xy = extract_team_xy(frames, home_slots)
away_players_xy = extract_team_xy(frames, away_slots)


# funkcja do zmienienia znormalizowanych współrzędnych na metry (0-1 na metry), NaN zostaje NaN
# obsłużone przez Olgę w DANE I EDA
# def to_meters(xy):
#     return xy * np.array([PITCH_L, PITCH_W])

# funkcja do sprawdzenia czy wiemy gdzie jest piłka
def ball_visible(ball_xy: np.array) -> np.array:
    return ~np.isnan(ball_xy).any(axis=1)

# funkcja określająca liczbę klatek od ostatniej na której było widać piłkę, 0 jeśli widzimy w bieżącej klatce, NaN jeśli jeszcze nie widzieliśmy
def frames_since_ball_seen(visible: np.array) ->np.array:
    out=np.full(len(visible), np.nan)
    last= None
    for i, v in enumerate(visible):
        if v:
            last=i
        if last is not None:
            out[i]= i-last
    return out

# dystans piłki od najbliższego zawodnika jednej i drugiej drużyny zwraca {'H': (N,), 'A': (N,)}; Nan gdy brak piłki lub brak graczy drużyny
# def nearest_dist_by_team(ball_xy, players_xy, team):
#     d = np.linalg.norm(players_xy - ball_xy[:, None, :], axis=2)
#     out = {}
#     for t in ("H", "A"):
#         dt = d[:, team == t]
#         all_nan = np.isnan(dt).all(axis=1)
#         m = np.where(np.isnan(dt), np.inf, dt).min(axis=1)
#         m[all_nan] = np.nan
#         out[t] = m
#     return out

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