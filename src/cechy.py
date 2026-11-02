import numpy as np

PITCH_L, PITCH_W=105.3, 68.0
FPS = 25

# funkcja do zmienienia znormalizowanych współrzędnych na metry (0-1 na metry), NaN zostaje NaN
def to_meters(xy):
    return xy * np.array([PITCH_L, PITCH_W])

# funkcja do sprawdzenia czy wiemy gdzie jest piłka
def ball_visible(ball_xy):
    return ~np.isnan(ball_xy).any(axis=1)

# funkcja określająca liczbę klatek od ostatniej na której było widać piłkę, 0 jeśli widzimy w bieżącej klatce, NaN jeśli jeszcze nie widzieliśmy
def frames_since_ball_seen(visible):
    out=np.full(len(visible), np.nan)
    last= None
    for i, v in enumerate(visible):
        if v:
            last=i
        if last is not None:
            out[i]= i-last
    return out

# dystans piłki od najbliższego zawodnika jednej i drugiej drużyny zwraca {'H': (N,), 'A': (N,)}; Nan gdy brak piłki lub brak graczy drużyny
def nearest_dist_by_team(ball_xy, players_xy, team):
    d = np.linalg.norm(players_xy - ball_xy[:, None, :], axis=2)
    out = {}
    for t in ("H", "A"):
        dt = d[:, team == t]
        all_nan = np.isnan(dt).all(axis=1)
        m = np.where(np.isnan(dt), np.inf, dt).min(axis=1)
        m[all_nan] = np.nan
        out[t] = m
    return out