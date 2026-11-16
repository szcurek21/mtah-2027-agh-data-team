import matplotlib.pyplot as plt

from src.decoder.load import load_frames, team_of

frames = load_frames("BAR-ATH")
teams = team_of("BAR-ATH")

# 1) jak wygląda tabelka
print(frames[["video_frame", "t", "half", "graded", "ball_x", "ball_y"]].head(10))

# wybierałam moment: 10. minuta pierwszej połowy
start = 11185 + 10 * 60 * 25  # początek połowy + 10 min (25 klatek = 1 s)
minute = frames[frames["video_frame"].between(start, start + 60 * 25)]
now = minute.iloc[0]

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.plot(minute["ball_x"], minute["ball_y"], color="gray", lw=1, label="piłka przez 1 min")
for slot, team in teams.items():
    x, y = now.get(f"{slot}_x"), now.get(f"{slot}_y")
    if x == x:  # pomija NaN
        ax.scatter(x, y, c="tab:blue" if team == "home" else "tab:red", s=60)
ax.set_xlim(-52.5, 52.5)
ax.set_ylim(-34, 34)
ax.axvline(0, color="lightgray")
ax.set_title("Niebiescy = Barcelona (atakuje →), czerwoni = Athletic")
ax.legend()
plt.savefig("results/demo.png")
print("Zapisano results/demo.png")
