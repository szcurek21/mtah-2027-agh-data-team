import numpy as np
from kloppy import sportscode

from src.decoder.load import DATA_DIR, FPS, HALF_BLOCKS, _clean, load_dataset

MATCH = "BAR-ATH"
CODES = {"BALL IN FINAL THIRD", "BALL IN THE BOX"}

raw = _clean(load_dataset(MATCH, "tactical").to_df())
ball_x = raw.set_index("video_frame")["ball_x"]

tags_path = next(DATA_DIR.rglob(f"{MATCH}_tactical_smart_tagging.xml"))
tags = sportscode.load(tags_path).to_df()
tags = tags[tags["code"].str.upper().isin(CODES)]

for half, first, last in HALF_BLOCKS[MATCH]:
    for team in ["Barcelona", "Atletic Club"]:
        rows = tags[tags["Team"].astype(str).str.lower() == team.lower()]
        xs = []
        for _, r in rows.iterrows():
            a = round(r["timestamp"].total_seconds() * FPS)
            b = round(r["end_timestamp"].total_seconds() * FPS)
            if first <= a <= last:
                xs.append(ball_x.loc[a:b].dropna().to_numpy())
        xs = np.concatenate(xs) if xs else np.array([])
        if len(xs) == 0:
            print(f"Połowa {half}, {team}: brak tagów")
            continue
        med = np.median(xs)
        side = "PRAWO (+x)" if med > 0 else "LEWO (-x)"
        print(f"Połowa {half}, {team}: {len(xs)} klatek z piłką, "
              f"mediana ball_x = {med:+.1f} m  ->  atakuje w {side}")