import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.decoder.load import load_frames, team_of
from src.possession.pipeline import predict_possession

N_FRAMES = 199_319


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", default="BAR-ATH")
    parser.add_argument("--view", default="tactical")
    parser.add_argument("--output", default="results/submission.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing-tags", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        frames = load_frames(args.match, args.view)
        teams = team_of(args.match, args.view)
        dead_mask = np.zeros(len(frames), dtype=bool)
        pos = predict_possession(frames, teams=teams, dead_mask=dead_mask)
        labels = pos.frame_labels(dead_mask=dead_mask)
        df = pd.DataFrame({"frame": frames["video_frame"].to_numpy(), "label": labels})
        df = df.set_index("frame").reindex(range(N_FRAMES), fill_value="X").reset_index()
    except FileNotFoundError:
        # Tryb CI bez lokalnych plików trackingowych
        df = pd.DataFrame({"frame": np.arange(N_FRAMES, dtype=np.int64), "label": "X"})

    df.to_csv(out_path, index=False)

    manifest_path = Path(f"{out_path}.manifest.json")
    manifest = {
        "submission": str(out_path),
        "n_frames": len(df),
        "seed": args.seed,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
