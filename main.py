import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.decoder.load import load_frames, team_of
from src.decoder.time_decoder import load_half, smooth_labels
from src.dead_ball.detector import DeadBallDetector
from src.possession.pipeline import predict_possession

N_FRAMES = 199_319
FPS = 25


def stage_load_data(match: str, view: str) -> dict:
    """Stage 1 (olgaerni): Load tracking data."""
    frames = load_frames(match, view)
    teams = team_of(match, view)
    return {"frames": frames, "teams": teams}


def stage_extract_features(data: dict) -> dict:
    """Stage 2 (szcurek21): Extract features from tracking data."""
    frames = data.get("frames")
    teams = data.get("teams")

    if frames is None or teams is None:
        print("No tracking data - features will be empty")
        return {}

    try:
        from src.cechy import compute_features
        features_df = compute_features(frames, teams)

        # Convert DataFrame to dict format expected by dead_ball
        features = {
            "median_speed": features_df.get("ball_speed_smooth"),
            "p90_speed": features_df.get("ball_speed_1s"),
            "ball_visible": features_df.get("ball_visible"),
            "ball_outside_pitch": features_df.get("ball_outside_pitch"),
            "dist_to_home": features_df.get("dist_H"),
            "dist_to_away": features_df.get("dist_A"),
            "frames_since_ball_seen": features_df.get("frames_since_ball_seen"),
        }
        print(f"Extracted {len([v for v in features.values() if v is not None])} features")
        return features
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return {}


def stage_detect_dead_ball(data: dict, features: dict, allow_missing_tags: bool) -> dict:
    """Stage 3 (PSzarkowicz): Detect dead ball intervals."""
    frames = data.get("frames")

    # Try to find smart_tagging.xml
    smart_tagging_path = None
    try:
        data_dir = Path(os.environ.get("MTAH_DATA_DIR", Path.home() / "more-than-a-hack-2027" / "data"))
        match = data.get("match", "BAR-ATH")
        smart_tagging_files = list(data_dir.rglob(f"{match}_tactical_smart_tagging.xml"))
        if smart_tagging_files:
            smart_tagging_path = smart_tagging_files[0]
            print(f"Found smart_tagging.xml: {smart_tagging_path}")
    except Exception:
        pass

    detector = DeadBallDetector(
        smart_tagging_path=smart_tagging_path,
        allow_missing_tags=allow_missing_tags,
    )

    dead_output = detector.predict(frames=frames, features=features)
    return {"dead_output": dead_output}


def stage_classify_possession(data: dict, dead_output) -> dict:
    """Stage 4 (mrudnicki6): Classify live possession."""
    frames = data.get("frames")
    teams = data.get("teams")

    if frames is None:
        return {"possession": None, "dead_mask": np.zeros(N_FRAMES, dtype=bool)}

    # Create dead_mask from dead_ball output
    dead_mask = dead_output.is_dead(threshold=0.5) if dead_output else np.zeros(len(frames), dtype=bool)

    pos = predict_possession(frames, teams=teams, dead_mask=dead_mask)
    return {"possession": pos, "dead_mask": dead_mask}


def stage_temporal_decode(data: dict, dead_mask: np.ndarray) -> np.ndarray:
    """Stage 5 (adriankulig13): Temporal decoding and smoothing."""
    possession = data.get("possession")
    if possession is None:
        return np.full(N_FRAMES, "X", dtype=object)

    # Get raw labels from possession
    raw_labels = possession.frame_labels(dead_mask=dead_mask)

    # Apply temporal smoothing
    half = load_half()
    smoothed = smooth_labels(raw_labels, half)

    return smoothed


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

    # Set seeds for determinism
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Stage 1: Load data
        print("Stage 1: Loading tracking data...")
        data = stage_load_data(args.match, args.view)
        data["match"] = args.match

        # Stage 2: Extract features
        print("Stage 2: Extracting features...")
        features = stage_extract_features(data)

        # Stage 3: Detect dead ball
        print("Stage 3: Detecting dead ball intervals...")
        dead_result = stage_detect_dead_ball(data, features, args.allow_missing_tags)
        dead_output = dead_result.get("dead_output")

        # Stage 4: Classify possession
        print("Stage 4: Classifying live possession...")
        possession_result = stage_classify_possession(data, dead_output)
        dead_mask = possession_result["dead_mask"]

        # Stage 5: Temporal decode
        print("Stage 5: Temporal decoding and smoothing...")
        labels = stage_temporal_decode(possession_result, dead_mask)

        # Create submission DataFrame
        df = pd.DataFrame({"frame": np.arange(N_FRAMES, dtype=np.int64), "label": labels})

    except FileNotFoundError as e:
        print(f"Data not found: {e}")
        print("Running in CI mode with placeholder output")
        df = pd.DataFrame({"frame": np.arange(N_FRAMES, dtype=np.int64), "label": "X"})

    # Save submission
    df.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")

    # Create manifest
    manifest_path = Path(f"{out_path}.manifest.json")
    manifest = {
        "submission": str(out_path),
        "n_frames": len(df),
        "seed": args.seed,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved manifest to {manifest_path}")

    # Print summary
    print("\nSubmission summary:")
    print(f"  Total frames: {len(df)}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
