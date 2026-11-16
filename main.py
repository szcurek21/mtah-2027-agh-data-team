"""Main execution pipeline for AGH Data Team: More than a Hack 2027.

Orchestrates data loading, feature extraction, dead ball detection,
live possession classification, temporal decoding, and submission generation.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

from src.dead_ball.contract import (
    FIRST_HALF_END,
    FIRST_HALF_START,
    N_FRAMES,
    SECOND_HALF_END,
    SECOND_HALF_START,
    DeadBallOutput,
)
from src.dead_ball.detector import DeadBallDetector
from src.utils.cache import cache_stage, clear_all_cache, load_cached_stage
from src.utils.manifest import write_run_manifest
from src.utils.validation import validate_submission_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AGH Data Team - Ball Possession Submission Generator"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/submission.csv"),
        help="Path where the output submission CSV will be saved",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip automatic validation check",
    )
    parser.add_argument(
        "--allow-missing-tags",
        action="store_true",
        help="Allow running without smart_tagging.xml (structure-only D). Do not use for submission.",
    )
    parser.add_argument(
        "--smart-tagging",
        type=Path,
        default=None,
        help="Path to tactical smart_tagging.xml",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached intermediate results before running",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching for this run (compute everything fresh)",
    )
    return parser.parse_args()


def stage_load_data() -> dict:
    """Stage 1 (olgaerni): Load tracking frames in metres, origin at centre."""
    from src.decoder.load import DATA_DIR, load_frames, team_of

    frames = None
    teams = {}
    try:
        frames = load_frames("BAR-ATH")
        teams = team_of("BAR-ATH")
    except Exception as exc:
        print(f"Tracking data not loaded ({exc}). Proceeding without tracking frames.")

    tag_paths = sorted(DATA_DIR.rglob("*smart_tagging.xml"))
    if not tag_paths:
        tag_paths = sorted(Path("data").rglob("*smart_tagging.xml"))

    return {
        "n_frames": N_FRAMES,
        "first_half": (FIRST_HALF_START, FIRST_HALF_END),
        "second_half": (SECOND_HALF_START, SECOND_HALF_END),
        "frames": frames,
        "teams": teams,
        "smart_tagging_path": tag_paths[0] if tag_paths else None,
        "data_dir": DATA_DIR,
    }


def stage_extract_features(data: dict) -> dict:
    """Stage 2 (szcurek21): Extract features from tracking data."""
    frames = data.get("frames")
    teams = data.get("teams")

    if frames is None or teams is None:
        print("No tracking data - features will be empty")
        return {}

    from src.cechy import compute_features

    features_df = compute_features(frames, teams)

    # Convert DataFrame to dict format expected by dead_ball module
    # Use szcurek21's smoothed speed as median_speed
    # Use szcurek21's 1s rolling median as p90_speed
    return {
        "median_speed": features_df.get("ball_speed_smooth"),
        "p90_speed": features_df.get("ball_speed_1s"),
        # Additional features that could be used in evidence computation:
        "ball_visible": features_df.get("ball_visible"),
        "ball_outside_pitch": features_df.get("ball_outside_pitch"),
        "dist_to_home": features_df.get("dist_H"),
        "dist_to_away": features_df.get("dist_A"),
    }


def stage_dead_ball(
    data: dict,
    features: dict,
    *,
    allow_missing_tags: bool = False,
    smart_tagging_path: Path | None = None,
    use_cache: bool = True,
) -> DeadBallOutput:
    """Stage 3 (PSzarkowicz): Detect out-of-play intervals and structure."""
    tag_path = smart_tagging_path or data.get("smart_tagging_path")

    # Cache key based on input configuration
    cache_config = {
        "tag_path": str(tag_path) if tag_path else None,
        "allow_missing_tags": allow_missing_tags,
        "n_frames": data.get("n_frames"),
    }

    if use_cache:
        cached = load_cached_stage("dead_ball", cache_config)
        if cached is not None:
            print("Loaded dead ball output from cache")
            return cached

    detector = DeadBallDetector(
        smart_tagging_path=tag_path,
        allow_missing_tags=allow_missing_tags,
    )
    result = detector.predict(frames=data.get("frames"), features=features)

    if use_cache:
        cache_stage("dead_ball", cache_config, result)

    return result


def stage_live_possession(data: dict, features: dict, dead_ball_out: DeadBallOutput) -> np.ndarray:
    """Stage 4 (mrudnicki6 - Possession Logic): Classify H vs A for live frames."""
    possession_labels = np.full(N_FRAMES, "H", dtype="<U1")
    possession_labels[SECOND_HALF_START : SECOND_HALF_END + 1] = "A"
    return possession_labels


def stage_temporal_decoder(
    dead_ball_out: DeadBallOutput, live_possession: np.ndarray
) -> np.ndarray:
    """Stage 5 (adriankulig13 - Temporal Decoder): Fuse and smooth final sequence."""
    final_labels = np.empty(N_FRAMES, dtype="<U1")

    final_labels[dead_ball_out.non_play] = "D"

    is_live = ~dead_ball_out.is_dead(threshold=0.5)
    final_labels[is_live] = live_possession[is_live]

    is_dead_in_game = dead_ball_out.is_dead(threshold=0.5) & (~dead_ball_out.non_play)
    final_labels[is_dead_in_game] = "D"

    return final_labels


def save_submission(output_path: Path, labels: np.ndarray) -> None:
    """Writes the submission CSV cleanly with exact official formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("frame,label\n")
        for frame_idx, label in enumerate(labels):
            f.write(f"{frame_idx},{label}\n")


def run_pipeline(
    output_path: Path,
    seed: int = 42,
    skip_validation: bool = False,
    allow_missing_tags: bool = False,
    smart_tagging_path: Path | None = None,
    clear_cache: bool = False,
    no_cache: bool = False,
) -> None:
    t_start = time.perf_counter()
    print("=" * 60)
    print("AGH Data Team: Ball Possession Pipeline")
    print(f"Random seed: {seed} | Target output: {output_path}")
    print("=" * 60)

    # Set PYTHONHASHSEED for determinism before any imports
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if clear_cache:
        n_deleted = clear_all_cache()
        print(f"Cleared {n_deleted} cached files")

    print("[1/5] Loading data...")
    data = stage_load_data()

    print("[2/5] Extracting features...")
    features = stage_extract_features(data)

    print("[3/5] Detecting dead ball (PSzarkowicz)...")
    dead_ball_out = stage_dead_ball(
        data,
        features,
        allow_missing_tags=allow_missing_tags,
        smart_tagging_path=smart_tagging_path,
        use_cache=not no_cache,
    )

    print("[4/5] Classifying live possession (mrudnicki6)...")
    live_possession = stage_live_possession(data, features, dead_ball_out)

    print("[5/5] Temporal decoding & fusion (adriankulig13)...")
    final_labels = stage_temporal_decoder(dead_ball_out, live_possession)

    print(f"Writing submission file to {output_path}...")
    save_submission(output_path, final_labels)

    csv_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"submission SHA-256: {csv_hash}")
    write_run_manifest(
        output_path,
        seed=seed,
        extra={
            "n_dead_segments": len(dead_ball_out.segments),
            "n_restart_frames": int(len(dead_ball_out.restart_frames)),
            "mean_p_dead": float(dead_ball_out.p_dead.mean()),
        },
    )

    elapsed = time.perf_counter() - t_start
    print(f"Pipeline finished in {elapsed:.2f}s")

    if not skip_validation:
        print("\nRunning official submission validator...")
        validate_submission_file(output_path)


def main() -> None:
    args = parse_args()
    try:
        run_pipeline(
            output_path=args.output,
            seed=args.seed,
            skip_validation=args.skip_validation,
            allow_missing_tags=args.allow_missing_tags,
            smart_tagging_path=args.smart_tagging,
            clear_cache=args.clear_cache,
            no_cache=args.no_cache,
        )
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
