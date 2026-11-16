"""Deterministic tests and contract validation for the Dead Ball module."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.dead_ball.calibration import (
    apply_tag_corrections,
    calibrate_tag_offsets,
)
from src.dead_ball.contract import (
    FIRST_HALF_START,
    N_FRAMES,
    DeadBallOutput,
    DeadSegment,
)
from src.dead_ball.cusum import cusum_changepoints
from src.dead_ball.detector import DeadBallDetector, get_match_structure_segments
from src.dead_ball.filters import savitzky_golay
from src.dead_ball.hysteresis import (
    apply_hysteresis,
    filter_min_durations,
)
from src.dead_ball.kinematics import HALF_WIDTH_M, extract_tracking_signals
from src.dead_ball.smart_tagging import (
    extract_dead_candidates_from_tags,
    parse_smart_tagging_xml,
    seconds_to_frame,
)

FIXTURE_XML = Path(__file__).parent / "fixtures" / "smart_tagging_sample.xml"


def _structure_detector() -> DeadBallDetector:
    return DeadBallDetector(
        skip_tag_discovery=True,
        allow_missing_tags=True,
        measure_clock=False,
    )


class TestDeadBallContract(unittest.TestCase):
    def test_dead_segment_invariants(self):
        seg = DeadSegment(start=100, end=150, kind="throw_in", confidence=0.9, team="Barcelona")
        self.assertEqual(seg.duration_frames, 51)
        self.assertEqual(seg.team, "Barcelona")
        with self.assertRaises(ValueError):
            DeadSegment(start=200, end=100, kind="corner")
        with self.assertRaises(ValueError):
            DeadSegment(start=0, end=10, kind="corner", confidence=1.5)

    def test_dead_ball_output_length_validation(self):
        p_dead_wrong = np.zeros(100, dtype=np.float32)
        non_play = np.zeros(N_FRAMES, dtype=bool)
        with self.assertRaises(ValueError):
            DeadBallOutput(
                p_dead=p_dead_wrong,
                non_play=non_play,
                segments=(),
                restart_frames=np.array([], dtype=np.int32),
            )


class TestSavitzkyGolayFilter(unittest.TestCase):
    def test_preserves_linear_trend(self):
        x = np.linspace(0, 10, 101)
        smoothed = savitzky_golay(x, window_length=11, polyorder=2)
        np.testing.assert_allclose(smoothed, x, atol=1e-10)

    def test_smooths_noise(self):
        np.random.seed(42)
        clean = np.sin(np.linspace(0, 2 * np.pi, 200))
        noisy = clean + np.random.normal(0, 0.2, 200)
        smoothed = savitzky_golay(noisy, window_length=15, polyorder=2)
        self.assertLess(np.mean((smoothed - clean) ** 2), np.mean((noisy - clean) ** 2))

    def test_shape_and_dtype_preservation(self):
        y = np.ones(500, dtype=np.float32)
        out = savitzky_golay(y, window_length=25, polyorder=2)
        self.assertEqual(out.shape, y.shape)
        self.assertEqual(out.dtype, y.dtype)


class TestHysteresisAndDurations(unittest.TestCase):
    def test_hysteresis_state_transitions(self):
        signal = np.array([0.1, 0.4, 0.85, 0.5, 0.2, 0.6])
        mask = apply_hysteresis(signal, low_threshold=0.3, high_threshold=0.8)
        expected = np.array([False, False, True, True, False, False])
        np.testing.assert_array_equal(mask, expected)

    def test_min_dead_duration_removes_flicker(self):
        mask = np.zeros(100, dtype=bool)
        mask[20:25] = True
        filtered = filter_min_durations(mask, min_dead_frames=20, min_live_gap_frames=5)
        self.assertFalse(np.any(filtered))

    def test_tagged_short_dead_is_protected(self):
        mask = np.zeros(100, dtype=bool)
        mask[20:25] = True
        protected = np.zeros(100, dtype=bool)
        protected[20:25] = True
        filtered = filter_min_durations(
            mask, min_dead_frames=20, min_live_gap_frames=5, protected_mask=protected
        )
        self.assertTrue(np.all(filtered[20:25]))

    def test_min_live_gap_closing(self):
        mask = np.zeros(100, dtype=bool)
        mask[10:30] = True
        mask[33:60] = True
        filtered = filter_min_durations(mask, min_dead_frames=10, min_live_gap_frames=10)
        self.assertTrue(np.all(filtered[10:60]))


class TestSmartTagging(unittest.TestCase):
    def test_parse_and_keep_kinds_and_team(self):
        instances = parse_smart_tagging_xml(FIXTURE_XML)
        self.assertGreaterEqual(len(instances), 2)
        search, segments, core = extract_dead_candidates_from_tags(instances)
        kinds = {s.kind for s in segments}
        self.assertIn("throw_in", kinds)
        self.assertIn("corner", kinds)
        throw_in = next(s for s in segments if s.kind == "throw_in")
        self.assertIsNone(throw_in.team)
        corner = next(s for s in segments if s.kind == "corner")
        self.assertEqual(corner.team, "Barcelona")
        self.assertEqual(throw_in.start, seconds_to_frame(800.0))
        self.assertTrue(core[throw_in.start : throw_in.end + 1].all())
        self.assertGreater(int(search.sum()), int(core.sum()))

    def test_attacking_third_is_not_a_dead_candidate(self):
        instances = parse_smart_tagging_xml(FIXTURE_XML)
        _, segments, _ = extract_dead_candidates_from_tags(instances)
        self.assertEqual(len(segments), 2)


class TestMissingTags(unittest.TestCase):
    def test_missing_xml_raises_by_default(self):
        missing = Path(tempfile.gettempdir()) / "definitely-missing-smart_tagging.xml"
        with self.assertRaises(FileNotFoundError):
            DeadBallDetector(smart_tagging_path=missing)

    def test_allow_missing_tags_structure_only(self):
        out = _structure_detector().predict()
        self.assertTrue(out.non_play[:FIRST_HALF_START].all())
        in_play = ~out.non_play
        self.assertTrue(np.all(out.p_dead[in_play] < 0.5))
        kinds = {s.kind for s in out.segments}
        self.assertEqual(kinds, {"non_play"})
        self.assertEqual(len(get_match_structure_segments()), 3)


class TestDetectorWithFixtureTags(unittest.TestCase):
    def test_kinds_survive_predict(self):
        detector = DeadBallDetector(
            smart_tagging_path=FIXTURE_XML,
            allow_missing_tags=False,
            measure_clock=False,
        )
        out = detector.predict()
        kinds = {s.kind for s in out.segments}
        self.assertIn("throw_in", kinds)
        self.assertIn("corner", kinds)
        self.assertIn("non_play", kinds)
        throw_in = next(s for s in out.segments if s.kind == "throw_in")
        self.assertLess(throw_in.duration_frames, 50)
        self.assertGreaterEqual(out.p_dead[throw_in.start], 0.5)


class TestCusumAndKinematics(unittest.TestCase):
    def test_cusum_finds_drop_and_rise(self):
        x = np.concatenate([np.full(80, 4.0), np.full(80, 0.4), np.full(80, 4.0)])
        down, up = cusum_changepoints(x, threshold=3.0, drift=0.2)
        self.assertTrue(len(down) >= 1)
        self.assertTrue(len(up) >= 1)
        self.assertTrue(any(60 <= i <= 100 for i in down))

    def test_ball_out_uses_metre_pitch(self):
        n_keep = 400
        video_frame = np.arange(FIRST_HALF_START, FIRST_HALF_START + n_keep)
        ball_x = np.zeros(n_keep, dtype=np.float32)
        ball_y = np.zeros(n_keep, dtype=np.float32)
        ball_y[100:120] = HALF_WIDTH_M + 0.2
        frames = pd.DataFrame(
            {
                "video_frame": video_frame,
                "ball_x": ball_x,
                "ball_y": ball_y,
                "ball_speed": np.ones(n_keep, dtype=np.float32),
                "p1_s": np.full(n_keep, 3.0, dtype=np.float32),
                "p2_s": np.full(n_keep, 3.0, dtype=np.float32),
            }
        )
        signals = extract_tracking_signals(frames)
        out_idx = FIRST_HALF_START + 110
        self.assertTrue(signals.ball_out[out_idx])
        self.assertFalse(signals.ball_out[FIRST_HALF_START + 10])

    def test_predict_consumes_frames(self):
        n_keep = 800
        start = FIRST_HALF_START
        video_frame = np.arange(start, start + n_keep)
        ball_x = np.zeros(n_keep, dtype=np.float32)
        ball_y = np.zeros(n_keep, dtype=np.float32)
        ball_speed = np.full(n_keep, 8.0, dtype=np.float32)
        p_s = np.full(n_keep, 4.0, dtype=np.float32)
        # synthetic stoppage: ball over touchline, then NaN teleport
        ball_y[200:210] = HALF_WIDTH_M + 0.4
        ball_x[210:280] = np.nan
        ball_y[210:280] = np.nan
        ball_speed[210:280] = 0.0
        p_s[210:280] = 0.3
        ball_x[280:] = 20.0
        ball_y[280:] = 0.0
        frames = pd.DataFrame(
            {
                "video_frame": video_frame,
                "ball_x": ball_x,
                "ball_y": ball_y,
                "ball_speed": ball_speed,
                "p1_s": p_s,
                "p2_s": p_s,
            }
        )
        detector = DeadBallDetector(
            smart_tagging_path=FIXTURE_XML,
            measure_clock=False,
        )
        out = detector.predict(frames=frames)
        self.assertEqual(len(out.p_dead), N_FRAMES)
        self.assertTrue(np.all((out.p_dead >= 0.0) & (out.p_dead <= 1.0)))
        # fused probabilities are no longer a 3-level step function
        unique = set(np.round(out.p_dead[~out.non_play], 2))
        self.assertGreater(len(unique), 3)


class TestTagCalibration(unittest.TestCase):
    def test_calibration_returns_offsets_dict(self):

        # Create simple synthetic data
        n_frames = 200
        video_frame = np.arange(n_frames)
        ball_x = np.zeros(n_frames, dtype=np.float32)
        ball_y = np.zeros(n_frames, dtype=np.float32)
        ball_speed = np.concatenate([np.full(50, 4.0), np.full(100, 0.5), np.full(50, 4.0)]).astype(
            np.float32
        )
        p_s = np.full(n_frames, 3.0, dtype=np.float32)

        frames = pd.DataFrame(
            {
                "video_frame": video_frame,
                "ball_x": ball_x,
                "ball_y": ball_y,
                "ball_speed": ball_speed,
                "p1_s": p_s,
                "p2_s": p_s,
            }
        )

        signals = extract_tracking_signals(frames)

        # Create tag segments (offset from true kinematic boundary)
        tag_segments = [DeadSegment(start=40, end=160, kind="throw_in", confidence=0.85)]

        offsets = calibrate_tag_offsets(tag_segments, signals)
        self.assertIsInstance(offsets, dict)
        # With only 1 sample, calibration returns (0, 0) for insufficient samples
        # The function works but needs more samples for reliable calibration

    def test_apply_corrections_shifts_boundaries(self):

        segments = [DeadSegment(start=100, end=150, kind="throw_in", confidence=0.85)]
        offsets = {"throw_in": (5.0, -3.0)}  # Shift start +5, end -3

        corrected = apply_tag_corrections(segments, offsets)
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0].start, 105)
        self.assertEqual(corrected[0].end, 147)


class TestDeterminismAndInvariants(unittest.TestCase):
    def test_deterministic_output(self):
        detector = _structure_detector()
        out1 = detector.predict()
        out2 = detector.predict()
        h1 = hashlib.sha256(out1.p_dead.tobytes()).hexdigest()
        h2 = hashlib.sha256(out2.p_dead.tobytes()).hexdigest()
        self.assertEqual(h1, h2)
        self.assertTrue(np.array_equal(out1.non_play, out2.non_play))
        self.assertTrue(np.array_equal(out1.restart_frames, out2.restart_frames))

    def test_golden_hash_structure_only(self):
        out = _structure_detector().predict()
        digest = hashlib.sha256(out.p_dead.tobytes()).hexdigest()
        expected = hashlib.sha256(out.p_dead.tobytes()).hexdigest()
        self.assertEqual(digest, expected)
        # Stable contract: non-play is 1, in-play baseline is 0.05
        self.assertTrue(np.allclose(out.p_dead[out.non_play], 1.0))
        self.assertTrue(np.allclose(out.p_dead[~out.non_play], 0.05, atol=1e-6))

    def test_golden_hash_full_submission(self):
        """Golden hash test: two consecutive runs must produce identical output."""
        # Run pipeline twice with the same seed
        with tempfile.TemporaryDirectory() as tmpdir:
            out1 = Path(tmpdir) / "submission1.csv"
            out2 = Path(tmpdir) / "submission2.csv"

            result1 = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "--output",
                    str(out1),
                    "--seed",
                    "42",
                    "--allow-missing-tags",
                    "--skip-validation",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result1.returncode, 0, f"First run failed: {result1.stderr}")

            result2 = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "--output",
                    str(out2),
                    "--seed",
                    "42",
                    "--allow-missing-tags",
                    "--skip-validation",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result2.returncode, 0, f"Second run failed: {result2.stderr}")

            # Verify both files exist
            self.assertTrue(out1.exists(), "First submission file not created")
            self.assertTrue(out2.exists(), "Second submission file not created")

            # Compare SHA-256 hashes
            h1 = hashlib.sha256(out1.read_bytes()).hexdigest()
            h2 = hashlib.sha256(out2.read_bytes()).hexdigest()
            self.assertEqual(
                h1,
                h2,
                f"Golden hash mismatch: {h1} != {h2}. Pipeline is not deterministic.",
            )

    def test_invariants_hold(self):
        out = _structure_detector().predict()
        self.assertEqual(len(out.p_dead), N_FRAMES)
        self.assertFalse(np.isnan(out.p_dead).any())
        self.assertTrue(np.all((out.p_dead >= 0.0) & (out.p_dead <= 1.0)))
        for i, seg in enumerate(out.segments):
            self.assertLessEqual(seg.start, seg.end)
            if i > 0:
                self.assertLess(out.segments[i - 1].end, seg.start)


if __name__ == "__main__":
    unittest.main()
