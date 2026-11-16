# Live Ball Possession Module — Implementation Notes

**Owner:** `mrudnicki6` · **Spec:** Possession Segment v3 ("10/10")
**Location:** `src/possession/` · **Status:** implemented, pending calibration

This module produces per-frame possession evidence for the live-ball task
(`H` = home/Barcelona, `A` = away/Athletic). It emits **evidence only** —
`p_home`, `confidence`, hard labels for live frames, and candidate takeover
boundaries. Dead-ball frames (`D`) belong to the Dead Ball module
(`PSzarkowicz`); final temporal decoding and the `X` policy belong to the
decoder module (`adriankulik13`).

## Pipeline (maps 1:1 to spec §4)

| Stage | File | What it does |
|---|---|---|
| 4.1 Team assignment | `geometry.py` | Slot→team map from match XML (`decoder.load.team_of`); fallback `infer_slot_teams`: k-means anchored at kick-off, propagated frame-by-frame by nearest neighbour |
| 4.2 Metres conversion | `geometry.py` | `detect_coord_scale` auto-detects metres vs normalized `[0,1]²` (loader returns metres; `cechy.py` assumes normalized — this resolves the discrepancy) |
| 4.3 Ball cleaning | `ball.py` | Savitzky–Golay(7, 2), linear interpolation only for gaps ≤ 10 frames, `ball_visible`, `frames_since_seen` |
| 4.4 Emission model | `emissions.py` | `d_H/d_A` = distance to nearest player per team (metres); `s_t = w_vis · clip((d_A − d_H)/τ, ±10)` |
| 4.5 Flight axiom | `emissions.py` | Ball in flight (speed > threshold or aerial proxy) ⇒ `w_vis = 0`, incumbent team retains possession |
| 4.6 Restart re-init | `restarts.py` | Executor team inferred geometrically (nearest player to restart point over first frames; first visible ball position used when ball is NaN at restart); prior added to emissions + decoder init |
| 4.7 Semi-Markov decoder | `decoder.py` | Two-pass per live segment: forward Viterbi with switch cost λ, min-dwell enforcement (spells < `d_min` merged into cheaper neighbour), backward pass, boundary fusion `min(fwd, bwd)` → takeover lands on first-contact frame |
| 4.8 Output | `pipeline.py` | `predict_possession()` → `PossessionResult(p_home, confidence, labels, boundaries, ...)` |

## Key design laws (from the spec)

- **Boundaries, not frames** — `spellF1` rewards correct spell boundaries;
  over-segmentation is catastrophic, so the decoder enforces a hard minimum
  dwell (`d_min` = 30 frames ≈ 1.2 s) and a switch cost λ ≈ 1–2 s of
  sustained evidence.
- **Ball NaN ≠ dead ball** — 41% of graded frames have no ball position;
  `w_vis = 0` holds the incumbent state instead of breaking possession.
- **Determinism** — pure function of inputs, no randomness, byte-reproducible.

## Parameters (`config.py`)

All thresholds are starting points pending the label-free calibration
pass (§7 of the spec): `tau` (proximity scale), `flight_speed`,
`switch_cost`, `min_dwell`, `restart_prior`. None are tuned on the public
5-minute ground truth — sanity floor to beat is `smooth50` = 0.2433 spellF1.

## Interface

```python
from src.possession import predict_possession

result = predict_possession(
    frames,  # pd.DataFrame from decoder.load.load_frames
    teams=teams,  # {slot: 'home'/'away'}; None → kick-off fallback
    dead_mask=dead_mask,  # bool array from Dead Ball module (None = all live)
    segments=segments,  # optional restart hints (restart_xy, confidence, end)
)
result.p_home  # (N,) probability of home possession
result.confidence  # (N,) per-frame reliability
result.labels  # 'H' / 'A' on live frames, 'X' elsewhere
result.boundaries  # [(frame, from, to)] candidate takeovers for the team decoder
result.frame_labels(dead_mask)  # 'H'/'A'/'D'/'X' view (D owned by Dead Ball module)
```

## Known gaps / open items

- `src/contracts.py` does not exist yet — `PossessionResult` is the de-facto contract.
- `smart_tags` parameter is accepted but unused: executor inference is geometric
  (spec §4.6 — geometry wins over tags).
- Optional corroborating evidence (pass continuity, goalkeeper-catch proxy,
  ball imputation) intentionally not implemented — spec requires adding each
  only if label-free diagnostics improve.
- `infer_slot_teams` fallback is O(N·S²); use only when XML team mapping is
  unavailable (spec open question §10.2 — whether ATD slots carry team IDs).
