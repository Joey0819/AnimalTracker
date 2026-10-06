# MART-GRAPH Execution Loop — 2026-07-28

This document records the autonomous development-loop iterations executed on this
repository on 2026-07-28, with exact baselines, files changed, verification
commands, real-video evidence, and the remaining blocker.

## Repository Audit (before changes)

Verified by reading the code, not assumed:

- DenseTrack-AI already ships detector/tracker orchestration, Norfair reuse,
  `mac_adaptive` fixed-count bidirectional tracking, identity-stable full-video
  smoothing, long-term closed-set stitching, hierarchical discrete geometry,
  per-frame `identity_risk.json/csv`, sparse GT review/evaluation with a
  reviewed-GT guard, behavior analysis, exports, CLI, Web UI, and many 4xPD runs.
- Contact handling inside the tracker existed only as **aggregate counters**
  (`contact_deferred_assignments`, `merged_detection_rejections` in
  `identity_continuity.json` diagnostics) and scalar per-frame risk scores.
  There were **no per-event merge/split records, no top-K incoming→outgoing
  identity hypotheses, no best-vs-second-best margins, and no review-facing
  uncertainty records**. This confirmed the MART-GRAPH contact-event layer as
  the highest-value reachable gap; nothing was duplicated.
- `results/4xPD_identity_guard_final_20260728` was the strongest existing run
  (identity stability 0.708, observed 2.78/4 tracks/frame).
- This folder is **not a Git repository** (`git status` fails; no `.git`).
  Per instructions, no `git init`, no commits, and no Git finish helper were
  run. All changes are plain file edits; all new artifacts went into a new
  timestamped results directory. No existing result folders were modified.

## Baseline

- `./.venv/bin/python tests/test_functional.py` → **32 passed, 0 failed, 32 total**.

## Iteration 1 — Contact-event layer

Implemented the auditable contact-event layer end to end.

### Files changed

| File | Change |
|---|---|
| `densetrack/tracking/contact_events.py` | **New module**: per-frame contact grouping (IoU/centroid union-find), temporal event chaining with 1-frame flicker tolerance, merged-measurement detection (predicted participant or center-containment + IoU), split-boundary top-K hypothesis enumeration over incoming→outgoing assignments with motion-continuity (median pre-event velocity, √gap noise growth) + log-area cost, softmax relative scores, best-vs-second margin, confident/ambiguous/uncertain/unresolved tiers, tracker-agreement flag, review priorities, `contact_events.json/csv` writers |
| `densetrack/config.py` | New `ContactEventConfig` (enabled, contact_distance_px=75, iou_threshold, merge_iou, recovery_window=30, top_k=3, max_hypothesis_participants=8, velocity_window=7, motion_noise_px=12, size_weight, hypothesis_temperature, confident/ambiguous margins) wired into `PipelineConfig.contact_events` |
| `densetrack/pipeline.py` | Step 5 writes contact-event artifacts after global optimization when enabled; logs summary |
| `densetrack/cli.py` | New `densetrack contact-events <run_dir>` command: recomputes the layer from immutable `tracks.mot` + `labels.sleap.json` provenance, optional separate `--output`, dashboard refresh when writing in place |
| `densetrack/reporting/static_dashboard.py` | New bilingual `接触事件 / Contact events` section (uncertainty metrics, event table with best hypothesis, margins, priorities, status); contact artifacts added to downloads |
| `densetrack/validation/review_package.py` | Loads `contact_events.json` frame priorities; `_score_frame` adds a 0.15 contact-uncertainty term (weights renormalized: 0.30 prediction + 0.15 crowding + 0.10 low-confidence + 0.10 count-gap + 0.20 identity-risk + 0.15 contact); `contact_event_score` field on `ReviewFrame`; manifest risk-signal list extended |
| `tests/test_functional.py` | New test 33 `test_contact_event_layer`; complete-workflow test asserts contact artifacts and dashboard section |

### Verification

- Functional suite: **33 passed, 0 failed** (32 baseline + 1 new; complete-workflow test extended).
- Synthetic deterministic scenarios verified: motion-continuous crossing →
  confident, identity best, margin 0.999; bounce-back → best hypothesis =
  swap, `tracker_matches_best=False`; symmetric perpendicular exit →
  uncertain, margin 0.0; no-contact → zero events.

## Iteration 2 — Conservative audit hardening (evidence-driven)

The first real 4xPD audit produced 355 events and 135 tracker-vs-best
disagreements. Failure analysis:

- 77/355 events had **zero merged frames** (proximity-only). Their
  disagreement rate was 6.5% (5/77) vs 51.8% (130/251 resolved) for
  measurement-ambiguous events — so proximity-only events were *not* the
  disagreement noise source, but they were indistinguishable in the output.
- Temporal clustering showed the 123 confident disagreements collapse into a
  small number of long spans: once two labels are inverted (true switch or a
  global smoothing permutation — the smoother changed 4398 assignments), every
  later contact between the same animals scores swap-best until labels
  re-align. **The raw count 135 overstates independent identity problems by
  ~7×.** That was the biggest provable defect of iteration 1.

Changes (additive, backwards compatible):

- `ContactEvent.event_kind`: `measurement_ambiguous` vs `proximity_only`.
- `_build_inversion_episodes`: chains disagreement events with overlapping
  participants and ≤ `episode_gap_frames` (default 120) gaps into
  label-inversion episodes — the honest review unit. Documented as a chaining
  heuristic, not a switch count.
- `contact_events.json` format_version 1.1: `inversion_episodes` array,
  `event_kind_counts`, `inversion_episode_count`, `cascade_caveat` in summary.
- CSV gains `event_kind`; dashboard gains kind column, kind-split metrics and
  the inversion-episode headline; CLI prints episodes vs raw disagreements.
- Tests: episode chaining (3 events → 2 episodes), kind classification,
  proximity-only scenario, dashboard assertions.

### Before/after on the identical immutable input (`results/4xPD_identity_guard_final_20260728`)

| Metric | v1 (before) | v1.1 (after) |
|---|---:|---:|
| Total contact events | 355 | 355 (detection unchanged) |
| Event kind split | n/a | 278 measurement-ambiguous / 77 proximity-only |
| Confident / ambiguous / uncertain / unresolved | 312 / 5 / 11 / 27 | 312 / 5 / 11 / 27 |
| Raw tracker-vs-best disagreements | 135 | 135 (signal preserved) |
| Review unit reported | 135 raw events | **18 inversion episodes** |
| Frames carrying review priority | 6262 | 6262 |

## Real-video evidence (4xPD.mp4, 9363 frames, 4 mice)

Artifacts root: `results/claude_mart_graph_loop_20260728_224813/`

| Path | Contents |
|---|---|
| `contact_audit_of_identity_guard_final/` | v1 audit of strongest prior run (immutable input) |
| `contact_audit_v2_of_identity_guard_final/` | v1.1 audit, identical input — before/after comparison above |
| `full_run_v2/` | Fresh full `densetrack complete` run exercising the shipped path end to end |

Commands:

```bash
./.venv/bin/python -m densetrack.cli contact-events \
  results/4xPD_identity_guard_final_20260728 \
  --output results/claude_mart_graph_loop_20260728_224813/contact_audit_v2_of_identity_guard_final \
  --no-refresh-dashboard

./.venv/bin/python -m densetrack.cli complete 4xPD.mp4 \
  --config configs/4xPD_mac_adaptive.yaml --n-animals 4 \
  --validation-frames 100 \
  --output results/claude_mart_graph_loop_20260728_224813/full_run_v2
```

Full-run verification:

- Total pipeline 146.7 s (tracking 85.5 s) — consistent with the documented
  2026-07-28 single-machine benchmark (142.9 s / 83.5 s); not a performance claim.
- Tracking behavior unchanged: 2.777 observed tracks/frame, identity stability
  0.708, same algorithm string — the layer is additive, no regression.
- Contact events from the fresh pipeline run are **identical** to the
  immutable-input recompute (355 events, 18 episodes) — determinism evidence.
- `dashboard.html` renders the bilingual contact-event section including
  inversion-episode headline and kind split.
- Sparse review package: 87/100 selected frames carry contact-event priority;
  manifest risk signals include the contact-event signal.

## Honest interpretation and limitations

- The 135 raw disagreements / 18 inversion episodes are **review targets, not
  measured ID switches**. Hypothesis probabilities are annotation-free
  relative scores from a fixed documented scoring function; they are not
  GT-calibrated and say nothing by themselves about identity accuracy.
- Episode chaining is a heuristic (participant overlap + 120-frame gap); it
  cannot distinguish one persistent inversion from a re-swap inside a chain.
- Proximity-only events (77) with full observation can still catch
  swaps-under-observation (5 did), but their confidence is near-trivially high
  because the tracker consumed real measurements; they are labeled so readers
  can discount them.
- No identity-accuracy claim is made: no reviewed GT exists for 4xPD. HOTA /
  AssA / IDF1 / ID-switch numbers remain blocked on researcher-corrected
  `reviewed_gt.mot` (the `evaluate-validation` guard enforces this).
- UDMT remains an external research backend; no UDMT code was vendored.

## Test results

- Baseline: 32 passed / 0 failed / 32 total.
- Final: **33 passed / 0 failed / 33 total**
  (`./.venv/bin/python tests/test_functional.py`).
- New test 33 covers: confident crossing, bounce-back swap detection,
  symmetric-exit uncertainty, zero-contact, event-kind classification,
  inversion-episode chaining, JSON/CSV artifacts, dashboard rendering, and
  review-package score boost.
- No packaged desktop app exists in this repository; no packaging/rebuild step
  applies.

## Remaining blocker / next loop target

1. **GT calibration (blocked on user)**: researcher-corrected `reviewed_gt.mot`
   for 4xPD would let the loop (a) validate whether the 18 inversion episodes
   are true switches, and (b) calibrate the hypothesis scoring temperature
   against true outcomes — turning relative margins into calibrated
   uncertainty. This is the single highest-value next step and requires manual
   review work only the user can do.
2. **Episode-aware review consolidation (reachable next iteration)**: extend
   the sparse GT package to sample *one representative frame per inversion
   episode* plus event-boundary frames, cutting redundant review of cascaded
   events; add an interactive episode view to the dashboard.
3. **Smoother feedback (reachable)**: feed high-confidence inversion episodes
   back as hard constraints into the full-video identity smoother and measure
   internal-consistency changes (still annotation-free until GT arrives).
