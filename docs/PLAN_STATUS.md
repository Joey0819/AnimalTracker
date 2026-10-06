# DenseTrack-AI Plan Status

Updated: 2026-07-28

## Completed Locally

### Contact-Event Audit Layer — 2026-07-28

- Auditable contact events from tracked output: per-event merge/split records, event kinds (`measurement_ambiguous` / `proximity_only`), top-K incoming→outgoing identity hypotheses with documented motion+size scoring, best-vs-second-best margins, confident/ambiguous/uncertain/unresolved tiers, and tracker-agreement flags
- Label-inversion episodes chain cascading tracker-disagreement events into honest review units (4xPD: 135 raw disagreement events → 18 episodes); raw counts are preserved but no longer the headline
- Artifacts `contact_events.json/csv` written by the pipeline; `densetrack contact-events <run_dir>` recomputes the layer from immutable run artifacts
- Dashboard bilingual contact-event section; sparse GT review package boosts frames with contact-event uncertainty (4xPD: 87/100 selected frames)
- Evidence root `results/claude_mart_graph_loop_20260728_224813/`: v1 vs v1.1 immutable-input audit comparison plus a fresh full `densetrack complete` 4xPD run (146.7 s) with unchanged tracking stats (2.777 observed tracks/frame, stability 0.708)
- Functional suite: 33/33 passing
- Honest boundary: hypothesis margins are annotation-free relative scores, not GT-calibrated; inversion episodes are review targets, not measured ID switches. Scientific identity metrics still require researcher-corrected GT.

### Phase 0 / Phase 1 MVP

- CLI pipeline runs on real video: `densetrack run`
- Video metadata and config inspection: `densetrack info`
- MOT evaluation command exists: `densetrack evaluate`
- Previous-run re-export works: `densetrack export`
- Rule-based behavior analysis works: `densetrack analyze`
- Annotation-free motif clustering works: `densetrack discover-behaviors`
- Standalone bilingual research dashboard works: `densetrack build-report`
- Zero-click video-to-dashboard workflow works: `densetrack complete`
- Local bilingual upload/process/download workspace works: `densetrack serve`
- Risk-prioritized sparse GT review package works: `densetrack prepare-validation`
- Bilingual browser GT reviewer works: frame navigation, drag/resize, add/remove boxes, reviewed progress, local `reviewed_gt.mot` download
- Web GT return path works: upload confirmed `reviewed_gt.mot`, reject unconfirmed uploads, publish sparse MOTA / IDF1 report, refresh dashboard metric cards
- Reviewed sparse GT evaluation guard works: `densetrack evaluate-validation`
- No-click initialization seed generation works: `densetrack auto-init`
- UDMT workspace-compatible no-click seed handoff works: `densetrack udmt-handoff`
- Optional UDMT external backend check: `densetrack udmt-check`
- UDMT-derived MOT import bridge: `densetrack udmt-import`
- Mature open-source Norfair MOT backend: `tracklet.backend: norfair`, optional dependency `densetrack[norfair]`, config `configs/mice_topview_norfair.yaml`
- Background subtraction detector
- Dark-object detector for black mice on light bedding
- Grounding DINO adapter with early dependency validation
- SAM 2 adapter with conservative fallback
- DLC SuperAnimal adapter with dependency/provenance boundary
- Geometry pose fallback
- ByteTrack-style local tracklet generation
- Fixed-count tracker for known `n_animals` top-view videos
- Mac-native `mac_adaptive` tracker: zero-click two-end initialization, lightweight appearance templates, forward/backward association, identity-space alignment, and per-frame fusion
- Identity-stable Viterbi smoothing after bidirectional fusion to reduce ID switches and large trajectory jumps
- Self-supervised ReID path with lazy model loading and skip-on-insufficient-triplets behavior
- Greedy, Hungarian, and SciPy ILP DAG path-cover global tracklet optimizers
- `min_cost_flow` config alias backed by the equivalent ILP path-cover formulation
- Geometry-pose CSV export and automatic pose-enriched motif discovery
- Installed UMAP + HDBSCAN behavior path with real 4xPD pose-enriched benchmark
- BEAST3D-inspired visual SSL representation export: `visual_ssl_report.json` and `visual_ssl_embeddings.csv`
- Dolph2Vec-inspired bioacoustic SSL export for WAV audio: `bioacoustic_ssl_report.json` and `bioacoustic_ssl_units.csv`
- Dashboard sections for no-training-annotation visual SSL and bioacoustic SSL extensions
- Standalone CLI commands: `densetrack extract-representations` and `densetrack analyze-audio-ssl`
- Confidence audit report using effective confidence
- Export formats: MOT, COCO, CSV/DLC-style, SLEAP-style JSON
- Overlay visualization video
- Clean short-trail playback generation from MOT or trajectories, with low-confidence and large-jump trail cutting
- Run summary/provenance file
- Species prompt templates
- Functional test suite: 30/30 passing

### Identity-Stable Tracking Update — 2026-06-25

Implemented a new `mac_adaptive_bidirectional_identity_stable` layer after bidirectional fusion:

- Uses full-video dynamic programming over per-frame mutually exclusive ID-to-detection assignments.
- Transition cost favors smooth physical motion; evidence cost still uses forward pass, backward pass, fused output, detector confidence, and first-frame appearance anchors.
- Descriptor caching keeps the full 4xPD run practical on Mac: the tracking stage completed in `37.6s`, and the full pipeline completed in `97.8s`.

4xPD verification output:

- Output directory: `results/4xPD_identity_stable_complete_v2`
- Dashboard: `results/4xPD_identity_stable_complete_v2/dashboard.html`
- Web URL: `http://127.0.0.1:8765/jobs/51d57ab1e001/results/dashboard.html`
- Algorithm: `mac_adaptive_bidirectional_identity_stable`
- Identity stability score: `99.7%`
- High-risk frames: `1372 -> 25`
- Average observed tracks/frame: `3.508 -> 3.519`
- Mostly observed frames: `91.3% -> 92.2%`
- Mean frame-to-frame MOT jump proxy: `99.29 px -> 23.36 px`
- P95 frame-to-frame MOT jump proxy: `499.28 px -> 62.64 px`
- Large jump proxy `>150 px`: `8331 -> 863`

Honest boundary:

- The improvement is measured by annotation-free internal consistency and motion-continuity proxies.
- True ID-switch count still requires researcher-corrected sparse or full GT.

### Mature OSS Tracking Reuse — 2026-06-26

Implemented direct reuse of mature open-source tracking where it fits the problem:

- Added `NorfairTrackletGenerator`, a thin adapter around Norfair `Tracker`.
- Added optional dependency extra `densetrack[norfair]`.
- Added `configs/mice_topview_norfair.yaml` for generic top-view tracking.
- Kept local `mac_adaptive` identity-stable tracking for the project-specific case that Norfair does not directly cover: known animal count, automatic/first-frame identity seeds, bidirectional full-video fusion, and stable identity assignment across the whole recording.

Dependency review:

- Norfair: BSD-3, mature Python MOT library, suitable optional dependency.
- Roboflow `trackers`: Apache-2.0, but current PyPI `trackers==2.0.1` failed import in a clean temporary environment because of a `supervision` API mismatch.
- BoxMOT: mature but AGPL-3.0, not suitable as a bundled/default dependency for this Apache-2.0 project.

### Clean Playback Product Inspection — 2026-06-20

Implemented after visual QA showed long crossing trail lines in the 4xPD dashboard:

- `OverlayVideoGenerator` now defaults to short trails and clears trail history when confidence is low or the next centroid jump is implausibly large.
- Existing MOT outputs can be re-rendered through `generate_from_mot()` without rerunning the tracker.
- The static dashboard prefers `tracking_preview_clean.mp4`, labels the playback as `干净轨迹回放 / Clean tracking playback`, and includes a `产品状态 / Product state` boundary panel.
- The current 4xPD Web job `97595b3d6d8e` was regenerated with a clean preview and a before/after frame comparison.

Honest boundary:

- Clean playback improves visual interpretability; it does not change the underlying MOT tracks or turn internal agreement into scientific accuracy.
- Scientific MOTA / IDF1 still requires researcher-corrected sparse GT.

### BEAST3D / Dolph2Vec-Inspired Optimization — 2026-06-14

Implemented project optimization based on two current research signals:

- BEAST3D, arXiv:2606.02937: self-supervised 3D representation learning from unlabeled calibrated multi-view video.
- Dolph2Vec, arXiv:2606.12503: species-specific SSL for dolphin vocalizations with downstream whistle tasks.

What is operational:

- `PipelineConfig.representation` enables no-training-annotation visual representation export after MOT tracking.
- `PipelineConfig.bioacoustic_ssl` enables optional WAV-based communication-behavior SSL export.
- `densetrack extract-representations <run_dir>` can add visual SSL artifacts to an existing run.
- `densetrack analyze-audio-ssl <audio.wav>` can analyze unlabeled WAV audio as a standalone operation.
- `dashboard.html` renders both modules bilingually when their reports exist.
- Functional tests cover synthetic visual SSL, synthetic bioacoustic SSL, and dashboard rendering.

Honest boundary:

- The visual module is BEAST3D-inspired but does not implement BEAST3D's vision-transformer-to-3D-Gaussian-splats training loop or differentiable held-out-view rendering.
- The audio module is Dolph2Vec-inspired but does not train Wav2Vec2/Dolph2Vec weights.
- For the existing 4xPD mouse video, the visual SSL path is directly applicable; the audio path requires a synchronized WAV recording.

### UDMT External Research Backend

Status: adapter scaffold implemented; upstream UDMT itself is not bundled.

- UDMT is treated as an optional external research backend because its public license is academic/non-profit non-commercial research use only.
- DenseTrack now targets no-click initialization before UDMT-style first-frame clicking: `densetrack auto-init` generates seed points and a visual QA image from detector results.
- `densetrack udmt-handoff` defaults to frame `0`, writes UDMT-compatible `start_pos_array.txt`, mirrors the official `tmp/<video>/extracted-images/` workspace path, and records whether direct first-frame drop-in is ready.
- `densetrack udmt-check` validates an external checkout, Python executable, license acknowledgement, and optional importability.
- `densetrack udmt-import` imports UDMT-derived MOT-like tracks and re-exports DenseTrack COCO/CSV/SLEAP/run-summary artifacts.
- Full UDMT execution still requires a separate Linux/CUDA/Python 3.8 environment and the upstream GUI workflow.

### 4xPD Auto Initialization Verification

Command:

```bash
./.venv/bin/python -m densetrack.cli auto-init \
  4xPD.mp4 \
  --config configs/mice_black_topview_fixed.yaml \
  --n-animals 4 \
  --output results/4xPD_auto_init \
  --max-frames 450 \
  --sample-every 3
```

Outputs:

- `results/4xPD_auto_init/initialization_points.json`
- `results/4xPD_auto_init/initialization_points.jpg`

Observed result:

- Status: ok
- Selected frame: 255
- Score: 0.986
- Points: 4 / 4
- Manual annotation used: false

### 4xPD UDMT No-Click Handoff Verification

Command:

```bash
./.venv/bin/python -m densetrack.cli udmt-handoff \
  4xPD.mp4 \
  --config configs/mice_black_topview_fixed.yaml \
  --n-animals 4 \
  --output results/4xPD_udmt_handoff
```

Observed result:

- First-frame automatic detection: `4 / 4` seed points
- First-frame score: `0.9511`
- `manual_training_annotation_required=false`
- `manual_first_frame_click_required=false`
- `direct_first_frame_drop_in_ready=true`
- Official-workspace mirror: `results/4xPD_udmt_handoff/udmt_workspace_seed/tmp/4xPD/extracted-images/start_pos_array.txt`

### 4xPD.mp4 Verification

Command:

```bash
./.venv/bin/python -m densetrack.cli run \
  4xPD.mp4 \
  --config configs/mice_black_topview_fixed.yaml \
  --n-animals 4 \
  --output results/4xPD_plan_complete
```

Outputs:

- `results/4xPD_plan_complete/tracks.mot`
- `results/4xPD_plan_complete/detections.coco.json`
- `results/4xPD_plan_complete/poses.csv`
- `results/4xPD_plan_complete/labels.sleap.json`
- `results/4xPD_plan_complete/confidence_report.json`
- `results/4xPD_plan_complete/overlay.mp4`
- `results/4xPD_plan_complete/run_summary.json`
- `results/4xPD_plan_complete/behavior_report.json`
- `results/4xPD_plan_complete/behavior_timeline.csv`
- `results/4xPD_plan_complete/tracking_preview_77s_92s.mp4`
- `results/4xPD_plan_complete/tracking_preview_frames_77s_92s.jpg`

Observed tracking summary:

- Trajectories: 4
- Video frames: 9363
- SLEAP-style frames: 9363
- Average detections per frame: 3.087
- Average observed tracks per frame: 3.071 / 4
- Frames with at least 3 observed tracks: 78.4%
- Per-ID effective confidence: 0.757, 0.767, 0.751, 0.797
- Confidence tier: medium for all 4 IDs

### 4xPD Zero-click Complete Workflow Verification

Command:

```bash
./.venv/bin/python -m densetrack.cli complete \
  4xPD.mp4 \
  --config configs/mice_black_topview_fixed.yaml \
  --n-animals 4 \
  --clusters 6 \
  --output results/4xPD_zero_click_complete
```

Outputs:

- `results/4xPD_zero_click_complete/complete_run_manifest.json`
- `results/4xPD_zero_click_complete/dashboard.html`
- `results/4xPD_zero_click_complete/tracking_preview_browser.mp4`
- `results/4xPD_zero_click_complete/auto_init/initialization_points.json`
- `results/4xPD_zero_click_complete/auto_init/initialization_points.jpg`
- MOT / COCO / CSV / SLEAP-style exports, confidence audit, behavior statistics, unsupervised motif outputs, and overlay video

Observed result:

- Manual annotation required: false
- Manual annotation used: false
- Auto-init: ok, selected frame 255, score 0.986, points 4 / 4
- Trajectories: 4
- Video frames: 9363
- Average observed tracks per frame: 3.545 / 4
- Frames with at least 3 observed tracks: 92.3%
- Detection and pose: 303.1 FPS
- Tracking pipeline total: 56.7s, 165.0 FPS
- Unsupervised motif samples: 12308
- Motif clusters: 6

Dashboard QA:

- Chinese / English result display is enabled.
- Desktop viewport: 1440px document width, 1440px viewport width.
- Mobile viewport: 390px document width, 390px viewport width.
- Browser-compatible tracking preview: `readyState=4`, no playback error.

### Local Web Workspace Verification

Command:

```bash
./.venv/bin/python -m densetrack.cli serve \
  --workspace results/web_workspace \
  --host 127.0.0.1 \
  --port 8765
```

Verified workflow:

- Open `http://127.0.0.1:8765`
- Upload a video with the bilingual form
- Create an asynchronous zero-click processing job
- Poll queued / running / completed status
- Open the generated bilingual dashboard
- Stream the browser-compatible tracking preview with HTTP Range requests
- Download run artifacts
- Upload browser-reviewed `reviewed_gt.mot` with explicit manual-review confirmation
- Reject unconfirmed GT uploads with HTTP `400`
- Generate sparse scientific metrics and refresh dashboard metric cards

Observed QA:

- Desktop workspace: 1280px document width, 1280px viewport width.
- Mobile workspace: 390px document width, 390px viewport width.
- Health endpoint: `GET /api/health` returns `可用 / healthy`.
- Preview streaming: `206 Partial Content`, `Accept-Ranges: bytes`.
- Browser result page: tracking video `readyState=4`, no playback error.
- Confirmed synthetic GT HTTP upload: `POST /api/jobs/c2c60d669996/validation` returns MOTA `1.0`, IDF1 `1.0`, ID switches `0`.
- Dashboard displays reviewed sparse metrics with an explicit sparse-only scientific boundary.

### 4xPD Geometry-Pose Enriched Workflow Verification

Command:

```bash
./.venv/bin/python -m densetrack.cli complete \
  4xPD.mp4 \
  --config configs/mice_black_topview_fixed.yaml \
  --n-animals 4 \
  --clusters 6 \
  --validation-frames 100 \
  --output results/4xPD_pose_enriched_complete
```

Observed result:

- Tracking pipeline total: 53.47s, 175.11 FPS
- Auto-init: ok, score 0.986, points 4 / 4
- Geometry poses computed: 33357
- `poses.csv`: 36924 rows
- Populated pose columns: orientation 32725, body length 29337, body area 32725
- Behavior method: `densetrack_bsoid_inspired_pose_clustering`
- Behavior features: 7 centroid/bbox features + 4 geometric pose features
- Unsupervised samples: 12308
- Motif clusters: 6
- Review package: 100 frames
- Web result: `http://127.0.0.1:8765/jobs/4d5000000002/results/dashboard.html`

### 4xPD UMAP + HDBSCAN Behavior Verification

Command:

```bash
./.venv/bin/python -m densetrack.cli discover-behaviors \
  results/4xPD_pose_enriched_complete \
  --pose-csv results/4xPD_pose_enriched_complete/poses.csv \
  --feature-mode pose \
  --embedding umap \
  --clustering hdbscan \
  --hdbscan-min-cluster-size 200 \
  --hdbscan-min-samples 2 \
  --output results/4xPD_pose_enriched_complete/behavior_clusters_umap_hdbscan.json
```

Observed result:

- IDs: 4
- Samples: 12308
- Effective HDBSCAN motifs: 4
- Noise samples: 900
- Output embedding: `behavior_clusters_umap_hdbscan_embedding.jpg`
- Output timeline: `behavior_clusters_umap_hdbscan_timeline.jpg`
- Dashboard status: `UMAP + HDBSCAN: 4 有效模体 / motifs, 900 噪声 / noise`
- Browser QA: desktop `1280x900` and mobile `390x844`, no horizontal overflow, both advanced images loaded, no console errors

Interpretation boundary: the path is operational and clusters the UMAP embedding before
HDBSCAN. Noise is reported explicitly and is not counted as a motif. These recurring
motion motifs are not validated semantic behavior labels.

### 4xPD Sparse GT Review Package

Command:

```bash
./.venv/bin/python -m densetrack.cli prepare-validation \
  results/4xPD_zero_click_complete \
  --frames 100 \
  --video 4xPD.mp4
```

Outputs:

- `results/4xPD_zero_click_complete/validation_review/validation_manifest.json`
- `results/4xPD_zero_click_complete/validation_review/review_grid.jpg`
- `results/4xPD_zero_click_complete/validation_review/frames/`
- `results/4xPD_zero_click_complete/validation_review/raw_frames/`
- `results/4xPD_zero_click_complete/validation_review/review.html`
- `results/4xPD_zero_click_complete/validation_review/review_data.json`
- `results/4xPD_zero_click_complete/validation_review/review_frames.csv`
- `results/4xPD_zero_click_complete/validation_review/prelabels.mot`
- `results/4xPD_zero_click_complete/validation_review/corrected_gt_template.mot`
- `results/4xPD_zero_click_complete/validation_review/README.md`

Observed sampling:

- Selected frames: 100
- Risk-priority frames: 65
- Temporal-coverage frames: 33
- Boundary frames: 2
- Frames with fixed-count prediction carry-forward: 72
- Frames with close-neighbor crowding under 100px: 60

Scientific boundary:

- Tracking manual annotation used: false
- Manual review required before scientific metrics: true
- The browser reviewer pre-fills boxes and lets the researcher correct errors with direct drag/resize, add/remove controls, progress persistence, and local `reviewed_gt.mot` export.
- `densetrack evaluate-validation` refuses to compute metrics unless `--confirm-reviewed` is supplied.
- The pre-filled template reduces manual work but is not GT until corrected and confirmed.

### 4xPD Unsupervised Behavior Discovery

Command:

```bash
./.venv/bin/python -m densetrack.cli discover-behaviors \
  results/4xPD_plan_complete \
  --clusters 6 \
  --window-frames 15 \
  --sample-every 3
```

Outputs:

- `results/4xPD_plan_complete/behavior_clusters.json`
- `results/4xPD_plan_complete/behavior_clusters_timeline.csv`
- `results/4xPD_plan_complete/behavior_clusters_embedding.jpg`
- `results/4xPD_plan_complete/behavior_clusters_timeline.jpg`

Observed result:

- IDs: 4
- Samples: 12484
- Motif clusters: 6
- Annotation required: false

Honest boundary: local runs now use a B-SOiD-inspired centroid/bbox + geometric-pose path. It discovers recurring motion motifs without behavior labels, but it does not claim validated semantic behavior names. UMAP + HDBSCAN is installed and verified on 4xPD; DLC keypoint evidence still requires an installed model backend.

### 4xPD Mac-native Adaptive Bidirectional Workflow — 2026-06-01

Command:

```bash
./.venv/bin/python -m densetrack.cli complete \
  4xPD.mp4 \
  --config configs/4xPD_mac_adaptive.yaml \
  --n-animals 4 \
  --clusters 6 \
  --validation-frames 100 \
  --output results/4xPD_mac_adaptive_complete
```

Implemented algorithm:

- No clicks and no training annotations
- Automatic initialization near both video ends: frame `22` and frame `9353`
- Lightweight crop-derived appearance templates with guarded EMA updates
- Independent forward and backward constrained association
- Identity-space alignment and per-frame vote-first consistency fusion
- Native NumPy / OpenCV / SciPy execution on macOS; no UDMT source, UDMT weights, or CUDA runtime

Observed result:

- Total pipeline: `72.42s`, `129.29 FPS`
- IDs: `4`, full output coverage `9363` frames per ID
- Observed tracks per frame: `3.519 / 4`
- Frames with at least 3 observed animals: `92.16%`
- Appearance template coverage: `99.72%`
- Forward/backward internal agreement: `32.53%`
- Fusion conflict records: `20168`
- Geometry pose rows: `37452`
- Advanced behavior path: `12484` samples, `10` effective UMAP + HDBSCAN motifs, `1689` noise samples
- Web result: `http://127.0.0.1:8765/jobs/4d5000000003/results/dashboard.html`
- Browser QA: desktop override `1280x900` and mobile `390x844`; no horizontal overflow after mobile status-wrap fix, video `readyState=4`, all 6 images loaded, no page console errors

Interpretation boundary: `32.53%` is an internal disagreement signal between two
annotation-free tracking directions, not an accuracy metric. The Dashboard renders it
as an orange warning. Researcher-reviewed sparse GT remains required before claiming
MOTA, IDF1, or ID-switch improvements over the fixed-count baseline or UDMT.

### 4xPD Mac-native Identity-risk Audit v2 — 2026-06-01

The Mac-native tracker was extended with:

- Foreground-aware descriptors that suppress light bedding inside detection crops
- Immutable anchor descriptors plus a recent template bank per identity
- Template updates gated by detector confidence, assignment cost, and ambiguity margin
- Per-frame `identity_risk.json` and `identity_risk.csv`
- Risk-prioritized sparse GT sampling that includes bidirectional disagreements
- Bilingual Dashboard identity-audit table with high-risk frame counts and top conflict frames

Observed comparison:

| Metric | v1 | v2 |
|---|---:|---:|
| Forward/backward internal agreement | `32.53%` | `35.10%` |
| Fusion conflicts | `20168` | `19446` |
| Observed tracks per frame | `3.5191 / 4` | `3.5192 / 4` |
| Frames with at least 3 observed animals | `92.16%` | `92.16%` |
| Total runtime | `72.42s` | `91.27s` |

Identity-risk audit:

- Audited frames: `9363`
- Frames with any bidirectional disagreement: `7839`
- High-risk frames: `1512`
- Mean internal risk: `0.346`
- Sparse review frames with identity risk: `94 / 100`
- Sparse review frames with direct bidirectional disagreement: `58 / 100`
- Web result: `http://127.0.0.1:8765/jobs/4d5000000004/results/dashboard.html`

The improvement is measurable but modest because these mice have similar appearance and
frequent contact. The v2 product value is not an unsupported accuracy claim: it provides
zero-click tracking plus explicit frame-level identity-risk evidence so experimental
review effort is directed to the ambiguous segments.

### Optional UDMT-style First-frame Click Assist — 2026-06-01

The bilingual Web workspace now offers two initialization modes:

- `零点击自动 / Zero-click auto`: the existing default detector-derived initialization.
- `首帧点击辅助 / First-frame click assist`: upload the video, click each visible animal
  once in ID order on frame `0`, then start tracking.

The click-assisted path is wired into the Mac-native `mac_adaptive` tracker:

- The Web API validates one click per requested animal and persists the initialization JSON.
- Click order is matched to detected instances on frame `0`.
- That ordering initializes the forward identity space before backward alignment and fusion.
- Result manifests distinguish `manual_first_frame_click_used=true` from
  `manual_training_annotation_used=false`.
- The result Dashboard displays the initialization source bilingually.

This is lightweight instance selection without training annotation. It is inspired by the
public UDMT interaction contract, but it does not claim to implement UDMT's unsupervised deep
transfer-learning network. The external Linux/CUDA UDMT adapter remains a separate benchmark path.

## Remaining External Blockers

These items cannot be honestly completed from the current local workspace alone:

- 3-5 independent real mouse top-view videos are not available.
- A 100-frame risk-prioritized pre-filled review package exists for 4xPD, but researcher-corrected `reviewed_gt.mot` is not available yet, so MOTA/IDF1/ID-switch metrics cannot be claimed.
- Grounding DINO weights/dependency are not installed or validated on this machine.
- SAM 2 weights/dependency are not installed or validated on this machine.
- DeepLabCut/SuperAnimal is not installed and the SuperAnimal weight license must be confirmed before any commercial bundling.
- UDMT is not installed locally. The no-click handoff is ready, but the Linux/CUDA external run remains pending. Commercial-distribution license review is deferred and does not block this research prototype.
- Phase 2 dedicated detector training requires reviewed/corrected labels.
- DLC keypoint evidence and LLM reports remain follow-ons; the local geometric-pose enrichment and UMAP/HDBSCAN comparison benchmark are implemented.
- Multi-user deployment, multi-species production support, and TensorRT remain Phase 4+ productization work. The local upload/process/download/review/evaluate Web workspace and ILP path-cover optimizer are implemented.

## Current Honest Product State

The project is a runnable AutoLabel MOT MVP for this black-mouse top-view video class. It produces usable visual tracking, auto-label exports, confidence reports, provenance, and first-layer rule-based behavior statistics. It is not yet a validated scientific tracker with published MOTA/IDF1 claims because no ground-truth validation set is present.
