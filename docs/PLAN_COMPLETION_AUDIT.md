# DenseTrack-AI Plan Completion Audit

Updated: 2026-06-25

This audit distinguishes verified local capabilities from work that still requires external data, hardware, or upstream UDMT runtime evidence.

## Verified Locally

| Requirement | Evidence | Status |
|---|---|---|
| Zero-training-label native MOT baseline | `densetrack complete 4xPD.mp4 ...` produced 4 trajectories and `manual_annotation_used=false` | Verified |
| Mac-native zero-click adaptive bidirectional MOT | `results/4xPD_mac_adaptive_complete`: automatic two-end initialization, appearance templates, forward/backward alignment, per-frame fusion, 4 trajectories, `3.519 / 4` observed tracks/frame | Verified |
| Identity-stable MOT smoothing | `results/4xPD_identity_stable_complete_v2`: full-video Viterbi identity smoothing, `99.7%` identity stability, high-risk frames `1372 -> 25`, p95 jump proxy `499.28 px -> 62.64 px` | Verified |
| Frame-level annotation-free identity audit | `results/4xPD_mac_adaptive_v2_complete/identity_risk.json`: `9363` audited frames, `1512` high-risk frames; sparse review package selected `94 / 100` frames with identity risk and `58 / 100` direct disagreement frames | Verified |
| Replace first-frame manual clicks | `auto_init/initialization_points.json`: frame 255, score 0.986, 4 / 4 points | Verified |
| UDMT first-frame no-click handoff | `results/4xPD_udmt_handoff`: frame 0, score 0.9511, 4 / 4 points, official-workspace `start_pos_array.txt`, `direct_first_frame_drop_in_ready=true` | Verified |
| Mature OSS MOT reuse | `densetrack.tracking.norfair_tracker.NorfairTrackletGenerator`, optional extra `densetrack[norfair]`, and `configs/mice_topview_norfair.yaml` reuse BSD-3 Norfair for generic MOT | Verified |
| Visual tracking result | `dashboard.html` and `tracking_preview_browser.mp4`; browser video `readyState=4` | Verified |
| Clean tracking playback | `tracking_preview_clean.mp4`, `tracking_preview_browser.mp4`, and `overlay_before_after_frame75.jpg`; short trails cut low-confidence or large-jump segments | Verified |
| Product-state dashboard boundary | `dashboard.html` shows `产品状态 / Product state` and says visual tracking is ready while sparse GT metrics still need review | Verified |
| Bilingual result display | Dashboard and Web workspace render Chinese / English labels | Verified |
| Upload, process, result download workflow | `densetrack serve`; synthetic Web job completed and exposed dashboard artifacts | Verified |
| Browser streaming | HTTP Range request returns `206 Partial Content` and `Accept-Ranges: bytes` | Verified |
| Annotation-free behavior motif baseline | 4xPD generated 12308 samples and 6 recurring motif clusters | Verified |
| Low-effort scientific validation preparation | `densetrack prepare-validation` generated a 100-frame risk-prioritized pre-filled review package | Verified |
| Browser-assisted sparse GT correction | `validation_review/review.html` provides bilingual frame review, drag/resize, add/remove, progress persistence, and local `reviewed_gt.mot` export | Verified |
| Guard against fake GT claims | `densetrack evaluate-validation` exits with code 2 without `--confirm-reviewed` | Verified |
| Web reviewed-GT return path | Synthetic HTTP upload rejects missing confirmation, then accepts reviewed MOT, writes evaluation reports, updates job state, and renders dashboard metric cards | Verified |
| ILP global association | Functional suite exercises `ilp` and `min_cost_flow` alias; both produce the expected 2 DAG paths | Verified |
| Geometry-pose motif enrichment | `results/4xPD_pose_enriched_complete`: exported geometry pose columns and clustered 12308 samples with 11 features | Verified |
| UMAP / HDBSCAN real behavior benchmark | `results/4xPD_pose_enriched_complete/behavior_clusters_umap_hdbscan.json`: UMAP embedding then HDBSCAN, 12308 samples, 4 effective motifs, 900 explicit noise samples; dashboard shows advanced embedding and timeline | Verified |
| BEAST3D-inspired visual SSL export | `results/web_workspace/jobs/97595b3d6d8e/results/visual_ssl_report.json`: 12484 samples, 4 IDs, 8-D embeddings, pose features used, 18726 same-frame negative pairs | Verified |
| Dolph2Vec-inspired bioacoustic SSL export | Functional suite synthetic WAV test writes `bioacoustic_ssl_report.json` and `bioacoustic_ssl_units.csv`; no 4xPD audio is available | Verified for WAV input |
| Dashboard SSL sections | Current short job and real 4xPD dashboard render the BEAST3D-inspired block and `visual_ssl_embeddings.csv` download link | Verified |
| Advanced dashboard responsive QA | UMAP/HDBSCAN dashboard verified at desktop `1280x900` and mobile `390x844`: both images loaded, no horizontal overflow, no console errors | Verified |
| Mac-adaptive dashboard responsive QA | Job `4d5000000003` verified at desktop `1280x900` and mobile `390x844`: video `readyState=4`, 6 / 6 images loaded, warning status wraps on mobile, no horizontal overflow, no page console errors | Verified |
| Identity-risk dashboard responsive QA | Job `4d5000000004` verified at desktop `1280x900` and mobile `390x844`: identity audit table visible, video `readyState=4`, 6 / 6 images loaded, no horizontal overflow, no page console errors | Verified |
| Current dashboard responsive QA | Job `97595b3d6d8e` verified at desktop `1280px` and mobile `390x844`: SSL block visible, visual download link present, video `readyState=4`, no horizontal overflow | Verified |
| Local regression suite | `python tests/test_functional.py`: 33 / 33 passed | Verified |
| Contact-event audit layer | `results/claude_mart_graph_loop_20260728_224813/`: 355 events, 278 measurement-ambiguous / 77 proximity-only, 135 raw disagreements → 18 inversion episodes; fresh full run reproduces identical event structure (146.7 s, tracking stats unchanged); review package carries contact priority on 87/100 frames | Verified (annotation-free audit; not identity accuracy) |

## Implemented But Not Yet Externally Validated

| Requirement | Current evidence | Missing proof |
|---|---|---|
| Grounding DINO detector | Adapter and dependency validation exist | Installed weights and real-video comparison |
| SAM 2 propagation | Adapter with conservative fallback exists | Installed weights and real-video stability comparison |
| DLC SuperAnimal | Adapter and geometry fallback exist | Installed model, inference evidence, bundling-license decision |
| DLC keypoint B-SOiD benchmark | Geometry-pose enrichment and real UMAP/HDBSCAN comparison exist | Installed DLC model and keypoint-level comparison benchmark |
| UDMT external research backend | Environment checker, no-click `start_pos_array.txt` handoff, MOT import bridge | Linux / CUDA upstream run and comparison benchmark |
| Full BEAST3D neural pretraining | Visual SSL export and multi-view readiness audit exist | Synchronized calibrated multi-view videos and differentiable rendering training stack |
| Full Dolph2Vec neural training | WAV feature/codebook path exists | Synchronized animal vocalization WAV data and Wav2Vec2/Dolph2Vec training/evaluation |
| Cross-video scientific metrics | 4xPD pre-filled review package exists | Researcher-corrected `reviewed_gt.mot` and 3-5 independent videos |

## Not Yet Implemented

| Requirement | Reason |
|---|---|
| Direct UDMT execution inside `Pipeline._run_udmt_tracking()` | Official workflow is GUI-centered; a Linux/CUDA headless runner has not been verified end to end |
| Dedicated detector fine-tuning | Requires reviewed corrections from validation packages |
| Multi-species production validation | Requires insect and zebrafish videos |
| Multi-user remote deployment, permissions, remote queue | Local researcher workspace is implemented; deployment infrastructure is not |
| TensorRT optimization | Requires NVIDIA deployment target |

## Honest Completion State

The local black-mouse top-view solution is operational and tested end to end. The
current dashboard now uses identity-stable tracking plus clean short-trail playback,
so visual review is much easier and the internal high-risk frame count is sharply
lower. The reported ID-switch improvement is still an annotation-free internal
consistency result; scientific ID-switch, MOTA, and IDF1 claims require
researcher-corrected GT. The full research plan is not complete until external
validation inputs and GPU environments are supplied and verified.
