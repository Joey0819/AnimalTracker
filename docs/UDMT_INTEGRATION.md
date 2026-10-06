# UDMT External Backend Integration

Updated: 2026-06-01

## Decision

UDMT should be added to this project as an optional external research backend, not vendored into the core DenseTrack package. The product direction should go beyond the current UDMT GUI interaction by replacing first-frame clicks with automatic initialization whenever possible.

Reason:

- It directly matches the project goal: no training annotation, multi-animal tracking, cross-species generalization.
- The Nature Methods article was published on 2026-05-04 and describes UDMT as an unsupervised deep transfer learning method for multi-animal tracking without training annotations.
- The public UDMT repository demonstrates mice, rats, Drosophila, C. elegans, and Betta splendens.
- The public code license is academic/non-profit non-commercial research use only, so it cannot be copied into this Apache-2.0 project or bundled into a commercial default build.

## Practical Boundary

UDMT is not completely zero human input. The upstream README says users click animals in the first frame to specify the individuals to track. This is still a strong fit because it avoids dense frame-by-frame training labels, but the target DenseTrack workflow is stricter: automatic initialization first, manual first-frame clicks only as a fallback.

DenseTrack now includes a no-click initialization step, an optional browser-based
first-frame click assist mode, and a UDMT workspace handoff:

```bash
densetrack auto-init /path/to/video.mp4 \
  --config configs/mice_black_topview_fixed.yaml \
  --n-animals 4 \
  --output results/auto_init
```

This writes detector-derived seed points:

- `initialization_points.json`
- `initialization_points.jpg`

Build an external UDMT handoff package:

```bash
densetrack udmt-handoff /path/to/video.mp4 \
  --config configs/mice_black_topview_fixed.yaml \
  --n-animals 4 \
  --output results/video_udmt_handoff
```

By default, `udmt-handoff` scans only frame `0` so the output can map directly into
the official UDMT workspace. It writes:

- `start_pos_array.txt`: one `x,y` centroid per animal
- `udmt_workspace_seed/tmp/<video>/extracted-images/start_pos_array.txt`
- `udmt_handoff.json`: bilingual automation boundary and source coordinate basis
- `initialization_points.json`
- `initialization_points.jpg`

If frame `0` is not separable, increase `--max-frames`. The package will then mark
that seed-frame alignment or an external runner extension is required.

UDMT also has a heavier runtime profile than the current DenseTrack MVP:

- Upstream recommended environment: Ubuntu 20.04+, Python 3.8, PyTorch/CUDA, NVIDIA GPU.
- Current DenseTrack package targets Python 3.10+ and already runs a local OpenCV-based MVP on this machine.
- Therefore UDMT should run in a separate conda environment and communicate through files.

## Mac-native Independent Alternative

DenseTrack now includes an independently implemented `mac_adaptive` backend for Macs.
It does not copy UDMT source code, weights, or ST-Net architecture. It targets the same
automation requirement with a lighter local algorithm:

```text
automatic detector initialization near both video ends
  -> foreground-aware lightweight appearance templates from detected crops
  -> forward and backward constrained association
  -> identity-space alignment from shared detections and geometry
  -> per-frame consistency fusion
  -> frame-level identity_risk.json/csv audit for sparse review prioritization
  -> standard DenseTrack exports, audit, motifs, and bilingual dashboard
```

Run it locally:

```bash
densetrack complete 4xPD.mp4 \
  --config configs/4xPD_mac_adaptive.yaml \
  --n-animals 4 \
  --output results/4xPD_mac_adaptive_complete
```

This backend requires no training labels, CUDA runtime, or UDMT installation. Its
default mode requires no clicks. The bilingual Web workspace also exposes an optional
`首帧点击辅助 / first-frame click assist` mode: users click each animal once in ID order,
and the click order initializes the forward identity space before bidirectional tracking.
These clicks are lightweight instance selection, not training annotations.

It is a practical Mac-native tracking path, not a claim that a CPU/OpenCV algorithm
matches UDMT accuracy or implements UDMT's unsupervised deep transfer-learning network.
Scientific comparisons still require researcher-reviewed GT.

## Integration Architecture

```text
DenseTrack core
  -> no-click auto initialization from detector/segmentation
     or optional first-frame click-assisted instance selection
  -> UDMT workspace-compatible start_pos_array.txt handoff
  -> optional external UDMT environment check
  -> user runs UDMT GUI/research workflow outside DenseTrack
  -> UDMT-derived tracks exported/imported as MOT-like rows
  -> DenseTrack re-exports COCO / CSV / SLEAP JSON / provenance
  -> existing confidence, behavior, and visualization layers can consume tracks.mot
```

Current implemented bridge:

- `densetrack.external.udmt.UDMTAdapter`
- `densetrack.initialization.auto_seed`
- `densetrack auto-init`
- `densetrack udmt-handoff`
- `densetrack udmt-check`
- `densetrack udmt-import`
- `configs/udmt_external.yaml`

## Commands

Validate an external UDMT checkout:

```bash
densetrack udmt-check \
  --udmt-home /path/to/UDMT \
  --udmt-python /path/to/conda/envs/udmt/bin/python \
  --accept-license
```

If the import check is too heavy or the GUI dependencies are not ready:

```bash
densetrack udmt-check \
  --udmt-home /path/to/UDMT \
  --udmt-python /path/to/conda/envs/udmt/bin/python \
  --accept-license \
  --skip-import-check
```

Import a UDMT-derived MOT-like result into DenseTrack outputs:

```bash
densetrack udmt-import path/to/udmt_tracks.mot \
  --output results/udmt_import \
  --video-width 1280 \
  --video-height 720 \
  --fps 30
```

The import writes:

- `tracks.mot`
- `detections.coco.json`
- `poses.csv`
- `labels.sleap.json`
- `run_summary.json`

## Upstream Source Check

The official UDMT `main` branch was inspected on 2026-06-01. The documented launcher
remains the GUI command:

```bash
python -m udmt.gui.launch_script
```

The current source also includes `auto_initial_prediction.py`. It can generate
`start_pos_array.txt` automatically and falls back to manual center clicks when its
prediction fails. DenseTrack's handoff writes the same centroid file contract without
requiring UDMT source or weights in this repository.

## Next Required Work

1. Install UDMT on a Linux/CUDA workstation in a separate conda environment.
2. Run `densetrack udmt-handoff` on the target video and place its generated `start_pos_array.txt` into the official UDMT workspace path.
3. Run the official 5-mice/7-mice demo to identify the exact stable output files produced by the GUI.
4. Extend `UDMTAdapter.import_mot_tracks` if UDMT exports another native format.
5. Run UDMT on `/Users/chengneng/Project/AnimalTracker/4xPD.mp4`.
6. Generate the same DenseTrack visual outputs already used for this project: overlay video, preview clip, SLEAP-style JSON, and confidence/behavior reports.

## Source References

- Nature Methods article: https://www.nature.com/articles/s41592-026-03051-8
- UDMT project page: https://cabooster.github.io/UDMT/
- UDMT GitHub repository: https://github.com/cabooster/UDMT
- UDMT license: https://github.com/cabooster/UDMT/blob/main/LICENSE
