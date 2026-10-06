# BEAST3D / Dolph2Vec Integration Notes

Updated: 2026-06-14

This project now includes runnable, annotation-free extensions inspired by two
current research signals:

- BEAST3D, arXiv:2606.02937: self-supervised 3D visual representations from
  unlabeled calibrated multi-view video.
- Dolph2Vec, arXiv:2606.12503: species-specific self-supervised representations
  for dolphin vocalizations, evaluated on signature whistle classification and
  whistle detection.

## Implemented Now

### Visual SSL for Tracking Runs

Command:

```bash
./.venv/bin/python -m densetrack.cli extract-representations results/cage_001
```

Pipeline behavior:

- `representation.enabled: true` now exports `visual_ssl_report.json` and
  `visual_ssl_embeddings.csv` after MOT export.
- The extractor uses no training labels.
- Inputs are `tracks.mot`, optional `poses.csv`, optional `identity_risk.json`,
  and optional calibrated multi-view camera/session metadata.
- Features include normalized position, speed, acceleration, heading, bounding
  box geometry, nearest-neighbor distance, identity-risk score, and optional
  geometric-pose features.
- The report records temporal positive pairs, same-frame negative pairs,
  feature statistics, PCA embedding variance, and multi-view readiness.

Scientific boundary:

- This is not the BEAST3D Gaussian-splatting neural pretraining implementation.
- Full BEAST3D-style 3D pretraining still needs synchronized calibrated
  multi-view videos and a dedicated differentiable rendering training stack.
- The current value is a runnable, auditable representation layer that plugs
  into the existing markerless MOT workflow without manual training annotation.

### Bioacoustic SSL for Communication Behavior

Command:

```bash
./.venv/bin/python -m densetrack.cli analyze-audio-ssl dolphin_recording.wav \
  --species dolphin \
  --codebook-size 12 \
  --whistle-min-frequency 1000
```

Pipeline behavior:

- `bioacoustic_ssl.enabled: true` plus `audio_path` exports
  `bioacoustic_ssl_report.json` and `bioacoustic_ssl_units.csv` during a run.
- The standalone command analyzes WAV files without behavior labels.
- The extractor computes windowed spectral features, PCA coordinates, a
  deterministic codebook, and whistle-candidate windows.
- The dashboard displays acoustic windows, unit count, whistle candidates, and
  candidate ratio beside the visual tracking result.

Scientific boundary:

- This is not a trained Dolph2Vec/Wav2Vec2 model.
- Whistle candidates are unsupervised acoustic events and need domain review
  before scientific claims.
- The current value is a lightweight communication-behavior channel that can be
  paired with visual markerless tracking outputs.

## Dashboard and Outputs

The bilingual dashboard now surfaces the new reports when they exist:

- BEAST3D-inspired visual SSL block
- Dolph2Vec-inspired bioacoustic SSL block
- Download links for the four new artifacts

New artifacts:

- `visual_ssl_report.json`
- `visual_ssl_embeddings.csv`
- `bioacoustic_ssl_report.json`
- `bioacoustic_ssl_units.csv`

## Config Example

```yaml
representation:
  enabled: true
  dim: 8
  window_frames: 15
  sample_every: 3
  camera_manifest: null

bioacoustic_ssl:
  enabled: false
  audio_path: null
  species: dolphin
  window_ms: 120.0
  hop_ms: 40.0
  n_bands: 16
  codebook_size: 12
  whistle_min_frequency_hz: 1000.0
```

## Practical Next Step

For the current 4xPD mouse video, visual SSL is directly useful because tracking
and geometric pose outputs already exist. Dolph2Vec-style audio SSL is ready for
WAV inputs, but it only becomes relevant when synchronized animal vocalization
recordings are available.
