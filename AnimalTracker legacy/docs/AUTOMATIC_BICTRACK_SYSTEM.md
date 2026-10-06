# Automatic BICTrack System

This document describes the current fully automatic tracking pipeline used by
AnimalTracker for multi-animal videos with contact, occlusion, and temporary ID
loss.

## Goal

The system creates a complete automatic trajectory estimate without manual
annotation. The output is a scientific estimate with confidence and uncertainty
scores. It is not manually verified ground truth.

## Algorithm

1. Track-guided threshold detection
   - Detects dark animal blobs frame by frame.
   - Uses recent track priors to split merged blobs during close contact.

2. AnimalMOT online tracking
   - Maintains identity slots equal to the expected animal count.
   - Uses motion, local matching, virtual markers, and ReID features.
   - Suppresses excess active tracks when more tracks than animals appear.

3. Bidirectional consistency refinement
   - Runs tracklet cleanup and global linking forward in time.
   - Runs the same process backward in time.
   - Aligns forward and backward identities with Hungarian matching.
   - Fuses both estimates per frame.
   - Produces a confidence score and conflict distance for every identity point.

4. Automatic gap filling
   - Interpolates short missing segments.
   - Holds leading or trailing gaps at the nearest reliable position.
   - Keeps exactly `n_animals` continuous identity timelines.

5. Scientific estimate reporting
   - Generates occupancy heatmaps.
   - Generates uncertainty heatmaps.
   - Generates trajectory estimate maps.
   - Generates risk timelines and summary statistics.

## Main Commands

Generate automatic labels for a full video:

```bash
python scripts\auto_label_video.py --video 4xPD.mp4 --n-animals 4 --reid-interval 5
```

Generate labels for a short validation segment:

```bash
python scripts\auto_label_video.py --video 4xPD.mp4 --n-animals 4 --max-frames 300 --output-dir data\auto_annotations_bic_sample_v2
```

Generate dynamic trajectory videos:

```bash
python scripts\analyze_videos_to_trajectory.py --videos 4xPD.mp4 6xWT.mp4 --reid-interval 5
```

Run the one-click automatic research pipeline:

```bash
python scripts\run_automatic_research_pipeline.py --video 4xPD.mp4 --n-animals 4 --reid-interval 5
```

## Output Interpretation

- `output_tracks`: final number of continuous identities.
- `observed_annotations`: labels attached to nearby observed boxes.
- `interpolated_annotations`: labels filled by interpolation or prediction.
- `uncertain_annotations`: labels with interpolation, prediction, or consistency conflict.
- `bidirectional_conflict_annotations`: frames where forward and backward passes disagreed.
- `mean_conflict_distance`: average forward/backward disagreement in pixels.
- `mean confidence`: average automatic label confidence in the report.

## Current Limits

- The pipeline is fully automatic, but automatic estimates should still be
  interpreted with uncertainty scores.
- Close-contact separation is currently based on track-guided blob splitting,
  not a trained instance segmentation model.
- The next accuracy jump should come from training an instance segmentation
  model and a mouse-specific ReID encoder on the generated automatic labels,
  then feeding those models back into the same bidirectional optimizer.
