# Research-Grade Tracking Workflow

This workflow is the path for reducing ID switches in dense mouse videos beyond
parameter tuning. The key idea is to build a measured loop:

1. create an annotated evaluation set,
2. train mouse-specific segmentation,
3. train mouse-specific ReID embeddings,
4. run tracking,
5. evaluate IDF1 / MOTA / ID switches,
6. apply offline global tracklet optimization.

## 1. Mine Hard Frames First

Do not annotate random frames first. Scan the video frame-by-frame and extract
the frames where the current tracker is most likely to fail:

```powershell
python scripts\mine_hard_annotation_frames.py ^
  --video 4xPD.mp4 ^
  --n-animals 4 ^
  --top-k 120 ^
  --min-gap 15 ^
  --output-dir data\annotation_mining
```

Risk signals include:

- active track count differs from expected animal count,
- detection count differs from expected animal count,
- new/lost IDs or total raw IDs exceed expected count,
- large jumps from the previous frame,
- close-contact clusters,
- low-confidence detections,
- virtual/predicted tracks.

Outputs:

- `data/annotation_mining/<video>/hard_frames.json`
- `data/annotation_mining/<video>/prelabels.json`
- `data/annotation_mining/<video>/frames/*.jpg`
- `data/annotation_mining/<video>/overlays/*.jpg`
- `data/annotation_mining/<video>/ANNOTATION_GUIDE.md`

## 2. Generate Automatic Pseudo Labels

You can automatically label every frame before manual review:

```powershell
python scripts\auto_label_video.py ^
  --video 4xPD.mp4 ^
  --n-animals 4 ^
  --output-dir data\auto_annotations ^
  --review-top-k 200
```

Outputs:

- `data/auto_annotations/<video>/auto_annotations.json`
- `data/auto_annotations/<video>/review_queue.json`
- `data/auto_annotations/<video>/review_frames/*.jpg`
- `data/auto_annotations/<video>/review_overlays/*.jpg`

The automatic file contains labels for every processed frame. Each annotation
has:

- `source=auto` for directly observed tracker boxes,
- `source=interpolated` when a missing frame was filled from neighboring frames,
- `source=predicted` when a leading/trailing gap was filled from the nearest track state,
- `review_required=true` for high-risk or non-observed labels.

This is the practical compromise: the system labels every frame automatically,
then sends only uncertain frames to people for confirmation. The reviewed output
is the training/evaluation ground truth; the raw auto labels are pseudo labels.

Use the guided annotation UI to review suggestions:

```powershell
streamlit run scripts\guided_annotation_app.py -- ^
  --mining-dir data\annotation_mining\4xPD
```

The app shows the raw frame, overlay, risk reasons, suggested IDs, suggested
boxes, and an editable table. Save reviewed labels to:

- `data/annotation_mining/<video>/annotations_reviewed.json`

## 3. Build An Annotation Set

Extract representative frames from each video. Use a smaller stride around
normal behavior and dense sampling around contact/occlusion windows.

```powershell
python scripts\build_research_dataset.py --video 4xPD.mp4 --stride 15 --output-dir data\research_dataset
```

The script creates:

- `data/research_dataset/<video>/frames/frame_000000.jpg`
- `data/research_dataset/<video>/annotations.json`
- `data/research_dataset/<video>/dataset_meta.json`

Use CVAT, Label Studio, or a simple manual tool to annotate every visible mouse
with a stable identity. Prefer polygon/mask annotations for clustered frames.

Project annotation schema:

```json
{
  "schema": "animaltracker.research_annotations.v1",
  "videos": [{"id": "4xPD", "file_name": "data/videos/4xPD.mp4"}],
  "annotations": [
    {
      "video_id": "4xPD",
      "frame": 120,
      "identity": 1,
      "bbox": [100, 120, 170, 185],
      "mask": [[100, 120, 170, 121, 168, 185, 102, 184]],
      "occluded": false
    }
  ]
}
```

## 4. Export Segmentation And ReID Data

Convert the annotation file into YOLO segmentation labels and ReID identity
crops.

```powershell
python scripts\prepare_training_data.py ^
  --annotations data\annotation_mining\4xPD\annotations_reviewed.json ^
  --video data\videos\4xPD.mp4 ^
  --output-dir data\training
```

Outputs:

- `data/training/yolo_mouse_seg/dataset.yaml`
- `data/training/yolo_mouse_seg/images/train/*.jpg`
- `data/training/yolo_mouse_seg/labels/train/*.txt`
- `data/training/reid_mouse/id_001/*.jpg`
- `data/training/reid_mouse/manifest.csv`

## 5. Train Mouse-Specific Segmentation

```powershell
python scripts\train_segmentation_yolo.py ^
  --data data\training\yolo_mouse_seg\dataset.yaml ^
  --model yolov8s-seg.pt ^
  --epochs 100 ^
  --imgsz 640 ^
  --batch 8 ^
  --device cuda:0
```

Use the trained model for detection/segmentation before tracking. This is the
highest-impact upgrade because clustered mice fail first at the detection layer.

## 6. Train Mouse ReID Embeddings

```powershell
python scripts\train_reid_classifier.py ^
  --data-dir data\training\reid_mouse ^
  --output data\models\reid\mouse_reid.pt ^
  --epochs 30 ^
  --batch-size 32 ^
  --device cuda:0
```

This creates a supervised identity embedding model. It should replace generic
ImageNet features when mice have similar color but still have small shape,
texture, tail, or pose differences.

## 7. Evaluate Against Ground Truth

After running the tracker, export predictions as project annotations or a simple
`{frame: {track_id: bbox}}` JSON. Then evaluate:

```powershell
python scripts\evaluate_tracking_against_annotations.py ^
  --gt data\annotation_mining\4xPD\annotations_reviewed.json ^
  --pred data\results\tracking_predictions.json ^
  --output data\results\mot_metrics_4xPD.json ^
  --iou-threshold 0.30
```

Core metrics:

- `idf1`: identity preservation, most important for ID switch reduction.
- `id_switches`: direct count of identity changes.
- `fragments`: how often a ground-truth trajectory disappears and reappears.
- `mota`: combined FP/FN/ID-switch tracking accuracy.
- `motp_iou` and `motp_center_px`: localization quality.
- `hota_proxy`: lightweight detection/association combined score.

## 8. Offline Global Optimization

Use online tracking to create short tracklets, then connect them globally with
motion, time-gap, and appearance costs. The code path is:

- `src/tracking/offline_global_optimizer.py`
- `scripts/analyze_videos_to_trajectory.py`

This is intentionally separate from online tracking because scientific results
should be globally optimized after the full video is available.

## Acceptance Target

For publication-grade mouse tracking, set target gates before using behavior
statistics:

- `IDF1 >= 0.90` on manually annotated validation clips.
- `id_switches <= 1 per 1000 frames` in dense interaction windows.
- `recall >= 0.95` for visible animals.
- Always report failure cases: full occlusion, wall contact, severe blur, and
  frames where animals are visually inseparable.
