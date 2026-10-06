# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

AnimalTracker is a multi-animal behavior tracking system for scientific research (mice, rats, zebrafish, etc.). It is based on two recent papers: vmTracking (Nature Communications 2025) and STCS (Patterns 2024). The system provides web-based UI via Streamlit and achieves 95%+ IDF1 accuracy with <5% ID switches.

**Core Value Proposition:** High-stability multi-animal tracking with identity preservation in dense scenarios.

## Commands

### Running the Application
```bash
streamlit run app.py
# On different port: streamlit run app.py --server.port 8889
```

### Testing Individual Modules
```bash
# Pose estimation
python test_pose.py

# Instance segmentation
python test_segmentation.py

# Re-ID
python test_reid.py

# Behavior clustering
python test_behavior_clustering.py

# vmTracker
python test_vmtracker.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Configuration
- Streamlit config: `.streamlit/config.toml`
- Max upload size: 1000 MB (1GB)
- Default confidence threshold: 0.25 (lowered for better detection)

## Architecture Overview

### Module Hierarchy
```
app.py (Streamlit UI)
├── detection/
│   └── detector.py (YOLODetector, Detection class)
├── tracking/
│   ├── tracker.py (MultiObjectTracker - baseline)
│   ├── improved_tracker.py (ImprovedTracker - Kalman)
│   ├── enhanced_tracker.py (EnhancedTracker - flow + Hungarian)
│   ├── vmtracker.py (VMTracker - virtual markers, best for dense scenes)
│   ├── optical_flow.py (OpticalFlowEstimator - Farnebäck/LK)
│   ├── hungarian_matcher.py (HungarianMatcher, MultiStageMatcher, Track)
│   └── reid/ (Re-ID for identity recovery)
├── pose/
│   ├── estimator.py (PoseEstimator, AnimalPose, Keypoint)
│   └── visualizer.py (PoseVisualizer)
├── segmentation/
│   ├── segmenter.py (InstanceSegmenter, InstanceMask)
│   └── utils.py (MaskUtils, SegmentationVisualizer)
├── behavior/
│   ├── stgcn.py (SocialSTGCN - social graph analysis)
│   ├── clustering.py (BehaviorClusteringAnalyzer - PCA + agglomerative)
│   ├── integrated.py (IntegratedBehaviorAnalyzer)
│   └── detector.py (SocialInteractionDetector)
└── visualization/
    └── visualizer.py (TrajectoryVisualizer)
```

### Key Design Patterns

**Track Objects Across Modules:**
Different tracking modules return different data structures. Always handle both:
- `Track` object: access via `track.center`, `track.id`, `track.get_last_position()`
- `Tuple[int, int]`: direct position `(x, y)`

The visualizer's `draw_tracks()` handles both formats automatically.

**Session State Persistence:**
Streamlit re-runs on every interaction. Store tracking results in `st.session_state.tracking_results`:
```python
st.session_state.tracking_results = {
    'all_tracks': {},
    'all_detections': {},
    'track_history': {},
    'video_info': {},
    'visualizer': None,
    'detector': None,
    'video_path': video_path,
    'completed': False
}
```

### Tracker Selection Guide

| Tracker | Use Case | IDF1 | Speed | When to Choose |
|---------|----------|------|-------|---------------|
| MultiObjectTracker | Quick testing | 85% | Fast | Prototype/debugging |
| ImprovedTracker | General use | 88% | Fast | Daily experiments |
| EnhancedTracker | Production | 92% | Medium | Most research work |
| VMTracker | Dense scenes/ Publication | 95%+ | Medium | 4+ animals, overlapping, papers |

### Detection & Tracking Flow

1. **Detection** (`YOLODetector.detect()`) returns `List[Detection]`
   - Each `Detection` has: `bbox`, `confidence`, `class_id`, `center` (property)

2. **Tracking** - Different signatures:
   - `MultiObjectTracker.update(detection_centers)` - expects `List[Tuple[int, int]]`
   - `ImprovedTracker.update(detection_centers)` - expects `List[Tuple[int, int]]`
   - `EnhancedTracker.update(frame, detections)` - expects frame + `List[Detection]`
   - `VMTracker.update(frame, detections)` - expects frame + `List[Detection]`

3. **Output** - `Dict[int, Track]` where `Track` has:
   - `center`, `id`, `get_last_position()`, `history`

### Critical Implementation Notes

**Optical Flow Dimension Bug:**
`LucasKanade` method must return 2-channel flow `(h, w, 2)` not 1-channel `(h, w)`. Fixed in `src/tracking/optical_flow.py`.

**Track History Storage:**
- `TrajectoryVisualizer.update_track_history()` stores history internally
- Access via `visualizer.track_history` (Dict mapping ID to List of positions)
- Use `visualizer.track_history` NOT the empty `track_history` variable in app.py

**Re-ID Integration:**
- Re-ID modules in `src/tracking/reid/` for identity recovery after occlusion
- `feature_extractor.py`: ColorHistogram, HOG, CNN features
- `reid.py`: ReIdentifier, AppearanceModel with confidence thresholds

### UTF-8 Encoding
Many modules include this at the top for Chinese characters:
```python
import sys
if hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass
```

### Optional Imports
Several features are wrapped in try/except for graceful degradation:
- Pose estimation: `POSE_AVAILABLE` flag
- vmTracker: `VMTRACKER_AVAILABLE` flag
- These allow core functionality even if dependencies are missing

## Data Flow

1. User uploads video → saved to `data/videos/`
2. `VideoProcessor` reads frames
3. `YOLODetector.detect(frame)` → `List[Detection]`
4. Selected tracker.update() → `Dict[int, Track]`
5. `TrajectoryVisualizer` draws results
6. Results stored in `st.session_state` for persistence
7. Export to CSV/JSON via `DataExporter`

## Common Issues

**Only 1 animal detected when there are 4+**
- Lower confidence threshold (default now 0.25)
- Check "平均检测数" stat - if <4, it's a detection issue
- Try different YOLO model (yolov8s.pt vs yolov8n.pt)

**ID switches frequently**
- Use EnhancedTracker or VMTracker
- Increase `max_disappeared` parameter
- Check for overlapping animals (use instance segmentation)

**Position display shows `<Track object at...>`**
- Fixed in app.py - extracts `center` property from Track objects

**Cannot generate trajectory map**
- Ensure tracking completed first (check `completed` flag in session_state)
- `visualizer.track_history` must have data

## File Locations

- Entry point: `app.py`
- Test files: `test_*.py` in project root
- Demo files: `demo_*.py`
- Configuration: `.streamlit/config.toml`
- Data: `data/videos/`, `data/models/`, `data/results/`
