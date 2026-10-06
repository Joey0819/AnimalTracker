"""COCO instance segmentation format exporter."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from densetrack.models import Trajectory

__all__ = ["COCOExporter"]

logger = logging.getLogger(__name__)


class COCOExporter:
    """Export tracking results in COCO instance segmentation format."""

    def export(
        self,
        trajectories: list[Trajectory],
        video_metadata: dict | None = None,
        output_path: str | Path = "detections.coco.json",
        min_confidence: float = 0.0,
    ) -> None:
        """
        Export as COCO JSON.

        Adds custom fields:
        - densetrack_confidence: segment link confidence
        - densetrack_source: detection source
        - densetrack_trajectory_id: global trajectory ID
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        coco = {
            "info": {
                "description": "DenseTrack-AI Auto-Labels",
                "version": "1.0",
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [],
            "categories": [
                {"id": 1, "name": "animal", "supercategory": "organism"},
            ],
            "images": [],
            "annotations": [],
        }

        seen_frames: set[int] = set()
        ann_id = 1

        meta = video_metadata or {}
        img_width = meta.get("width", 1920)
        img_height = meta.get("height", 1080)

        for traj in trajectories:
            for seg in traj.segments:
                if seg.link_confidence < min_confidence and seg.link_confidence > 0:
                    continue

                for det in seg.tracklet.detections:
                    # Add image entry (once per frame)
                    if det.frame_idx not in seen_frames:
                        coco["images"].append({
                            "id": det.frame_idx,
                            "file_name": f"frame_{det.frame_idx:06d}.jpg",
                            "width": img_width,
                            "height": img_height,
                        })
                        seen_frames.add(det.frame_idx)

                    # Annotation
                    x, y, w, h = det.bbox_xywh
                    ann = {
                        "id": ann_id,
                        "image_id": det.frame_idx,
                        "category_id": 1,
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "area": float(det.area or w * h),
                        "iscrowd": 0,
                        "score": float(det.confidence),
                        # DenseTrack custom fields
                        "densetrack_trajectory_id": traj.global_id,
                        "densetrack_confidence": float(seg.link_confidence),
                        "densetrack_source": det.source.value,
                    }

                    # Add segmentation if available
                    if det.mask_polygon:
                        ann["segmentation"] = [det.mask_polygon]
                    else:
                        # Use bbox as polygon
                        ann["segmentation"] = [[
                            x, y, x + w, y, x + w, y + h, x, y + h
                        ]]

                    coco["annotations"].append(ann)
                    ann_id += 1

        with open(path, "w") as f:
            json.dump(coco, f, indent=2)

        logger.info(
            f"Exported {ann_id - 1} annotations across "
            f"{len(seen_frames)} frames to {path}"
        )
