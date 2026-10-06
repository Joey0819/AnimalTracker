"""Re-export utilities for previous DenseTrack output directories."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

__all__ = [
    "MOTRow",
    "load_mot_rows",
    "load_run_metadata",
    "reexport_run",
]


@dataclass
class MOTRow:
    frame: int
    track_id: int
    x: float
    y: float
    width: float
    height: float
    confidence: float

    @property
    def cx(self) -> float:
        return self.x + self.width / 2

    @property
    def cy(self) -> float:
        return self.y + self.height / 2


def load_mot_rows(path: str | Path) -> list[MOTRow]:
    rows: list[MOTRow] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            rows.append(
                MOTRow(
                    frame=int(float(parts[0])),
                    track_id=int(float(parts[1])),
                    x=float(parts[2]),
                    y=float(parts[3]),
                    width=float(parts[4]),
                    height=float(parts[5]),
                    confidence=float(parts[6]),
                )
            )
    return rows


def load_run_metadata(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return {}
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def reexport_run(
    run_dir: str | Path,
    output_path: str | Path,
    fmt: str,
    min_confidence: float = 0.0,
) -> Path:
    run_dir = Path(run_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mot_path = run_dir / "tracks.mot"
    if not mot_path.exists():
        raise FileNotFoundError(f"tracks.mot not found in run directory: {run_dir}")

    rows = [row for row in load_mot_rows(mot_path) if row.confidence >= min_confidence]
    metadata = load_run_metadata(run_dir)
    fmt = fmt.lower()

    if fmt == "mot":
        _write_mot(rows, output_path, min_confidence)
    elif fmt in {"csv", "dlc"}:
        _write_centroid_csv(rows, output_path)
    elif fmt == "coco":
        _write_coco(rows, output_path, metadata, min_confidence)
    elif fmt == "sleap":
        _write_sleap(rows, output_path, metadata, min_confidence)
    else:
        raise ValueError(f"Unsupported export format: {fmt}")

    return output_path


def _write_mot(rows: list[MOTRow], path: Path, min_confidence: float) -> None:
    lines = [
        "# DenseTrack-AI MOT Re-Export",
        f"# Generated: {datetime.now().isoformat()}",
        f"# Min confidence filter: {min_confidence}",
        "# Format: frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1",
    ]
    for row in sorted(rows, key=lambda item: (item.frame, item.track_id)):
        lines.append(
            f"{row.frame},{row.track_id},{row.x:.2f},{row.y:.2f},"
            f"{row.width:.2f},{row.height:.2f},{row.confidence:.4f},-1,-1,-1"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_centroid_csv(rows: list[MOTRow], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame", "id", "x", "y", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "confidence"],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.frame, item.track_id)):
            writer.writerow(
                {
                    "frame": row.frame,
                    "id": row.track_id,
                    "x": row.cx,
                    "y": row.cy,
                    "bbox_x": row.x,
                    "bbox_y": row.y,
                    "bbox_w": row.width,
                    "bbox_h": row.height,
                    "confidence": row.confidence,
                }
            )


def _video_dimensions(metadata: dict) -> tuple[int, int]:
    video_meta = metadata.get("video_metadata", {}) if metadata else {}
    return int(video_meta.get("width", 0) or 0), int(video_meta.get("height", 0) or 0)


def _write_coco(rows: list[MOTRow], path: Path, metadata: dict, min_confidence: float) -> None:
    width, height = _video_dimensions(metadata)
    images = {}
    annotations = []
    for ann_id, row in enumerate(sorted(rows, key=lambda item: (item.frame, item.track_id)), start=1):
        images.setdefault(
            row.frame,
            {
                "id": row.frame,
                "file_name": f"frame_{row.frame:06d}.jpg",
                "width": width,
                "height": height,
            },
        )
        annotations.append(
            {
                "id": ann_id,
                "image_id": row.frame,
                "category_id": 1,
                "bbox": [row.x, row.y, row.width, row.height],
                "area": row.width * row.height,
                "iscrowd": 0,
                "score": row.confidence,
                "segmentation": [[
                    row.x,
                    row.y,
                    row.x + row.width,
                    row.y,
                    row.x + row.width,
                    row.y + row.height,
                    row.x,
                    row.y + row.height,
                ]],
                "densetrack_trajectory_id": row.track_id,
            }
        )
    coco = {
        "info": {
            "description": "DenseTrack-AI Re-Exported Auto-Labels",
            "date_created": datetime.now().isoformat(),
            "min_confidence": min_confidence,
        },
        "categories": [{"id": 1, "name": "animal", "supercategory": "organism"}],
        "images": [images[idx] for idx in sorted(images)],
        "annotations": annotations,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)


def _write_sleap(rows: list[MOTRow], path: Path, metadata: dict, min_confidence: float) -> None:
    frame_map: dict[int, list[dict]] = {}
    for row in sorted(rows, key=lambda item: (item.frame, item.track_id)):
        frame_map.setdefault(row.frame, []).append(
            {
                "track": f"animal_{row.track_id}",
                "track_id": row.track_id,
                "points": {"centroid": {"x": row.cx, "y": row.cy, "score": row.confidence}},
                "bbox_xywh": [row.x, row.y, row.width, row.height],
                "score": row.confidence,
            }
        )
    labels = {
        "format": "densetrack_sleap_json",
        "format_version": "1.0",
        "created_at": datetime.now().isoformat(),
        "video": metadata.get("video_metadata", {}) if metadata else {},
        "skeleton": {"nodes": [{"name": "centroid"}], "edges": []},
        "tracks": [
            {"name": f"animal_{track_id}", "id": track_id}
            for track_id in sorted({row.track_id for row in rows})
        ],
        "frames": [
            {"frame_idx": frame_idx, "instances": frame_map[frame_idx]}
            for frame_idx in sorted(frame_map)
        ],
        "provenance": {
            "generator": "DenseTrack-AI re-export",
            "min_confidence": min_confidence,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
