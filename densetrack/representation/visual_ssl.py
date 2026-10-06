"""Annotation-free visual SSL representations for tracked animal behavior.

This module is inspired by the BEAST3D direction: use unlabeled experimental
video to learn transferable behavior features. It is not a Gaussian-splatting
implementation. It produces deterministic, auditable transfer embeddings from
MOT trajectories, optional pose features, identity-risk signals, and optional
multi-view camera metadata.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from densetrack.utils.metrics import load_mot_file

__all__ = ["VisualSSLRepresentationAnalyzer", "VisualSSLRepresentationConfig"]


@dataclass
class VisualSSLRepresentationConfig:
    """Controls unlabeled visual representation export."""

    dim: int = 8
    window_frames: int = 15
    sample_every: int = 3
    camera_manifest: str | None = None


class VisualSSLRepresentationAnalyzer:
    """Build BEAST3D-inspired transfer features without training labels."""

    base_features = [
        "x_norm",
        "y_norm",
        "speed_norm",
        "acceleration_norm",
        "heading_sin",
        "heading_cos",
        "bbox_area_norm",
        "aspect_ratio",
        "nearest_neighbor_norm",
        "identity_risk_score",
    ]

    pose_features = [
        "pose_orientation_sin",
        "pose_orientation_cos",
        "pose_body_length_norm",
        "pose_body_area_norm",
    ]

    def __init__(self, config: VisualSSLRepresentationConfig | None = None):
        self.config = config or VisualSSLRepresentationConfig()

    def analyze_run(
        self,
        run_dir: str | Path,
        output_json: str | Path | None = None,
        output_csv: str | Path | None = None,
    ) -> dict[str, Any]:
        """Analyze a completed DenseTrack run and write SSL representation artifacts."""
        run_path = Path(run_dir).expanduser().resolve()
        mot_path = run_path / "tracks.mot"
        df = load_mot_file(mot_path)
        if df.empty:
            report = self._empty_report(run_path)
            self._write_outputs(report, [], output_json, output_csv)
            return report

        risk_by_frame = self._load_identity_risk(run_path / "identity_risk.json")
        df, pose_available = self._merge_pose_features(df, run_path / "poses.csv")
        samples, feature_names = self._extract_samples(df, risk_by_frame, pose_available)
        if not samples:
            report = self._empty_report(run_path)
            self._write_outputs(report, [], output_json, output_csv)
            return report

        feature_matrix = np.asarray(
            [[float(sample[name]) for name in feature_names] for sample in samples],
            dtype=np.float64,
        )
        standardized, means, scales = self._standardize(feature_matrix)
        embeddings, explained = self._pca(standardized, dim=self.config.dim)
        rows = []
        for idx, sample in enumerate(samples):
            row = {
                "frame": int(sample["frame"]),
                "track_id": int(sample["track_id"]),
                "view_id": str(sample.get("view_id", "single_view")),
            }
            for feature_name in feature_names:
                row[feature_name] = float(sample[feature_name])
            for dim_idx in range(embeddings.shape[1]):
                row[f"ssl_{dim_idx}"] = float(embeddings[idx, dim_idx])
            rows.append(row)

        positives, negatives = self._pair_counts(rows)
        camera_report = self._camera_manifest_report()
        report = {
            "created_at": datetime.now().isoformat(),
            "format": "densetrack_visual_ssl_report",
            "format_version": "1.0",
            "source_run": str(run_path),
            "annotation_required": False,
            "paper_signal": {
                "name": "BEAST3D",
                "arxiv": "2606.02937",
                "used_as": (
                    "Design signal for unlabeled multi-view/3D-ready behavior representations; "
                    "this module does not implement differentiable Gaussian splatting."
                ),
            },
            "config": asdict(self.config),
            "summary": {
                "num_samples": len(rows),
                "num_ids": len({row["track_id"] for row in rows}),
                "frame_range": [
                    min(row["frame"] for row in rows),
                    max(row["frame"] for row in rows),
                ],
                "embedding_dim": int(embeddings.shape[1]),
                "feature_count": len(feature_names),
                "pose_features_used": pose_available,
                "temporal_positive_pairs": positives,
                "same_frame_negative_pairs": negatives,
                "multiview_status": camera_report["status"],
                "multiview_ready": camera_report["ready"],
            },
            "features": {
                "names": feature_names,
                "means": means.tolist(),
                "scales": scales.tolist(),
                "pca_explained_variance_ratio": explained.tolist(),
            },
            "camera_manifest": camera_report,
            "limitations": [
                "This is a deterministic unlabeled transfer-feature export, not BEAST3D model training.",
                "Full 3D Gaussian-splatting pretraining still requires synchronized calibrated multi-view videos.",
            ],
        }
        self._write_outputs(report, rows, output_json, output_csv)
        return report

    def _extract_samples(
        self,
        df,
        risk_by_frame: dict[int, float],
        pose_available: bool,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        df = df.copy()
        df["Cx"] = df["X"] + df["Width"] / 2.0
        df["Cy"] = df["Y"] + df["Height"] / 2.0
        df["Area"] = df["Width"] * df["Height"]
        frame_w = max(float((df["X"] + df["Width"]).max()), 1.0)
        frame_h = max(float((df["Y"] + df["Height"]).max()), 1.0)
        area_scale = max(float(df["Area"].quantile(0.95)), 1.0)
        neighbor = self._nearest_neighbors(df)
        feature_names = list(self.base_features)
        if pose_available:
            feature_names.extend(self.pose_features)

        rows = []
        for track_id, group in df.groupby("Id"):
            group = group.sort_values("FrameId").reset_index(drop=True)
            if len(group) < 2:
                continue
            frames = group["FrameId"].to_numpy(dtype=int)
            coords = group[["Cx", "Cy"]].to_numpy(dtype=float)
            dt = np.diff(frames, prepend=frames[0] - 1).astype(float)
            dt = np.maximum(dt, 1.0)
            velocity = np.diff(coords, axis=0, prepend=coords[[0]]) / dt[:, None]
            speed = np.linalg.norm(velocity, axis=1)
            speed_scale = max(float(np.quantile(speed, 0.95)), 1.0)
            acceleration = np.abs(np.diff(speed, prepend=speed[0])) / dt
            acceleration_scale = max(float(np.quantile(acceleration, 0.95)), 1.0)
            heading = np.arctan2(velocity[:, 1], velocity[:, 0])

            feature_columns = {
                "x_norm": coords[:, 0] / frame_w,
                "y_norm": coords[:, 1] / frame_h,
                "speed_norm": speed / speed_scale,
                "acceleration_norm": acceleration / acceleration_scale,
                "heading_sin": np.sin(heading),
                "heading_cos": np.cos(heading),
                "bbox_area_norm": group["Area"].to_numpy(dtype=float) / area_scale,
                "aspect_ratio": group["Width"].to_numpy(dtype=float)
                / np.maximum(group["Height"].to_numpy(dtype=float), 1e-6),
                "nearest_neighbor_norm": np.asarray(
                    [
                        neighbor.get((int(frame), int(track_id)), max(frame_w, frame_h))
                        / max(frame_w, frame_h)
                        for frame in frames
                    ],
                    dtype=float,
                ),
                "identity_risk_score": np.asarray(
                    [risk_by_frame.get(int(frame), 0.0) for frame in frames],
                    dtype=float,
                ),
            }
            if pose_available:
                feature_columns.update(self._pose_columns(group, area_scale))

            smoothed = {
                name: self._rolling_mean(np.asarray(values, dtype=float), self.config.window_frames)
                for name, values in feature_columns.items()
            }
            for idx in range(0, len(group), max(1, self.config.sample_every)):
                row = {
                    "frame": int(frames[idx]),
                    "track_id": int(track_id),
                    "view_id": "single_view",
                }
                for name in feature_names:
                    row[name] = float(smoothed[name][idx])
                rows.append(row)
        return rows, feature_names

    @staticmethod
    def _pose_columns(group, area_scale: float) -> dict[str, np.ndarray]:
        orientation = group.get("pose_orientation")
        if orientation is None:
            orientation_values = np.zeros(len(group), dtype=float)
        else:
            orientation_values = _fill_missing(orientation.to_numpy(dtype=float))
        body_length = group.get("pose_body_length_px")
        body_area = group.get("pose_body_area_px2")
        return {
            "pose_orientation_sin": np.sin(orientation_values),
            "pose_orientation_cos": np.cos(orientation_values),
            "pose_body_length_norm": _fill_missing(
                body_length.to_numpy(dtype=float) if body_length is not None else np.zeros(len(group))
            )
            / max(float(group["Width"].quantile(0.95)), 1.0),
            "pose_body_area_norm": _fill_missing(
                body_area.to_numpy(dtype=float) if body_area is not None else np.zeros(len(group))
            )
            / area_scale,
        }

    @staticmethod
    def _merge_pose_features(df, pose_path: Path):
        if not pose_path.exists():
            return df, False
        import pandas as pd

        pose_df = pd.read_csv(pose_path)
        if "frame" not in pose_df or "id" not in pose_df:
            return df, False
        pose_df = pose_df.rename(columns={"frame": "FrameId", "id": "Id"})
        columns = ["FrameId", "Id"]
        available = False
        for source, target in [
            ("orientation", "pose_orientation"),
            ("body_length", "pose_body_length_px"),
            ("body_area", "pose_body_area_px2"),
        ]:
            if source in pose_df:
                pose_df[target] = pd.to_numeric(pose_df[source], errors="coerce")
                columns.append(target)
                available = True
        if not available:
            return df, False
        pose_df = pose_df.groupby(["FrameId", "Id"], as_index=False).first()
        return df.merge(pose_df[columns], on=["FrameId", "Id"], how="left"), True

    @staticmethod
    def _nearest_neighbors(df) -> dict[tuple[int, int], float]:
        result = {}
        for frame_id, group in df.groupby("FrameId"):
            ids = group["Id"].to_numpy(dtype=int)
            coords = group[["Cx", "Cy"]].to_numpy(dtype=float)
            if len(ids) < 2:
                continue
            distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            for idx, track_id in enumerate(ids):
                result[(int(frame_id), int(track_id))] = float(np.min(distances[idx]))
        return result

    @staticmethod
    def _load_identity_risk(path: Path) -> dict[int, float]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return {
            int(row["frame"]): float(row.get("risk_score", 0.0))
            for row in payload.get("frames", [])
            if "frame" in row
        }

    @staticmethod
    def _standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        means = np.mean(features, axis=0)
        scales = np.std(features, axis=0)
        scales[scales < 1e-6] = 1.0
        return np.clip((features - means) / scales, -8.0, 8.0), means, scales

    @staticmethod
    def _pca(features: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
        centered = features - np.mean(features, axis=0)
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        output_dim = max(1, min(dim, vt.shape[0]))
        embedding = centered @ vt[:output_dim].T
        if output_dim < dim:
            embedding = np.column_stack([embedding, np.zeros((len(features), dim - output_dim))])
        variance = singular**2
        total = max(float(np.sum(variance)), 1e-12)
        explained = np.zeros(dim, dtype=float)
        explained[:output_dim] = variance[:output_dim] / total
        return embedding.astype(float), explained

    @staticmethod
    def _pair_counts(rows: list[dict[str, Any]]) -> tuple[int, int]:
        by_track: dict[int, int] = {}
        by_frame: dict[int, int] = {}
        for row in rows:
            by_track[int(row["track_id"])] = by_track.get(int(row["track_id"]), 0) + 1
            by_frame[int(row["frame"])] = by_frame.get(int(row["frame"]), 0) + 1
        positives = sum(max(0, count - 1) for count in by_track.values())
        negatives = sum(count * (count - 1) // 2 for count in by_frame.values())
        return int(positives), int(negatives)

    def _camera_manifest_report(self) -> dict[str, Any]:
        if not self.config.camera_manifest:
            return {
                "status": "single_view_only / 仅单视角",
                "ready": False,
                "views": 1,
                "calibrated_views": 0,
                "note": "Provide synchronized calibrated multi-view metadata for BEAST3D-style 3D pretraining.",
            }
        path = Path(self.config.camera_manifest).expanduser()
        if not path.exists():
            return {
                "status": "manifest_missing / 多视角清单缺失",
                "ready": False,
                "views": 0,
                "calibrated_views": 0,
                "path": str(path),
            }
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        views = payload.get("views", [])
        calibrated = [
            view
            for view in views
            if view.get("intrinsics") is not None and view.get("extrinsics") is not None
        ]
        return {
            "status": "multiview_calibrated / 多视角已标定"
            if len(calibrated) >= 2
            else "multiview_unready / 多视角未就绪",
            "ready": len(calibrated) >= 2,
            "views": len(views),
            "calibrated_views": len(calibrated),
            "path": str(path),
        }

    @staticmethod
    def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return values
        kernel = np.ones(window, dtype=float)
        return np.convolve(values, kernel, mode="same") / np.maximum(
            np.convolve(np.ones(len(values)), kernel, mode="same"),
            1.0,
        )

    @staticmethod
    def _write_outputs(
        report: dict[str, Any],
        rows: list[dict[str, Any]],
        output_json: str | Path | None,
        output_csv: str | Path | None,
    ) -> None:
        if output_json:
            target = Path(output_json)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if output_csv:
            target = Path(output_csv)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not rows:
                target.write_text("", encoding="utf-8")
                return
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    @staticmethod
    def _empty_report(run_path: Path) -> dict[str, Any]:
        return {
            "created_at": datetime.now().isoformat(),
            "format": "densetrack_visual_ssl_report",
            "format_version": "1.0",
            "source_run": str(run_path),
            "annotation_required": False,
            "summary": {
                "num_samples": 0,
                "num_ids": 0,
                "embedding_dim": 0,
                "multiview_ready": False,
            },
            "features": {"names": [], "means": [], "scales": []},
        }


def _fill_missing(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(len(values), dtype=float)
    indices = np.arange(len(values))
    return np.interp(indices, indices[finite], values[finite])
