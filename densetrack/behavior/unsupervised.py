"""Unsupervised behavior discovery from MOT trajectories.

This is a lightweight B-SOiD-inspired baseline for environments where pose
keypoints are not yet available. It clusters motion, shape, and proximity
features without behavior annotations.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from densetrack.utils.metrics import load_mot_file

__all__ = ["BehaviorDiscoveryConfig", "UnsupervisedBehaviorDiscovery"]


@dataclass
class BehaviorDiscoveryConfig:
    """Configuration for annotation-free behavior clustering."""

    fps: float = 15.0
    n_clusters: int = 5
    window_frames: int = 15
    sample_every: int = 3
    contact_distance_px: float = 95.0
    max_iterations: int = 100
    pose_csv: str | None = None
    feature_mode: str = "auto"
    embedding_method: str = "pca"
    clustering_method: str = "kmeans"
    hdbscan_min_cluster_size: int = 20
    hdbscan_min_samples: int | None = None


class UnsupervisedBehaviorDiscovery:
    """Cluster trajectory-derived features into recurring behavioral motifs."""

    base_feature_names = [
        "speed_px_s",
        "acceleration_px_s2",
        "turn_rate_rad_s",
        "bbox_area_px2",
        "aspect_ratio",
        "nearest_neighbor_px",
        "close_neighbor",
    ]

    def __init__(self, config: BehaviorDiscoveryConfig | None = None):
        self.config = config or BehaviorDiscoveryConfig()
        self.feature_names = list(self.base_feature_names)

    def analyze_mot(
        self,
        mot_path: str | Path,
        output_json: str | Path | None = None,
        output_csv: str | Path | None = None,
        embedding_image: str | Path | None = None,
        timeline_image: str | Path | None = None,
    ) -> dict[str, Any]:
        """Discover clusters from a MOT file and write optional outputs."""
        df = load_mot_file(mot_path)
        if df.empty:
            report = self._empty_report(mot_path)
            self._write_outputs(report, [], output_json, output_csv)
            return report

        df, pose_feature_names = self._merge_pose_features(df)
        self.feature_names = list(self.base_feature_names) + pose_feature_names
        samples = self._extract_samples(df)
        if not samples:
            report = self._empty_report(mot_path)
            self._write_outputs(report, [], output_json, output_csv)
            return report

        feature_matrix = np.array(
            [[sample[name] for name in self.feature_names] for sample in samples],
            dtype=float,
        )
        standardized, means, scales = self._standardize(feature_matrix)
        embedding = self._embedding_2d(standardized)
        labels, centers = self._cluster(standardized, embedding)
        cluster_names = self._cluster_names(samples, labels)

        timeline_rows = []
        for idx, sample in enumerate(samples):
            row = dict(sample)
            row["cluster"] = int(labels[idx])
            row["behavior_label"] = cluster_names[int(labels[idx])]
            row["embedding_x"] = float(embedding[idx, 0])
            row["embedding_y"] = float(embedding[idx, 1])
            timeline_rows.append(row)

        report = {
            "created_at": datetime.now().isoformat(),
            "source": str(mot_path),
            "method": (
                "densetrack_bsoid_inspired_pose_clustering"
                if pose_feature_names
                else "densetrack_bsoid_inspired_centroid_clustering"
            ),
            "annotation_required": False,
            "config": asdict(self.config),
            "summary": {
                "num_ids": len({int(sample["track_id"]) for sample in samples}),
                "num_samples": len(samples),
                "num_clusters": len(centers),
                "frame_range": [
                    min(int(sample["frame"]) for sample in samples),
                    max(int(sample["frame"]) for sample in samples),
                ],
                "feature_mode": "pose_enriched" if pose_feature_names else "centroid_bbox",
                "embedding_method": self.config.embedding_method,
                "clustering_method": self.config.clustering_method,
                "clustering_space": self._clustering_space(),
                "num_noise_samples": int(np.sum(labels == -1)),
            },
            "features": {
                "names": self.feature_names,
                "means": means.tolist(),
                "scales": scales.tolist(),
            },
            "clusters": self._cluster_summary(samples, labels, cluster_names),
            "limitations": [
                "Clusters are recurring motion motifs, not validated semantic behavior labels.",
                (
                    "Pose-enriched features are active. Keypoint-level quality still depends on the "
                    "configured pose backend."
                    if pose_feature_names
                    else "This baseline uses MOT centroid/bbox features. Provide poses.csv from DLC "
                    "SuperAnimal or the geometry backend to enable pose enrichment."
                ),
            ],
        }
        self._write_outputs(report, timeline_rows, output_json, output_csv)

        if embedding_image:
            self._draw_embedding(embedding, labels, cluster_names, embedding_image)
        if timeline_image:
            self._draw_timeline(timeline_rows, cluster_names, timeline_image)

        return report

    def _merge_pose_features(self, df):
        """Attach optional geometric and keypoint pose columns from poses.csv."""
        mode = self.config.feature_mode.lower()
        if mode not in {"auto", "centroid", "pose"}:
            raise ValueError(f"Unknown feature mode / 未知特征模式: {self.config.feature_mode}")
        if mode == "centroid":
            return df, []
        if not self.config.pose_csv:
            if mode == "pose":
                raise ValueError("Pose mode requires --pose-csv / 姿态模式需要 --pose-csv")
            return df, []

        import pandas as pd

        path = Path(self.config.pose_csv).expanduser()
        if not path.exists():
            if mode == "pose":
                raise FileNotFoundError(f"Pose CSV not found / 未找到姿态 CSV: {path}")
            return df, []
        pose_df = pd.read_csv(path)
        if "frame" not in pose_df or "id" not in pose_df:
            raise ValueError("Pose CSV requires frame,id columns / 姿态 CSV 需要 frame,id 列")
        pose_df = pose_df.rename(columns={"frame": "FrameId", "id": "Id"})
        pose_df = pose_df.groupby(["FrameId", "Id"], as_index=False).first()
        columns = ["FrameId", "Id"]
        pose_features: list[str] = []

        for source, target in [
            ("orientation", "pose_orientation"),
            ("body_length", "pose_body_length_px"),
            ("body_area", "pose_body_area_px2"),
        ]:
            if source in pose_df and pd.to_numeric(pose_df[source], errors="coerce").notna().any():
                pose_df[target] = pd.to_numeric(pose_df[source], errors="coerce")
                columns.append(target)
                if source == "orientation":
                    pose_df["pose_orientation_sin"] = np.sin(pose_df[target])
                    pose_df["pose_orientation_cos"] = np.cos(pose_df[target])
                    columns.extend(["pose_orientation_sin", "pose_orientation_cos"])
                    pose_features.extend(["pose_orientation_sin", "pose_orientation_cos"])
                else:
                    pose_features.append(target)

        keypoint_indices = sorted(
            {
                int(match.group(1))
                for column in pose_df.columns
                if (match := re.fullmatch(r"kp(\d+)_x", str(column)))
                and f"kp{match.group(1)}_y" in pose_df
            }
        )
        for index in keypoint_indices:
            for axis in ("x", "y"):
                source = f"kp{index}_{axis}"
                if pd.to_numeric(pose_df[source], errors="coerce").notna().any():
                    target = f"pose_kp{index}_rel_{axis}"
                    pose_df[target] = pd.to_numeric(pose_df[source], errors="coerce")
                    columns.append(target)
                    pose_features.append(target)
            confidence = f"kp{index}_conf"
            if confidence in pose_df and pd.to_numeric(pose_df[confidence], errors="coerce").notna().any():
                target = f"pose_kp{index}_confidence"
                pose_df[target] = pd.to_numeric(pose_df[confidence], errors="coerce")
                columns.append(target)
                pose_features.append(target)
        if not pose_features and mode == "pose":
            raise ValueError("Pose CSV has no usable pose features / 姿态 CSV 没有可用姿态特征")
        if not pose_features:
            return df, []
        return df.merge(pose_df[columns], on=["FrameId", "Id"], how="left"), pose_features

    def _extract_samples(self, df) -> list[dict[str, float | int]]:
        df = df.copy()
        df["Cx"] = df["X"] + df["Width"] / 2
        df["Cy"] = df["Y"] + df["Height"] / 2
        df["Area"] = df["Width"] * df["Height"]
        df["Aspect"] = df["Width"] / np.maximum(df["Height"], 1e-6)
        for name in self.feature_names:
            match = re.fullmatch(r"pose_kp\d+_rel_([xy])", name)
            if match:
                df[name] = df[name] - df["Cx" if match.group(1) == "x" else "Cy"]
        nearest = self._nearest_neighbors(df)

        samples: list[dict[str, float | int]] = []
        for track_id, group in df.groupby("Id"):
            group = group.sort_values("FrameId").reset_index(drop=True)
            if len(group) < 3:
                continue

            coords = group[["Cx", "Cy"]].to_numpy(dtype=float)
            frames = group["FrameId"].to_numpy(dtype=int)
            dt = np.diff(frames, prepend=frames[0] - 1) / max(self.config.fps, 1e-6)
            dt = np.maximum(dt, 1.0 / max(self.config.fps, 1e-6))
            velocity = np.diff(coords, axis=0, prepend=coords[[0]]) / dt[:, None]
            speed = np.linalg.norm(velocity, axis=1)
            acceleration = np.abs(np.diff(speed, prepend=speed[0])) / dt
            heading = np.arctan2(velocity[:, 1], velocity[:, 0])
            turn_rate = np.abs(self._angle_delta(heading)) / dt

            feature_columns = {
                "speed_px_s": speed,
                "acceleration_px_s2": acceleration,
                "turn_rate_rad_s": turn_rate,
                "bbox_area_px2": group["Area"].to_numpy(dtype=float),
                "aspect_ratio": group["Aspect"].to_numpy(dtype=float),
                "nearest_neighbor_px": np.array(
                    [nearest.get((int(frame), int(track_id)), self.config.contact_distance_px * 3) for frame in frames],
                    dtype=float,
                ),
            }
            feature_columns["close_neighbor"] = (
                feature_columns["nearest_neighbor_px"] <= self.config.contact_distance_px
            ).astype(float)
            for name in self.feature_names:
                if name not in feature_columns:
                    feature_columns[name] = self._fill_missing(
                        group[name].to_numpy(dtype=float)
                    )

            smoothed = {
                name: self._rolling_mean(values, self.config.window_frames)
                for name, values in feature_columns.items()
            }
            for idx in range(0, len(group), max(1, self.config.sample_every)):
                sample: dict[str, float | int] = {
                    "frame": int(frames[idx]),
                    "track_id": int(track_id),
                }
                for name in self.feature_names:
                    sample[name] = float(smoothed[name][idx])
                samples.append(sample)
        return samples

    @staticmethod
    def _fill_missing(values: np.ndarray) -> np.ndarray:
        """Interpolate sparse pose rows while preserving a deterministic fallback."""
        values = values.astype(float)
        finite = np.isfinite(values)
        if not np.any(finite):
            return np.zeros(len(values), dtype=float)
        indices = np.arange(len(values))
        return np.interp(indices, indices[finite], values[finite])

    @staticmethod
    def _nearest_neighbors(df) -> dict[tuple[int, int], float]:
        result: dict[tuple[int, int], float] = {}
        for frame_id, group in df.groupby("FrameId"):
            ids = group["Id"].to_numpy(dtype=int)
            centers = group[["Cx", "Cy"]].to_numpy(dtype=float)
            if len(ids) < 2:
                continue
            diffs = centers[:, None, :] - centers[None, :, :]
            distances = np.sqrt(np.sum(diffs * diffs, axis=2))
            np.fill_diagonal(distances, np.inf)
            for idx, track_id in enumerate(ids):
                result[(int(frame_id), int(track_id))] = float(np.min(distances[idx]))
        return result

    @staticmethod
    def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return values.astype(float)
        kernel = np.ones(window, dtype=float)
        numerator = np.convolve(values, kernel, mode="same")
        denominator = np.convolve(np.ones(len(values)), kernel, mode="same")
        return numerator / np.maximum(denominator, 1)

    @staticmethod
    def _angle_delta(heading: np.ndarray) -> np.ndarray:
        delta = np.diff(heading, prepend=heading[0])
        return (delta + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        means = np.mean(features, axis=0)
        scales = np.std(features, axis=0)
        scales[scales < 1e-6] = 1.0
        standardized = np.clip((features - means) / scales, -6.0, 6.0)
        return standardized, means, scales

    @staticmethod
    def _pca_2d(features: np.ndarray) -> np.ndarray:
        centered = features - np.mean(features, axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        if len(vt) == 1:
            return np.column_stack([centered @ vt[0], np.zeros(len(centered))])
        return centered @ vt[:2].T

    def _embedding_2d(self, features: np.ndarray) -> np.ndarray:
        method = self.config.embedding_method.lower()
        if method == "pca":
            return self._pca_2d(features)
        if method == "umap":
            try:
                import umap
            except ImportError as exc:
                raise ImportError(
                    "UMAP behavior embedding requires optional dependencies. "
                    "Install with: pip install -e '.[behavior]' / "
                    "UMAP 行为嵌入需要可选依赖。"
                ) from exc
            return np.asarray(
                umap.UMAP(n_components=2, random_state=42).fit_transform(features),
                dtype=float,
            )
        raise ValueError(f"Unknown embedding method / 未知嵌入方法: {method}")

    def _cluster(
        self,
        features: np.ndarray,
        embedding: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        method = self.config.clustering_method.lower()
        if method == "kmeans":
            return self._kmeans(features)
        if method == "hdbscan":
            try:
                import hdbscan
            except ImportError as exc:
                raise ImportError(
                    "HDBSCAN behavior clustering requires optional dependencies. "
                    "Install with: pip install -e '.[behavior]' / "
                    "HDBSCAN 行为聚类需要可选依赖。"
                ) from exc
            cluster_features = (
                embedding
                if embedding is not None and self._clustering_space() == "embedding_2d"
                else features
            )
            min_cluster_size = max(
                2,
                min(self.config.hdbscan_min_cluster_size, max(2, len(cluster_features) - 1)),
            )
            kwargs: dict[str, Any] = {"min_cluster_size": min_cluster_size}
            if self.config.hdbscan_min_samples is not None:
                kwargs["min_samples"] = max(1, int(self.config.hdbscan_min_samples))
            labels = np.asarray(
                hdbscan.HDBSCAN(**kwargs).fit_predict(cluster_features),
                dtype=int,
            )
            cluster_ids = sorted(set(int(label) for label in labels if label >= 0))
            centers = np.array(
                [np.mean(features[labels == label], axis=0) for label in cluster_ids],
                dtype=float,
            ).reshape(-1, features.shape[1])
            return labels, centers
        raise ValueError(f"Unknown clustering method / 未知聚类方法: {method}")

    def _clustering_space(self) -> str:
        if (
            self.config.clustering_method.lower() == "hdbscan"
            and self.config.embedding_method.lower() == "umap"
        ):
            return "embedding_2d"
        return "standardized_features"

    def _kmeans(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_clusters = max(1, min(self.config.n_clusters, len(features)))
        centers = [features[int(np.argmax(np.linalg.norm(features, axis=1)))]]
        while len(centers) < n_clusters:
            distances = np.min(
                np.stack([np.sum((features - center) ** 2, axis=1) for center in centers]),
                axis=0,
            )
            centers.append(features[int(np.argmax(distances))])
        center_matrix = np.array(centers, dtype=float)

        labels = np.zeros(len(features), dtype=int)
        for _ in range(max(1, self.config.max_iterations)):
            distances = np.stack(
                [np.sum((features - center) ** 2, axis=1) for center in center_matrix],
                axis=1,
            )
            new_labels = np.argmin(distances, axis=1)
            new_centers = center_matrix.copy()
            for cluster_id in range(n_clusters):
                members = features[new_labels == cluster_id]
                if len(members):
                    new_centers[cluster_id] = np.mean(members, axis=0)
            if np.array_equal(labels, new_labels) and np.allclose(center_matrix, new_centers):
                break
            labels = new_labels
            center_matrix = new_centers
        return labels, center_matrix

    def _cluster_names(self, samples: list[dict], labels: np.ndarray) -> dict[int, str]:
        names: dict[int, str] = {}
        all_speeds = np.array([sample["speed_px_s"] for sample in samples], dtype=float)
        low_speed = float(np.quantile(all_speeds, 0.30))
        high_speed = float(np.quantile(all_speeds, 0.75))
        all_turns = np.array([sample["turn_rate_rad_s"] for sample in samples], dtype=float)
        high_turn = float(np.quantile(all_turns, 0.75))

        for cluster_id in sorted(set(int(label) for label in labels)):
            if cluster_id == -1:
                names[cluster_id] = "noise_c-1"
                continue
            members = [samples[idx] for idx, label in enumerate(labels) if label == cluster_id]
            speed = float(np.mean([sample["speed_px_s"] for sample in members]))
            turn = float(np.mean([sample["turn_rate_rad_s"] for sample in members]))
            close = float(np.mean([sample["close_neighbor"] for sample in members]))
            if close >= 0.55:
                motif = "close_interaction"
            elif speed <= low_speed:
                motif = "low_activity"
            elif speed >= high_speed:
                motif = "high_activity"
            elif turn >= high_turn:
                motif = "turning_exploration"
            else:
                motif = "locomotion"
            names[cluster_id] = f"{motif}_c{cluster_id}"
        return names

    def _cluster_summary(
        self,
        samples: list[dict],
        labels: np.ndarray,
        cluster_names: dict[int, str],
    ) -> list[dict[str, Any]]:
        result = []
        for cluster_id in sorted(cluster_names):
            members = [samples[idx] for idx, label in enumerate(labels) if label == cluster_id]
            result.append(
                {
                    "cluster": cluster_id,
                    "label": cluster_names[cluster_id],
                    "num_samples": len(members),
                    "ratio": len(members) / max(len(samples), 1),
                    "feature_means": {
                        name: float(np.mean([sample[name] for sample in members]))
                        for name in self.feature_names
                    },
                }
            )
        return result

    @staticmethod
    def _write_outputs(
        report: dict[str, Any],
        timeline_rows: list[dict[str, Any]],
        output_json: str | Path | None,
        output_csv: str | Path | None,
    ) -> None:
        if output_json:
            path = Path(output_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if output_csv:
            path = Path(output_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not timeline_rows:
                path.write_text("", encoding="utf-8")
            else:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(timeline_rows[0]))
                    writer.writeheader()
                    writer.writerows(timeline_rows)

    @staticmethod
    def _draw_embedding(
        embedding: np.ndarray,
        labels: np.ndarray,
        cluster_names: dict[int, str],
        output_path: str | Path,
    ) -> None:
        canvas = np.full((720, 1080, 3), 250, dtype=np.uint8)
        colors = _cluster_colors()
        mapped = _map_to_canvas(embedding, width=780, height=620, offset_x=40, offset_y=50)
        stride = max(1, len(mapped) // 12000)
        for idx in range(0, len(mapped), stride):
            x, y = mapped[idx]
            cv2.circle(canvas, (int(x), int(y)), 2, colors[int(labels[idx]) % len(colors)], -1)
        _draw_legend(canvas, cluster_names, colors, x=840, y=70)
        cv2.putText(canvas, "Unsupervised behavior embedding", (40, 32), 0, 0.75, (20, 20, 20), 2)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), canvas)

    @staticmethod
    def _draw_timeline(
        rows: list[dict[str, Any]],
        cluster_names: dict[int, str],
        output_path: str | Path,
    ) -> None:
        ids = sorted({int(row["track_id"]) for row in rows})
        min_frame = min(int(row["frame"]) for row in rows)
        max_frame = max(int(row["frame"]) for row in rows)
        canvas = np.full((130 + 66 * len(ids), 1400, 3), 250, dtype=np.uint8)
        colors = _cluster_colors()
        id_y = {track_id: 70 + 56 * idx for idx, track_id in enumerate(ids)}
        for track_id in ids:
            y = id_y[track_id]
            cv2.putText(canvas, f"id {track_id}", (16, y + 6), 0, 0.55, (30, 30, 30), 1)
            cv2.line(canvas, (70, y), (1180, y), (190, 190, 190), 1)
        frame_span = max(1, max_frame - min_frame)
        for row in rows:
            x = 70 + int(1110 * (int(row["frame"]) - min_frame) / frame_span)
            y = id_y[int(row["track_id"])]
            color = colors[int(row["cluster"]) % len(colors)]
            cv2.line(canvas, (x, y - 8), (x, y + 8), color, 2)
        _draw_legend(canvas, cluster_names, colors, x=1210, y=36)
        cv2.putText(canvas, "Behavior motif timeline", (16, 28), 0, 0.7, (20, 20, 20), 2)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), canvas)

    def _empty_report(self, mot_path: str | Path) -> dict[str, Any]:
        return {
            "created_at": datetime.now().isoformat(),
            "source": str(mot_path),
            "method": "densetrack_bsoid_inspired_centroid_clustering",
            "annotation_required": False,
            "config": asdict(self.config),
            "summary": {"num_ids": 0, "num_samples": 0, "num_clusters": 0},
            "features": {"names": self.feature_names, "means": [], "scales": []},
            "clusters": [],
        }


def _cluster_colors() -> list[tuple[int, int, int]]:
    return [
        (230, 126, 34),
        (46, 204, 113),
        (52, 152, 219),
        (155, 89, 182),
        (231, 76, 60),
        (26, 188, 156),
        (241, 196, 15),
        (52, 73, 94),
    ]


def _map_to_canvas(
    points: np.ndarray,
    width: int,
    height: int,
    offset_x: int,
    offset_y: int,
) -> np.ndarray:
    lower = np.quantile(points, 0.01, axis=0)
    upper = np.quantile(points, 0.99, axis=0)
    scale = np.maximum(upper - lower, 1e-6)
    normalized = np.clip((points - lower) / scale, 0.0, 1.0)
    mapped = np.empty_like(normalized)
    mapped[:, 0] = offset_x + normalized[:, 0] * width
    mapped[:, 1] = offset_y + (1.0 - normalized[:, 1]) * height
    return mapped


def _draw_legend(
    canvas: np.ndarray,
    names: dict[int, str],
    colors: list[tuple[int, int, int]],
    x: int,
    y: int,
) -> None:
    for offset, cluster_id in enumerate(sorted(names)):
        row_y = y + 28 * offset
        color = colors[cluster_id % len(colors)]
        cv2.rectangle(canvas, (x, row_y - 11), (x + 16, row_y + 5), color, -1)
        cv2.putText(canvas, names[cluster_id], (x + 24, row_y + 3), 0, 0.42, (30, 30, 30), 1)
