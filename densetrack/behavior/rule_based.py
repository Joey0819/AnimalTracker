"""Rule-based behavior statistics from MOT tracking results."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from densetrack.utils.metrics import load_mot_file

__all__ = ["BehaviorConfig", "RuleBasedBehaviorAnalyzer"]


@dataclass
class BehaviorConfig:
    fps: float = 15.0
    activity_speed_px_s: float = 35.0
    contact_distance_px: float = 95.0
    approach_distance_px: float = 160.0
    chase_speed_px_s: float = 60.0
    chase_alignment: float = 0.65


class RuleBasedBehaviorAnalyzer:
    """Compute simple activity and interaction summaries from tracks."""

    def __init__(self, config: BehaviorConfig | None = None):
        self.config = config or BehaviorConfig()

    def analyze_mot(
        self,
        mot_path: str | Path,
        output_json: str | Path | None = None,
        output_timeline_csv: str | Path | None = None,
    ) -> dict:
        df = load_mot_file(mot_path)
        if df.empty:
            report = self._empty_report()
            self._write_outputs(report, [], output_json, output_timeline_csv)
            return report

        df = df.copy()
        df["Cx"] = df["X"] + df["Width"] / 2
        df["Cy"] = df["Y"] + df["Height"] / 2
        df = df.sort_values(["Id", "FrameId"])

        per_id = self._per_id_activity(df)
        interaction = self._interactions(df)
        report = {
            "created_at": datetime.now().isoformat(),
            "source": str(mot_path),
            "config": self.config.__dict__,
            "summary": {
                "num_ids": int(df["Id"].nunique()),
                "num_frames": int(df["FrameId"].nunique()),
                "frame_range": [
                    int(df["FrameId"].min()),
                    int(df["FrameId"].max()),
                ],
                "fps": self.config.fps,
            },
            "per_id": per_id,
            "interactions": interaction["summary"],
        }
        self._write_outputs(report, interaction["timeline"], output_json, output_timeline_csv)
        return report

    def _per_id_activity(self, df) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for track_id, group in df.groupby("Id"):
            group = group.sort_values("FrameId")
            coords = group[["Cx", "Cy"]].to_numpy(dtype=float)
            frames = group["FrameId"].to_numpy(dtype=float)
            if len(group) < 2:
                speeds = np.array([], dtype=float)
            else:
                deltas = np.linalg.norm(np.diff(coords, axis=0), axis=1)
                dt = np.maximum(np.diff(frames) / self.config.fps, 1e-6)
                speeds = deltas / dt
            active = speeds >= self.config.activity_speed_px_s
            result[str(int(track_id))] = {
                "frames_tracked": int(len(group)),
                "mean_speed_px_s": float(np.mean(speeds)) if len(speeds) else 0.0,
                "max_speed_px_s": float(np.max(speeds)) if len(speeds) else 0.0,
                "activity_ratio": float(np.mean(active)) if len(active) else 0.0,
                "distance_px": float(np.sum(speeds / self.config.fps)) if len(speeds) else 0.0,
            }
        return result

    def _interactions(self, df) -> dict:
        ids = sorted(int(v) for v in df["Id"].unique())
        id_index = {track_id: idx for idx, track_id in enumerate(ids)}
        contact_frames = np.zeros((len(ids), len(ids)), dtype=int)
        approach_frames = np.zeros((len(ids), len(ids)), dtype=int)
        chase_frames = np.zeros((len(ids), len(ids)), dtype=int)
        timeline: list[dict] = []

        previous_centers: dict[int, np.ndarray] = {}
        for frame_id, group in df.groupby("FrameId"):
            current = {
                int(row.Id): np.array([float(row.Cx), float(row.Cy)], dtype=float)
                for row in group.itertuples(index=False)
            }
            velocities = {
                track_id: (center - previous_centers[track_id]) * self.config.fps
                for track_id, center in current.items()
                if track_id in previous_centers
            }

            contacts: list[str] = []
            approaches: list[str] = []
            chases: list[str] = []
            for i, id_a in enumerate(ids):
                if id_a not in current:
                    continue
                for id_b in ids[i + 1:]:
                    if id_b not in current:
                        continue
                    distance = float(np.linalg.norm(current[id_a] - current[id_b]))
                    if distance <= self.config.contact_distance_px:
                        contact_frames[id_index[id_a], id_index[id_b]] += 1
                        contact_frames[id_index[id_b], id_index[id_a]] += 1
                        contacts.append(f"{id_a}-{id_b}")
                    if distance <= self.config.approach_distance_px:
                        approach_frames[id_index[id_a], id_index[id_b]] += 1
                        approach_frames[id_index[id_b], id_index[id_a]] += 1
                        approaches.append(f"{id_a}-{id_b}")
                        self._mark_chase(
                            id_a,
                            id_b,
                            current,
                            velocities,
                            chase_frames,
                            chases,
                            id_index,
                        )
                        self._mark_chase(
                            id_b,
                            id_a,
                            current,
                            velocities,
                            chase_frames,
                            chases,
                            id_index,
                        )

            timeline.append(
                {
                    "frame": int(frame_id),
                    "contacts": ";".join(contacts),
                    "approaches": ";".join(approaches),
                    "chase_candidates": ";".join(chases),
                }
            )
            previous_centers = current

        total_frames = max(1, int(df["FrameId"].nunique()))
        return {
            "summary": {
                "ids": ids,
                "contact_frames": contact_frames.tolist(),
                "contact_ratio": (contact_frames / total_frames).tolist(),
                "approach_frames": approach_frames.tolist(),
                "chase_candidate_frames": chase_frames.tolist(),
            },
            "timeline": timeline,
        }

    def _mark_chase(
        self,
        source_id: int,
        target_id: int,
        current: dict[int, np.ndarray],
        velocities: dict[int, np.ndarray],
        chase_frames: np.ndarray,
        chases: list[str],
        id_index: dict[int, int],
    ) -> None:
        velocity = velocities.get(source_id)
        if velocity is None:
            return
        speed = float(np.linalg.norm(velocity))
        if speed < self.config.chase_speed_px_s:
            return
        direction = current[target_id] - current[source_id]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-6:
            return
        alignment = float(np.dot(velocity, direction) / (speed * direction_norm))
        if alignment >= self.config.chase_alignment:
            chase_frames[id_index[source_id], id_index[target_id]] += 1
            chases.append(f"{source_id}->{target_id}")

    def _write_outputs(
        self,
        report: dict,
        timeline: list[dict],
        output_json: str | Path | None,
        output_timeline_csv: str | Path | None,
    ) -> None:
        if output_json:
            path = Path(output_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        if output_timeline_csv:
            path = Path(output_timeline_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as f:
                fieldnames = ["frame", "contacts", "approaches", "chase_candidates"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in timeline:
                    writer.writerow(row)

    @staticmethod
    def _empty_report() -> dict:
        return {
            "created_at": datetime.now().isoformat(),
            "summary": {"num_ids": 0, "num_frames": 0, "frame_range": [0, 0]},
            "per_id": {},
            "interactions": {
                "ids": [],
                "contact_frames": [],
                "contact_ratio": [],
                "approach_frames": [],
                "chase_candidate_frames": [],
            },
        }
