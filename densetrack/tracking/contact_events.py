"""Auditable contact-event layer for persistent identity tracking.

This module closes the review-facing gap between aggregate contact counters
(e.g. ``contact_deferred_assignments``) and frame-level risk scores: it turns
every contact/overlap episode into an explicit, auditable event record with

- the participant identities and frame range,
- merged-measurement frames inside the event (at least one participant
  carried by prediction or covered by another participant's box),
- top-K incoming-to-outgoing identity hypotheses at the split boundary,
  scored by a documented motion-continuity + size-consistency function,
- the best-vs-second-best score margin and an uncertainty tier,
- whether the shipped post-event assignment matches the best hypothesis.

Scientific boundary: hypothesis probabilities come from a fixed, documented
scoring function over annotation-free motion/size evidence. They are *not*
calibrated against ground truth and must not be reported as identity
accuracy. Calibrated margins require reviewed GT; the events and priorities
here exist to direct that review.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from densetrack.config import ContactEventConfig
from densetrack.export.run_reexport import MOTRow
from densetrack.models import DetectionSource, Trajectory

__all__ = [
    "FrameBox",
    "ContactEvent",
    "detect_contact_events",
    "frame_boxes_from_trajectories",
    "frame_boxes_from_mot_rows",
    "load_sleap_source_map",
    "write_contact_events_report",
]

_PREDICTION_SOURCES = {
    DetectionSource.ADAPTIVE_PREDICTION.value,
    DetectionSource.FIXED_COUNT_PREDICTION.value,
}


@dataclass
class FrameBox:
    """One animal's box on one frame, with observation provenance."""

    frame: int
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    observed: bool
    confidence: float = 1.0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return max(self.x2 - self.x1, 1.0) * max(self.y2 - self.y1, 1.0)


@dataclass
class Hypothesis:
    """One candidate incoming-ID -> outgoing-box assignment."""

    rank: int
    assignment: dict[int, int]  # incoming track_id -> outgoing track_id
    cost: float
    probability: float
    matches_shipped_assignment: bool


@dataclass
class ContactEvent:
    """Auditable record of one contact/overlap episode."""

    event_id: str
    start_frame: int
    end_frame: int
    duration_frames: int
    participant_ids: list[int]
    max_group_size: int
    merged_frames: int
    merge_fraction: float
    min_centroid_distance_px: float | None
    max_pair_iou: float
    recovery_frame: int | None
    resolved: bool
    uncertainty: str  # confident | ambiguous | uncertain | unresolved
    event_kind: str  # measurement_ambiguous | proximity_only
    best_assignment: dict[int, int] | None
    best_probability: float | None
    second_probability: float | None
    score_margin: float | None
    tracker_matches_best: bool | None
    hypotheses: list[Hypothesis] = field(default_factory=list)
    review_priority: float = 0.0
    notes: list[str] = field(default_factory=list)


def frame_boxes_from_trajectories(
    trajectories: list[Trajectory],
) -> dict[int, list[FrameBox]]:
    """Flatten final trajectories into per-frame boxes (observed wins duplicates)."""
    by_frame: dict[int, dict[int, FrameBox]] = defaultdict(dict)
    for trajectory in trajectories:
        for detection in trajectory.get_all_detections():
            observed = detection.source.value not in _PREDICTION_SOURCES
            candidate = FrameBox(
                frame=int(detection.frame_idx),
                track_id=int(trajectory.global_id),
                x1=float(detection.bbox_xyxy[0]),
                y1=float(detection.bbox_xyxy[1]),
                x2=float(detection.bbox_xyxy[2]),
                y2=float(detection.bbox_xyxy[3]),
                observed=observed,
                confidence=float(detection.confidence),
            )
            existing = by_frame[candidate.frame].get(candidate.track_id)
            if existing is None or (candidate.observed and not existing.observed):
                by_frame[candidate.frame][candidate.track_id] = candidate
    return {frame: list(boxes.values()) for frame, boxes in by_frame.items()}


def frame_boxes_from_mot_rows(
    rows: Iterable[MOTRow],
    source_map: dict[tuple[int, int], str] | None = None,
) -> dict[int, list[FrameBox]]:
    """Build per-frame boxes from MOT rows plus an optional SLEAP source map."""
    source_map = source_map or {}
    by_frame: dict[int, list[FrameBox]] = defaultdict(list)
    for row in rows:
        source = source_map.get((row.frame, row.track_id), "")
        observed = source not in _PREDICTION_SOURCES
        by_frame[row.frame].append(
            FrameBox(
                frame=row.frame,
                track_id=row.track_id,
                x1=row.x,
                y1=row.y,
                x2=row.x + row.width,
                y2=row.y + row.height,
                observed=observed,
                confidence=row.confidence,
            )
        )
    return dict(by_frame)


def load_sleap_source_map(path: str | Path) -> dict[tuple[int, int], str]:
    """Load (frame, track_id) -> densetrack_source from a labels.sleap.json export."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    sources: dict[tuple[int, int], str] = {}
    for frame in payload.get("frames", []):
        frame_idx = int(frame.get("frame_idx", 0))
        for instance in frame.get("instances", []):
            sources[(frame_idx, int(instance.get("track_id", 0)))] = str(
                instance.get("densetrack_source", "")
            )
    return sources


def _pair_iou(a: FrameBox, b: FrameBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _contact_groups(
    boxes: list[FrameBox], config: ContactEventConfig
) -> list[list[FrameBox]]:
    """Union-find grouping of boxes in contact on one frame."""
    parent = list(range(len(boxes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            distance = math.hypot(a.cx - b.cx, a.cy - b.cy)
            if (
                _pair_iou(a, b) >= config.iou_threshold
                or distance < config.contact_distance_px
            ):
                union(i, j)

    groups: dict[int, list[FrameBox]] = defaultdict(list)
    for index, box in enumerate(boxes):
        groups[find(index)].append(box)
    return [group for group in groups.values() if len(group) >= 2]


def _robust_velocity(
    history: list[FrameBox], window: int
) -> tuple[np.ndarray, FrameBox] | None:
    """Median per-frame displacement over the last `window` observed steps."""
    observed = [box for box in history if box.observed]
    if len(observed) < 2:
        return None
    tail = observed[-(window + 1) :]
    displacements = []
    for previous, current in zip(tail, tail[1:]):
        gap = max(current.frame - previous.frame, 1)
        displacements.append(
            np.array(
                [
                    (current.cx - previous.cx) / gap,
                    (current.cy - previous.cy) / gap,
                ]
            )
        )
    if not displacements:
        return None
    velocity = np.median(np.stack(displacements), axis=0)
    return velocity, observed[-1]


def _score_hypotheses(
    participants: list[int],
    pre_event_histories: dict[int, list[FrameBox]],
    outgoing: dict[int, FrameBox],
    recovery_frame: int,
    config: ContactEventConfig,
) -> list[Hypothesis]:
    """Enumerate and score incoming->outgoing assignments.

    Cost per assignment sums, over participants, the squared
    motion-noise-normalized extrapolation error plus a weighted squared
    log-area ratio. Incoming motion references use only frames before the
    event start. Probabilities are softmax(-cost / temperature); they are an
    annotation-free relative score, not a GT-calibrated confidence.
    """
    predictions: dict[int, tuple[np.ndarray, float, float]] = {}
    for track_id in participants:
        history = pre_event_histories.get(track_id, [])
        estimate = _robust_velocity(history, config.velocity_window)
        if estimate is None:
            # No usable incoming velocity: fall back to the last pre-event box.
            if not history:
                return []
            velocity = np.zeros(2)
            reference = history[-1]
        else:
            velocity, reference = estimate
        gap = max(recovery_frame - reference.frame, 1)
        sigma = config.motion_noise_px * math.sqrt(gap)
        position = np.array([reference.cx, reference.cy]) + velocity * gap
        predictions[track_id] = (position, sigma, reference.area)

    scored: list[tuple[float, dict[int, int]]] = []
    for permutation in permutations(participants):
        assignment = {
            track_id: outgoing_id
            for track_id, outgoing_id in zip(participants, permutation)
        }
        cost = 0.0
        for track_id, outgoing_id in assignment.items():
            position, sigma, reference_area = predictions[track_id]
            target = outgoing[outgoing_id]
            distance = math.hypot(
                position[0] - target.cx, position[1] - target.cy
            )
            cost += (distance / max(sigma, 1e-6)) ** 2
            cost += config.size_weight * (
                math.log(max(target.area, 1.0) / max(reference_area, 1.0)) ** 2
            )
        scored.append((cost, assignment))

    scored.sort(key=lambda item: item[0])
    top = scored[: max(config.top_k, 1)]
    temperature = max(config.hypothesis_temperature, 1e-6)
    weights = np.array([math.exp(-cost / temperature) for cost, _ in top])
    total = float(weights.sum())
    identity = {track_id: track_id for track_id in participants}
    return [
        Hypothesis(
            rank=rank + 1,
            assignment=assignment,
            cost=float(cost),
            probability=float(weight / total) if total > 0 else 0.0,
            matches_shipped_assignment=(assignment == identity),
        )
        for rank, ((cost, assignment), weight) in enumerate(zip(top, weights))
    ]


def detect_contact_events(
    frame_boxes: dict[int, list[FrameBox]],
    config: ContactEventConfig | None = None,
    n_animals: int | None = None,
) -> dict[str, Any]:
    """Detect contact events and score split-boundary identity hypotheses."""
    config = config or ContactEventConfig()
    frames = sorted(frame_boxes)
    events: list[ContactEvent] = []
    open_events: list[dict[str, Any]] = []

    for frame in frames:
        groups = _contact_groups(frame_boxes[frame], config)
        matched: set[int] = set()
        for group in groups:
            group_ids = {box.track_id for box in group}
            target_index = None
            for index, event in enumerate(open_events):
                if index in matched:
                    continue
                if set(event["participant_ids"]) & group_ids:
                    target_index = index
                    break
            if target_index is None:
                open_events.append(
                    {
                        "start_frame": frame,
                        "end_frame": frame,
                        "participant_ids": set(group_ids),
                        "max_group_size": len(group),
                        "merged_frames": 0,
                        "min_distance": None,
                        "max_iou": 0.0,
                    }
                )
                target_index = len(open_events) - 1
            matched.add(target_index)
            event = open_events[target_index]
            event["end_frame"] = frame
            event["participant_ids"] |= group_ids
            event["max_group_size"] = max(event["max_group_size"], len(group))
            observed = [box for box in group if box.observed]
            merged = len(observed) < len(group)
            if not merged:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        a, b = group[i], group[j]
                        if (
                            a.x1 <= b.cx <= a.x2
                            and a.y1 <= b.cy <= a.y2
                            and _pair_iou(a, b) >= config.merge_iou
                        ) or (
                            b.x1 <= a.cx <= b.x2
                            and b.y1 <= a.cy <= b.y2
                            and _pair_iou(a, b) >= config.merge_iou
                        ):
                            merged = True
            event["merged_frames"] += int(merged)
            distances = [
                math.hypot(a.cx - b.cx, a.cy - b.cy)
                for i, a in enumerate(group)
                for b in group[i + 1 :]
            ]
            ious = [
                _pair_iou(a, b)
                for i, a in enumerate(group)
                for b in group[i + 1 :]
            ]
            if distances:
                frame_min = min(distances)
                event["min_distance"] = (
                    frame_min
                    if event["min_distance"] is None
                    else min(event["min_distance"], frame_min)
                )
            if ious:
                event["max_iou"] = max(event["max_iou"], max(ious))

        still_open = []
        for index, event in enumerate(open_events):
            if index in matched or event["end_frame"] >= frame - 1:
                still_open.append(event)
            else:
                events.append(event)
        open_events = still_open
    events.extend(open_events)

    frame_index = {frame: boxes for frame, boxes in frame_boxes.items()}
    histories: dict[int, list[FrameBox]] = defaultdict(list)
    for frame in frames:
        for box in frame_boxes[frame]:
            histories[box.track_id].append(box)

    finalized: list[ContactEvent] = []
    for index, event in enumerate(sorted(events, key=lambda item: item["start_frame"])):
        finalized.append(
            _finalize_event(
                index + 1,
                event,
                frame_index,
                histories,
                frames[-1] if frames else 0,
                config,
            )
        )

    episodes = _build_inversion_episodes(finalized, config.episode_gap_frames)
    summary = _summarize_events(finalized, n_animals, episodes)
    frame_priorities: dict[int, float] = {}
    for event in finalized:
        last = event.recovery_frame or event.end_frame
        for frame in range(event.start_frame, last + 1):
            frame_priorities[frame] = max(
                frame_priorities.get(frame, 0.0), event.review_priority
            )

    return {
        "events": finalized,
        "inversion_episodes": episodes,
        "summary": summary,
        "frame_review_priorities": {
            str(frame): round(priority, 4)
            for frame, priority in sorted(frame_priorities.items())
        },
    }


def _finalize_event(
    ordinal: int,
    raw: dict[str, Any],
    frame_index: dict[int, list[FrameBox]],
    histories: dict[int, list[FrameBox]],
    last_frame: int,
    config: ContactEventConfig,
) -> ContactEvent:
    participants = sorted(raw["participant_ids"])
    start_frame = raw["start_frame"]
    end_frame = raw["end_frame"]
    duration = end_frame - start_frame + 1
    notes: list[str] = []

    recovery_frame = None
    outgoing: dict[int, FrameBox] = {}
    for frame in range(end_frame + 1, min(end_frame + config.recovery_window, last_frame) + 1):
        boxes = {
            box.track_id: box
            for box in frame_index.get(frame, [])
            if box.track_id in participants and box.observed
        }
        if len(boxes) >= 2:
            recovery_frame = frame
            outgoing = boxes
            break

    hypotheses: list[Hypothesis] = []
    best_assignment = None
    best_probability = None
    second_probability = None
    margin = None
    tracker_matches = None
    uncertainty = "unresolved"

    if recovery_frame is None:
        notes.append("no_recovery_within_window")
    elif len(outgoing) > config.max_hypothesis_participants:
        notes.append("too_many_participants_for_enumeration")
    else:
        pre_event_histories = {
            track_id: [
                box for box in histories.get(track_id, []) if box.frame < start_frame
            ]
            for track_id in outgoing
        }
        hypotheses = _score_hypotheses(
            sorted(outgoing),
            pre_event_histories,
            outgoing,
            recovery_frame,
            config,
        )
        if hypotheses:
            best = hypotheses[0]
            best_assignment = best.assignment
            best_probability = best.probability
            second_probability = (
                hypotheses[1].probability if len(hypotheses) > 1 else 0.0
            )
            margin = best_probability - second_probability
            tracker_matches = best.matches_shipped_assignment
            if margin >= config.confident_margin:
                uncertainty = "confident"
            elif margin >= config.ambiguous_margin:
                uncertainty = "ambiguous"
            else:
                uncertainty = "uncertain"
        else:
            notes.append("no_incoming_motion_reference")

    merge_fraction = raw["merged_frames"] / max(duration, 1)
    resolved = uncertainty != "unresolved"
    uncertainty_component = 1.0 - (margin if margin is not None else 0.0)
    review_priority = min(
        1.0,
        0.5 * uncertainty_component
        + 0.3 * merge_fraction
        + 0.2 * min(duration / 30.0, 1.0),
    )
    if tracker_matches is False:
        notes.append("tracker_assignment_differs_from_best_hypothesis")
        review_priority = min(1.0, review_priority + 0.15)

    return ContactEvent(
        event_id=f"CE-{ordinal:04d}",
        start_frame=start_frame,
        end_frame=end_frame,
        duration_frames=duration,
        participant_ids=participants,
        max_group_size=int(raw["max_group_size"]),
        merged_frames=int(raw["merged_frames"]),
        merge_fraction=float(merge_fraction),
        min_centroid_distance_px=(
            float(raw["min_distance"]) if raw["min_distance"] is not None else None
        ),
        max_pair_iou=float(raw["max_iou"]),
        recovery_frame=recovery_frame,
        resolved=resolved,
        uncertainty=uncertainty,
        event_kind=(
            "measurement_ambiguous" if raw["merged_frames"] > 0 else "proximity_only"
        ),
        best_assignment=best_assignment,
        best_probability=best_probability,
        second_probability=second_probability,
        score_margin=margin,
        tracker_matches_best=tracker_matches,
        hypotheses=hypotheses,
        review_priority=float(round(review_priority, 4)),
        notes=notes,
    )


def _build_inversion_episodes(
    events: list[ContactEvent], gap_frames: int
) -> list[dict[str, Any]]:
    """Chain tracker-disagreement events into label-inversion episodes.

    Once two labels are inverted (by a true switch or a global smoothing
    permutation), every later contact between the same animals scores as
    swap-best until the labels re-align, so raw disagreement counts
    overstate the number of independent identity problems. Episodes chain
    disagreement events whose participant sets overlap and whose temporal
    gap is at most ``gap_frames``; they are review units, not switch counts.
    """
    disagreements = sorted(
        (
            event
            for event in events
            if event.tracker_matches_best is False and event.best_assignment
        ),
        key=lambda event: event.start_frame,
    )
    episodes: list[dict[str, Any]] = []
    for event in disagreements:
        target = None
        for episode in episodes:
            if event.start_frame - episode["end_frame"] > gap_frames:
                continue
            if set(episode["participant_ids"]) & set(event.participant_ids):
                target = episode
                break
        if target is None:
            episodes.append(
                {
                    "start_frame": event.start_frame,
                    "end_frame": event.end_frame,
                    "participant_ids": list(event.participant_ids),
                    "event_ids": [event.event_id],
                    "max_review_priority": event.review_priority,
                    "event_kinds": [event.event_kind],
                }
            )
        else:
            target["end_frame"] = max(target["end_frame"], event.end_frame)
            target["participant_ids"] = sorted(
                set(target["participant_ids"]) | set(event.participant_ids)
            )
            target["event_ids"].append(event.event_id)
            target["max_review_priority"] = max(
                target["max_review_priority"], event.review_priority
            )
            if event.event_kind not in target["event_kinds"]:
                target["event_kinds"].append(event.event_kind)

    return [
        {
            "episode_id": f"IE-{index + 1:03d}",
            "start_frame": episode["start_frame"],
            "end_frame": episode["end_frame"],
            "span_frames": episode["end_frame"] - episode["start_frame"] + 1,
            "participant_ids": episode["participant_ids"],
            "n_events": len(episode["event_ids"]),
            "event_ids": episode["event_ids"],
            "max_review_priority": episode["max_review_priority"],
            "event_kinds": sorted(episode["event_kinds"]),
        }
        for index, episode in enumerate(episodes)
    ]


def _summarize_events(
    events: list[ContactEvent],
    n_animals: int | None,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    margins = [
        event.score_margin for event in events if event.score_margin is not None
    ]
    uncertainty_counts: dict[str, int] = defaultdict(int)
    kind_counts: dict[str, int] = defaultdict(int)
    for event in events:
        uncertainty_counts[event.uncertainty] += 1
        kind_counts[event.event_kind] += 1
    return {
        "total_events": len(events),
        "resolved_events": sum(event.resolved for event in events),
        "unresolved_events": sum(not event.resolved for event in events),
        "uncertainty_counts": dict(sorted(uncertainty_counts.items())),
        "event_kind_counts": dict(sorted(kind_counts.items())),
        "tracker_disagreement_events": sum(
            event.tracker_matches_best is False for event in events
        ),
        "inversion_episode_count": len(episodes),
        "cascade_caveat": (
            "Raw disagreement events cascade: one persistent label inversion makes "
            "every later contact between the same animals score as swap-best. Use "
            "inversion_episode_count as the review unit, not the raw count. "
            "原始不一致事件会级联：一次持续的标签反转会让同一对动物的后续每次接触 "
            "都显示为交换最优。请以反转段数量作为审核单位，而不是原始事件数。"
        ),
        "total_merged_frames": sum(event.merged_frames for event in events),
        "mean_duration_frames": (
            float(np.mean([event.duration_frames for event in events])) if events else 0.0
        ),
        "max_duration_frames": max((event.duration_frames for event in events), default=0),
        "mean_score_margin": float(np.mean(margins)) if margins else None,
        "min_score_margin": min(margins) if margins else None,
        "expected_animals": n_animals,
    }


def write_contact_events_report(
    output_dir: str | Path,
    *,
    config: ContactEventConfig | None = None,
    trajectories: list[Trajectory] | None = None,
    mot_rows: list[MOTRow] | None = None,
    source_map: dict[tuple[int, int], str] | None = None,
    n_animals: int | None = None,
    source_description: str = "pipeline",
) -> tuple[dict[str, str], dict[str, Any]]:
    """Detect contact events and write contact_events.json / contact_events.csv.

    Returns the artifact paths plus the in-memory summary for logging.
    """
    config = config or ContactEventConfig()
    if trajectories is not None:
        frame_boxes = frame_boxes_from_trajectories(trajectories)
    elif mot_rows is not None:
        frame_boxes = frame_boxes_from_mot_rows(mot_rows, source_map)
    else:
        raise ValueError("trajectories or mot_rows must be provided")

    result = detect_contact_events(frame_boxes, config, n_animals=n_animals)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "contact_events.json"
    csv_path = output / "contact_events.csv"

    payload = {
        "created_at": datetime.now().isoformat(),
        "format": "densetrack_contact_events",
        "format_version": "1.1",
        "source": source_description,
        "method": {
            "description": (
                "Auditable contact-event layer: per-event merge/split records with "
                "top-K incoming-to-outgoing identity hypotheses scored by motion "
                "continuity and size consistency. "
                "可审计接触事件层：逐事件合并/拆分记录，"
                "并用运动连续性与体型一致性对入→出身份假设打分。"
            ),
            "boundary": (
                "Hypothesis probabilities are annotation-free relative scores from a "
                "fixed documented function; they are not calibrated against ground "
                "truth and are not an identity-accuracy metric. Tracker-disagreement "
                "events cascade across a persistent label inversion; inversion "
                "episodes are the review unit. Reviewed GT is required for "
                "calibrated margins and switch counts. / 假设概率是无标注相对分数，"
                "未经 GT 校准，不是身份准确率；不一致事件会级联，反转段才是审核单位。"
            ),
            "config": config.model_dump(),
        },
        "summary": result["summary"],
        "events": [asdict(event) for event in result["events"]],
        "inversion_episodes": result["inversion_episodes"],
        "frame_review_priorities": result["frame_review_priorities"],
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fieldnames = [
        "event_id",
        "start_frame",
        "end_frame",
        "duration_frames",
        "participant_ids",
        "event_kind",
        "max_group_size",
        "merged_frames",
        "merge_fraction",
        "min_centroid_distance_px",
        "max_pair_iou",
        "recovery_frame",
        "resolved",
        "uncertainty",
        "best_assignment",
        "best_probability",
        "second_probability",
        "score_margin",
        "tracker_matches_best",
        "review_priority",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for event in result["events"]:
            row = asdict(event)
            row.pop("hypotheses", None)
            row["participant_ids"] = "|".join(map(str, event.participant_ids))
            row["best_assignment"] = (
                json.dumps(
                    {str(key): value for key, value in event.best_assignment.items()}
                )
                if event.best_assignment
                else ""
            )
            row["notes"] = "|".join(event.notes)
            writer.writerow(row)

    return {
        "contact_events_json": str(json_path),
        "contact_events_csv": str(csv_path),
    }, result["summary"]
