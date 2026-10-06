"""
MOT evaluation metrics: HOTA, AssA, MOTA, IDF1, and ID switches.

Uses the `motmetrics` library for standard MOT Challenge evaluation.
HOTA-family metrics follow the TrackEval reference formulation while operating
directly on DenseTrack's MOT files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "compute_mot_metrics",
    "compute_hota_metrics",
    "load_mot_file",
    "compute_iou_matrix",
    "summarize_metrics",
]

logger = logging.getLogger(__name__)


def load_mot_file(path: str | Path) -> pd.DataFrame:
    """
    Load a MOT Challenge format file into a DataFrame.

    MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

    Returns DataFrame with columns: [FrameId, Id, X, Y, Width, Height, Confidence]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MOT file not found: {path}")

    # Skip comment lines (starting with #)
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)

    if not lines:
        return pd.DataFrame(columns=["FrameId", "Id", "X", "Y", "Width", "Height", "Confidence"])

    data = []
    for line in lines:
        parts = line.split(",")
        if len(parts) >= 6:
            data.append({
                "FrameId": int(parts[0]),
                "Id": int(parts[1]),
                "X": float(parts[2]),
                "Y": float(parts[3]),
                "Width": float(parts[4]),
                "Height": float(parts[5]),
                "Confidence": float(parts[6]) if len(parts) > 6 else 1.0,
            })

    return pd.DataFrame(data)


def compute_iou_matrix(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU matrix between two sets of bounding boxes.

    Args:
        bboxes_a: (N, 4) array of [x, y, w, h]
        bboxes_b: (M, 4) array of [x, y, w, h]

    Returns:
        (N, M) IoU matrix
    """
    a = bboxes_a.copy()
    b = bboxes_b.copy()

    # Convert xywh to xyxy
    a[:, 2] = a[:, 0] + a[:, 2]
    a[:, 3] = a[:, 1] + a[:, 3]
    b[:, 2] = b[:, 0] + b[:, 2]
    b[:, 3] = b[:, 1] + b[:, 3]

    # Intersection
    x1 = np.maximum(a[:, 0][:, np.newaxis], b[:, 0])
    y1 = np.maximum(a[:, 1][:, np.newaxis], b[:, 1])
    x2 = np.minimum(a[:, 2][:, np.newaxis], b[:, 2])
    y2 = np.minimum(a[:, 3][:, np.newaxis], b[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    union = area_a[:, np.newaxis] + area_b - inter

    iou = np.where(union > 0, inter / union, 0.0)
    return iou


def compute_mot_metrics(
    gt_path: str | Path,
    pred_path: str | Path,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Compute MOT metrics comparing ground-truth to predictions.

    Args:
        gt_path: Path to ground-truth MOT file.
        pred_path: Path to predicted MOT file.
        iou_threshold: IoU threshold for matching (default: 0.5).

    Returns:
        Dictionary with metric names and values.
    """
    try:
        import motmetrics as mm
    except ImportError:
        raise ImportError(
            "motmetrics is required for evaluation. "
            "Install with: pip install motmetrics"
        )

    gt_df = load_mot_file(gt_path)
    pred_df = load_mot_file(pred_path)

    # Create accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    # Get all frame IDs from both files
    all_frames = sorted(set(gt_df["FrameId"].tolist() + pred_df["FrameId"].tolist()))

    for frame_id in all_frames:
        gt_frame = gt_df[gt_df["FrameId"] == frame_id]
        pred_frame = pred_df[pred_df["FrameId"] == frame_id]

        gt_ids = gt_frame["Id"].values
        pred_ids = pred_frame["Id"].values

        if len(gt_ids) == 0 and len(pred_ids) == 0:
            continue

        if len(gt_ids) == 0 or len(pred_ids) == 0:
            # All are FP or FN
            distances = np.empty((len(gt_ids), len(pred_ids)))
        else:
            gt_bboxes = gt_frame[["X", "Y", "Width", "Height"]].values
            pred_bboxes = pred_frame[["X", "Y", "Width", "Height"]].values
            iou_matrix = compute_iou_matrix(gt_bboxes, pred_bboxes)
            # Convert IoU to distance (1 - IoU), NaN for below threshold
            distances = 1 - iou_matrix
            distances[distances > (1 - iou_threshold)] = np.nan

        acc.update(gt_ids, pred_ids, distances)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "mota",
            "motp",
            "idf1",
            "num_switches",
            "num_fragmentations",
            "mostly_tracked",
            "mostly_lost",
            "num_false_positives",
            "num_misses",
            "precision",
            "recall",
        ],
        name="DenseTrack",
    )

    # Extract to plain dict
    result = {}
    for col in summary.columns:
        val = summary[col].iloc[0]
        result[col] = float(val) if not pd.isna(val) else 0.0
    result.update(compute_hota_metrics(gt_df, pred_df))

    return result


def compute_hota_metrics(
    ground_truth: pd.DataFrame | str | Path,
    predictions: pd.DataFrame | str | Path,
) -> dict[str, float]:
    """Compute HOTA, DetA, AssA, and LocA over the standard 0.05:0.95 thresholds."""
    gt_df = (
        ground_truth
        if isinstance(ground_truth, pd.DataFrame)
        else load_mot_file(ground_truth)
    )
    pred_df = (
        predictions
        if isinstance(predictions, pd.DataFrame)
        else load_mot_file(predictions)
    )
    gt_id_values = sorted(int(value) for value in gt_df["Id"].unique())
    pred_id_values = sorted(int(value) for value in pred_df["Id"].unique())
    gt_id_map = {value: index for index, value in enumerate(gt_id_values)}
    pred_id_map = {value: index for index, value in enumerate(pred_id_values)}
    all_frames = sorted(set(gt_df["FrameId"].tolist() + pred_df["FrameId"].tolist()))

    frame_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for frame_id in all_frames:
        gt_frame = gt_df[gt_df["FrameId"] == frame_id]
        pred_frame = pred_df[pred_df["FrameId"] == frame_id]
        gt_ids = np.asarray(
            [gt_id_map[int(value)] for value in gt_frame["Id"].values],
            dtype=int,
        )
        pred_ids = np.asarray(
            [pred_id_map[int(value)] for value in pred_frame["Id"].values],
            dtype=int,
        )
        if len(gt_ids) and len(pred_ids):
            similarity = compute_iou_matrix(
                gt_frame[["X", "Y", "Width", "Height"]].values,
                pred_frame[["X", "Y", "Width", "Height"]].values,
            )
        else:
            similarity = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
        frame_data.append((gt_ids, pred_ids, similarity))

    num_gt_ids = len(gt_id_values)
    num_pred_ids = len(pred_id_values)
    potential_matches = np.zeros((num_gt_ids, num_pred_ids), dtype=np.float64)
    gt_counts = np.zeros((num_gt_ids, 1), dtype=np.float64)
    pred_counts = np.zeros((1, num_pred_ids), dtype=np.float64)

    for gt_ids, pred_ids, similarity in frame_data:
        if len(gt_ids) and len(pred_ids):
            denominator = (
                similarity.sum(axis=0)[None, :]
                + similarity.sum(axis=1)[:, None]
                - similarity
            )
            normalized = np.divide(
                similarity,
                denominator,
                out=np.zeros_like(similarity),
                where=denominator > np.finfo(float).eps,
            )
            potential_matches[np.ix_(gt_ids, pred_ids)] += normalized
        if len(gt_ids):
            np.add.at(gt_counts[:, 0], gt_ids, 1)
        if len(pred_ids):
            np.add.at(pred_counts[0], pred_ids, 1)

    alignment_denominator = gt_counts + pred_counts - potential_matches
    global_alignment = np.divide(
        potential_matches,
        alignment_denominator,
        out=np.zeros_like(potential_matches),
        where=alignment_denominator > np.finfo(float).eps,
    )

    thresholds = np.arange(0.05, 1.0, 0.05)
    true_positives = np.zeros(len(thresholds), dtype=np.float64)
    false_positives = np.zeros(len(thresholds), dtype=np.float64)
    false_negatives = np.zeros(len(thresholds), dtype=np.float64)
    localization_sum = np.zeros(len(thresholds), dtype=np.float64)
    match_counts = [
        np.zeros((num_gt_ids, num_pred_ids), dtype=np.float64)
        for _ in thresholds
    ]

    from scipy.optimize import linear_sum_assignment

    for gt_ids, pred_ids, similarity in frame_data:
        if not len(gt_ids):
            false_positives += len(pred_ids)
            continue
        if not len(pred_ids):
            false_negatives += len(gt_ids)
            continue
        alignment = global_alignment[np.ix_(gt_ids, pred_ids)] * similarity
        rows, columns = linear_sum_assignment(-alignment)
        matched_similarity = similarity[rows, columns]
        for threshold_index, threshold in enumerate(thresholds):
            accepted = matched_similarity >= threshold - np.finfo(float).eps
            accepted_rows = rows[accepted]
            accepted_columns = columns[accepted]
            matches = int(np.sum(accepted))
            true_positives[threshold_index] += matches
            false_negatives[threshold_index] += len(gt_ids) - matches
            false_positives[threshold_index] += len(pred_ids) - matches
            if matches:
                localization_sum[threshold_index] += float(
                    np.sum(similarity[accepted_rows, accepted_columns])
                )
                np.add.at(
                    match_counts[threshold_index],
                    (gt_ids[accepted_rows], pred_ids[accepted_columns]),
                    1,
                )

    detection_accuracy = np.divide(
        true_positives,
        true_positives + false_negatives + false_positives,
        out=np.zeros_like(true_positives),
        where=(true_positives + false_negatives + false_positives) > 0,
    )
    association_accuracy = np.zeros_like(true_positives)
    for threshold_index, counts in enumerate(match_counts):
        association = np.divide(
            counts,
            gt_counts + pred_counts - counts,
            out=np.zeros_like(counts),
            where=(gt_counts + pred_counts - counts) > 0,
        )
        association_accuracy[threshold_index] = (
            float(np.sum(counts * association))
            / max(float(true_positives[threshold_index]), 1.0)
        )
    localization_accuracy = np.divide(
        localization_sum,
        true_positives,
        out=np.ones_like(localization_sum),
        where=true_positives > 0,
    )
    hota = np.sqrt(detection_accuracy * association_accuracy)
    midpoint = int(np.argmin(np.abs(thresholds - 0.5)))
    return {
        "hota": float(np.mean(hota)),
        "deta": float(np.mean(detection_accuracy)),
        "assa": float(np.mean(association_accuracy)),
        "loca": float(np.mean(localization_accuracy)),
        "hota_50": float(hota[midpoint]),
        "assa_50": float(association_accuracy[midpoint]),
    }


def summarize_metrics(metrics: dict) -> str:
    """Format metrics dict as a human-readable summary string."""
    lines = [
        "═══════════════════════════════════",
        "  MOT Evaluation Results",
        "═══════════════════════════════════",
        f"  HOTA:          {metrics.get('hota', 0):.1%}",
        f"  DetA:          {metrics.get('deta', 0):.1%}",
        f"  AssA:          {metrics.get('assa', 0):.1%}",
        f"  MOTA:          {metrics.get('mota', 0):.1%}",
        f"  MOTP:          {metrics.get('motp', 0):.3f}",
        f"  IDF1:          {metrics.get('idf1', 0):.1%}",
        f"  ID Switches:   {metrics.get('num_switches', 0):.0f}",
        f"  Fragmentations:{metrics.get('num_fragmentations', 0):.0f}",
        f"  Mostly Tracked:{metrics.get('mostly_tracked', 0):.0f}",
        f"  Mostly Lost:   {metrics.get('mostly_lost', 0):.0f}",
        f"  False Pos:     {metrics.get('num_false_positives', 0):.0f}",
        f"  Misses:        {metrics.get('num_misses', 0):.0f}",
        f"  Precision:     {metrics.get('precision', 0):.1%}",
        f"  Recall:        {metrics.get('recall', 0):.1%}",
        "═══════════════════════════════════",
    ]
    return "\n".join(lines)
