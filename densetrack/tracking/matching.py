"""
Matching utilities: IoU computation and linear assignment.

Fully implemented — uses lap for fast Hungarian, scipy as fallback.
"""

from __future__ import annotations

import logging

import numpy as np

__all__ = ["iou_batch", "linear_assignment"]

logger = logging.getLogger(__name__)


def iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IoU between two sets of bounding boxes.

    Args:
        bboxes_a: (N, 4) array of [x1, y1, x2, y2].
        bboxes_b: (M, 4) array of [x1, y1, x2, y2].

    Returns:
        (N, M) IoU matrix.
    """
    a = np.asarray(bboxes_a, dtype=np.float64)
    b = np.asarray(bboxes_b, dtype=np.float64)

    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))

    # Intersection
    x1 = np.maximum(a[:, 0][:, np.newaxis], b[:, 0])
    y1 = np.maximum(a[:, 1][:, np.newaxis], b[:, 1])
    x2 = np.minimum(a[:, 2][:, np.newaxis], b[:, 2])
    y2 = np.minimum(a[:, 3][:, np.newaxis], b[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Union
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, np.newaxis] + area_b - inter

    return np.where(union > 0, inter / union, 0.0)


def linear_assignment(
    cost_matrix: np.ndarray,
    thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform linear assignment (Hungarian algorithm) with threshold filtering.

    Uses `lap.lapjv` for speed if available, otherwise falls back
    to `scipy.optimize.linear_sum_assignment`.

    Args:
        cost_matrix: (N, M) cost matrix. Lower is better.
                     Typically 1 - IoU for tracking.
        thresh: Maximum cost threshold. Assignments with cost > thresh
                are rejected.

    Returns:
        Tuple of:
        - matches: (K, 2) array of [row_idx, col_idx] matched pairs.
        - unmatched_rows: (P,) array of unmatched row indices.
        - unmatched_cols: (Q,) array of unmatched column indices.
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )

    matches, unmatched_rows, unmatched_cols = _solve_assignment(cost_matrix)

    # Filter by threshold
    final_matches = []
    rejected_rows = list(unmatched_rows)
    rejected_cols = list(unmatched_cols)

    for row, col in matches:
        if cost_matrix[row, col] > thresh:
            rejected_rows.append(row)
            rejected_cols.append(col)
        else:
            final_matches.append([row, col])

    if final_matches:
        return (
            np.array(final_matches, dtype=int),
            np.array(sorted(rejected_rows), dtype=int),
            np.array(sorted(rejected_cols), dtype=int),
        )
    else:
        return (
            np.empty((0, 2), dtype=int),
            np.array(sorted(rejected_rows), dtype=int),
            np.array(sorted(rejected_cols), dtype=int),
        )


def _solve_assignment(
    cost_matrix: np.ndarray,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solve the assignment problem using lap or scipy."""
    try:
        import lap

        # lap.lapjv expects a square or rectangular matrix
        # Use large cost for invalid assignments
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=1e6)

        matches = []
        unmatched_rows = []
        unmatched_cols = set(range(cost_matrix.shape[1]))

        for row in range(cost_matrix.shape[0]):
            if x[row] >= 0 and x[row] < cost_matrix.shape[1]:
                matches.append((row, x[row]))
                unmatched_cols.discard(x[row])
            else:
                unmatched_rows.append(row)

        return matches, unmatched_rows, sorted(unmatched_cols)

    except (ImportError, Exception):
        logger.debug("lap not available, using scipy for linear assignment")
        from scipy.optimize import linear_sum_assignment as scipy_lsa

        row_indices, col_indices = scipy_lsa(cost_matrix)
        matches = list(zip(row_indices.tolist(), col_indices.tolist()))

        matched_rows = set(row_indices)
        matched_cols = set(col_indices)
        unmatched_rows = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
        unmatched_cols = [j for j in range(cost_matrix.shape[1]) if j not in matched_cols]

        return matches, unmatched_rows, unmatched_cols
