"""
Global tracklet-to-trajectory optimizer.

Takes fragmented tracklets and connects them into full trajectories
using multi-factor similarity scoring and greedy/Hungarian/ILP assignment.

The key insight: identity recovery after occlusion cannot rely on
a single cue. We combine motion continuity, appearance (ReID),
body size, orientation, pose, and temporal gap into a weighted score.

Includes a SciPy MILP path-cover solver for offline global association.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from densetrack.config import GlobalOptimizerConfig
from densetrack.models import Tracklet, Trajectory
from densetrack.tracking.discrete_geometry import (
    HierarchicalGeometryEncoder,
    robust_detection_velocity,
)
from densetrack.utils.geometry import angle_difference

__all__ = ["GlobalTrackletOptimizer", "GlobalOptimizerStats"]

logger = logging.getLogger(__name__)


@dataclass
class GlobalOptimizerStats:
    """Diagnostics for WaspMOT-style long-term tracklet reconstruction."""

    method: str
    input_tracklets: int
    output_trajectories: int
    valid_candidate_links: int
    selected_links: int
    closed_set_target: int
    closed_set_achieved: bool
    discrete_geometry_enabled: bool
    mean_selected_link_confidence: float
    min_selected_link_confidence: float


class GlobalTrackletOptimizer:
    """
    Offline global tracklet-to-trajectory optimizer.

    Models all tracklets as nodes in a graph:
    - Edge weight = multi-factor similarity between two tracklets
    - Constraint: same time → same ID only once
    - Prior: n_animals (user-provided)

    Phase 1: Greedy/Hungarian assignment (fast)
    Phase 2: ILP path cover (global optimum over valid association edges)
    """

    def __init__(
        self,
        config: GlobalOptimizerConfig,
        n_animals: int,
        frame_size: tuple[int, int] | None = None,
    ):
        self.config = config
        self.n_animals = n_animals
        self.weights = config.weights
        self.frame_size = frame_size
        self.stats: GlobalOptimizerStats | None = None
        self._long_term = config.method.lower() == "long_term"
        self._geometry_encoder: HierarchicalGeometryEncoder | None = None

    def build_graph(
        self,
        tracklets: list[Tracklet],
        reid_similarities: dict[tuple[int, int], float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the tracklet association graph.

        Args:
            tracklets: List of Tracklet objects (sorted by start_frame).
            reid_similarities: Optional dict mapping (id_a, id_b) -> similarity.

        Returns:
            Tuple of (cost_matrix, confidence_matrix), both (N, N).
        """
        n = len(tracklets)
        cost_matrix = np.full((n, n), np.inf)
        confidence_matrix = np.zeros((n, n))
        if self._long_term:
            self._geometry_encoder = HierarchicalGeometryEncoder(
                frame_size=self.frame_size or self._infer_frame_size(tracklets),
                coarse_bins=self.config.long_term_coarse_bins,
                fine_bins=self.config.long_term_fine_bins,
            )

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                ti = tracklets[i]
                tj = tracklets[j]

                # ti must end before tj starts (non-overlapping)
                if ti.end_frame >= tj.start_frame:
                    continue

                # Temporal gap check
                gap = tj.start_frame - ti.end_frame
                if gap > self.config.max_temporal_gap:
                    continue

                # Compute multi-factor similarity
                scores = self._compute_similarity(ti, tj, gap, reid_similarities)
                total = self._weighted_similarity(scores)

                if total < self.config.min_link_confidence:
                    continue

                cost_matrix[i, j] = 1.0 - total
                confidence_matrix[i, j] = total

        return cost_matrix, confidence_matrix

    def _weighted_similarity(self, scores: dict[str, float]) -> float:
        """Combine legacy cues and reserve explicit mass for discrete geometry."""
        base_weight = sum(max(float(weight), 0.0) for weight in self.weights.values())
        if base_weight <= 0:
            return 0.0
        base_score = sum(
            max(float(self.weights.get(name, 0.0)), 0.0) * float(value)
            for name, value in scores.items()
            if name != "discrete_geometry"
        ) / base_weight
        if not self._long_term:
            return float(np.clip(base_score, 0.0, 1.0))
        geometry_weight = self.config.long_term_geometry_weight
        geometry_score = float(scores.get("discrete_geometry", 0.0))
        return float(
            np.clip(
                (1.0 - geometry_weight) * base_score + geometry_weight * geometry_score,
                0.0,
                1.0,
            )
        )

    def _compute_similarity(
        self,
        ti: Tracklet,
        tj: Tracklet,
        gap: int,
        reid_similarities: dict[tuple[int, int], float] | None = None,
    ) -> dict[str, float]:
        """Compute multi-factor similarity between two tracklets."""
        scores: dict[str, float] = {}

        # 1. Motion continuity. Long-term mode rejects one-frame endpoint jitter
        # and accepts either a ballistic continuation or a spatially local stitch.
        velocity_i = (
            robust_detection_velocity(
                ti.detections,
                endpoint="end",
                window=self.config.long_term_velocity_window,
            )
            if self._long_term
            else ti.last_velocity
        )
        predicted_pos = ti.last_position + velocity_i * gap
        actual_pos = tj.first_position
        ballistic_dist = float(np.linalg.norm(predicted_pos - actual_pos))
        endpoint_dist = float(np.linalg.norm(ti.last_position - actual_pos))
        motion_dist = (
            min(ballistic_dist, endpoint_dist * 1.15)
            if self._long_term
            else ballistic_dist
        )
        motion_scale = 50.0 + (0.65 * np.sqrt(max(gap, 1)) if self._long_term else 0.0)
        scores["motion"] = float(np.exp(-motion_dist / motion_scale))

        # 2. ReID appearance similarity
        if reid_similarities is not None:
            key = (ti.tracklet_id, tj.tracklet_id)
            key_rev = (tj.tracklet_id, ti.tracklet_id)
            if key in reid_similarities:
                scores["appearance"] = reid_similarities[key]
            elif key_rev in reid_similarities:
                scores["appearance"] = reid_similarities[key_rev]
            else:
                scores["appearance"] = 0.5  # Unknown
        else:
            scores["appearance"] = 0.5  # No ReID available

        # 3. Body size consistency
        area_i = ti.avg_area
        area_j = tj.avg_area
        if area_i > 0 and area_j > 0:
            scores["size"] = min(area_i, area_j) / max(area_i, area_j)
        else:
            scores["size"] = 0.5

        # 4. Orientation consistency
        orient_diff = angle_difference(ti.last_orientation, tj.first_orientation)
        scores["orientation"] = float(np.exp(-orient_diff / 0.5))

        # 5. Pose similarity (if available)
        if ti.has_pose and tj.has_pose:
            pose_sim = self._pose_similarity(ti, tj)
            scores["pose"] = pose_sim
        else:
            scores["pose"] = 0.5

        # 6. Temporal gap penalty
        scores["temporal_gap"] = float(np.exp(-gap / 50.0))

        # 7. HieDG-inspired two-stage discrete geometry. Position, scale, and
        # polar velocity remain separate before their weighted token distance.
        if self._long_term and self._geometry_encoder is not None:
            velocity_j = robust_detection_velocity(
                tj.detections,
                endpoint="start",
                window=self.config.long_term_velocity_window,
            )
            expected_bbox = self._shift_bbox_to_center(
                ti.detections[-1].bbox_xyxy,
                predicted_pos,
            )
            expected = self._geometry_encoder.encode(expected_bbox, velocity_i)
            candidate = self._geometry_encoder.encode(
                tj.detections[0].bbox_xyxy,
                velocity_j,
            )
            tolerance = min(gap / max(self.config.max_temporal_gap, 1), 1.0) * 0.08
            scores["discrete_geometry"] = self._geometry_encoder.similarity(
                expected,
                candidate,
                position_tolerance=tolerance,
            )

        return scores

    @staticmethod
    def _pose_similarity(ti: Tracklet, tj: Tracklet) -> float:
        """Compare pose features between tracklet endpoints."""
        last_pose = ti.last_pose
        first_pose = tj.first_pose

        if last_pose is None or first_pose is None:
            return 0.5

        similarities = []

        # Compare body area
        if last_pose.body_area and first_pose.body_area:
            area_ratio = min(last_pose.body_area, first_pose.body_area) / max(
                last_pose.body_area, first_pose.body_area, 1e-6
            )
            similarities.append(area_ratio)

        # Compare aspect ratio
        if last_pose.aspect_ratio and first_pose.aspect_ratio:
            ar_ratio = min(last_pose.aspect_ratio, first_pose.aspect_ratio) / max(
                last_pose.aspect_ratio, first_pose.aspect_ratio, 1e-6
            )
            similarities.append(ar_ratio)

        # Compare body length
        if last_pose.body_length and first_pose.body_length:
            bl_ratio = min(last_pose.body_length, first_pose.body_length) / max(
                last_pose.body_length, first_pose.body_length, 1e-6
            )
            similarities.append(bl_ratio)

        return float(np.mean(similarities)) if similarities else 0.5

    def optimize(
        self,
        tracklets: list[Tracklet],
        reid_similarities: dict[tuple[int, int], float] | None = None,
    ) -> list[Trajectory]:
        """
        Assign global IDs to tracklets.

        Args:
            tracklets: All tracklets from the video.
            reid_similarities: Optional ReID similarity scores.

        Returns:
            List of Trajectory objects (one per animal).
        """
        if not tracklets:
            return []

        # Sort by start frame
        tracklets = sorted(tracklets, key=lambda t: t.start_frame)

        cost_matrix, confidence_matrix = self.build_graph(
            tracklets, reid_similarities
        )

        method = self.config.method.lower()
        if method == "greedy":
            assignments = self._greedy_assignment(tracklets, cost_matrix, confidence_matrix)
        elif method == "hungarian":
            assignments = self._hungarian_assignment(tracklets, cost_matrix, confidence_matrix)
        elif method in {"ilp", "min_cost_flow", "long_term"}:
            assignments = self._ilp_assignment(tracklets, cost_matrix, confidence_matrix)
        else:
            logger.warning(f"Unknown method '{method}', falling back to greedy")
            assignments = self._greedy_assignment(tracklets, cost_matrix, confidence_matrix)

        # Build Trajectory objects
        trajectories = []
        for global_id, assigned_list in assignments.items():
            trajectory = Trajectory(global_id=global_id)

            for idx, (tracklet_idx, tracklet) in enumerate(assigned_list):
                if idx == 0:
                    link_conf = 1.0
                    reason = "first segment"
                else:
                    prev_idx = assigned_list[idx - 1][0]
                    link_conf = float(confidence_matrix[prev_idx, tracklet_idx])
                    reason = self._link_reason(link_conf)

                trajectory.add_segment(
                    tracklet=tracklet,
                    link_confidence=link_conf,
                    confidence_reason=reason,
                )

            trajectories.append(trajectory)

        selected_confidences = [
            float(segment.link_confidence)
            for trajectory in trajectories
            for segment_index, segment in enumerate(trajectory.segments)
            if segment_index > 0
        ]
        selected_links = len(selected_confidences)
        self.stats = GlobalOptimizerStats(
            method=method,
            input_tracklets=len(tracklets),
            output_trajectories=len(trajectories),
            valid_candidate_links=int(np.sum(confidence_matrix > 0)),
            selected_links=selected_links,
            closed_set_target=self.n_animals,
            closed_set_achieved=len(trajectories) == self.n_animals,
            discrete_geometry_enabled=self._long_term,
            mean_selected_link_confidence=(
                float(np.mean(selected_confidences)) if selected_confidences else 1.0
            ),
            min_selected_link_confidence=(
                float(np.min(selected_confidences)) if selected_confidences else 1.0
            ),
        )

        logger.info(
            f"Assigned {len(tracklets)} tracklets → "
            f"{len(trajectories)} trajectories "
            f"(expected {self.n_animals} animals)"
        )

        return trajectories

    @staticmethod
    def _shift_bbox_to_center(
        bbox_xyxy: tuple[float, float, float, float],
        center: np.ndarray,
    ) -> np.ndarray:
        bbox = np.asarray(bbox_xyxy, dtype=np.float64)
        width = max(float(bbox[2] - bbox[0]), 1.0)
        height = max(float(bbox[3] - bbox[1]), 1.0)
        return np.array(
            [
                center[0] - width / 2.0,
                center[1] - height / 2.0,
                center[0] + width / 2.0,
                center[1] + height / 2.0,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _infer_frame_size(tracklets: list[Tracklet]) -> tuple[int, int] | None:
        bboxes = [
            detection.bbox_xyxy
            for tracklet in tracklets
            for detection in tracklet.detections
        ]
        if not bboxes:
            return None
        values = np.asarray(bboxes, dtype=np.float64)
        width = max(int(np.ceil(np.max(values[:, 2]))), 1)
        height = max(int(np.ceil(np.max(values[:, 3]))), 1)
        return width, height

    def _greedy_assignment(
        self,
        tracklets: list[Tracklet],
        cost_matrix: np.ndarray,
        confidence_matrix: np.ndarray,
    ) -> dict[int, list[tuple[int, Tracklet]]]:
        """
        Greedy assignment: sort all possible links by confidence,
        assign greedily while respecting temporal non-overlap.
        """
        n = len(tracklets)

        # Collect all valid links sorted by confidence (descending)
        links = []
        for i in range(n):
            for j in range(n):
                if confidence_matrix[i, j] > 0 and cost_matrix[i, j] < np.inf:
                    links.append((confidence_matrix[i, j], i, j))
        links.sort(reverse=True)

        # Each tracklet starts as its own chain
        # chain_id[tracklet_idx] = which chain it belongs to
        chain_id = list(range(n))
        # For each chain, track its members (in temporal order)
        chains: dict[int, list[int]] = {i: [i] for i in range(n)}
        # Track the tail (last tracklet idx) of each chain
        chain_tail: dict[int, int] = {i: i for i in range(n)}
        # Track the head (first tracklet idx) of each chain
        chain_head: dict[int, int] = {i: i for i in range(n)}

        for conf, i, j in links:
            ci = chain_id[i]
            cj = chain_id[j]

            # Skip if already in the same chain
            if ci == cj:
                continue

            # i must be the tail of its chain
            if chain_tail[ci] != i:
                continue

            # j must be the head of its chain, otherwise merging would pull
            # earlier segments after a later segment and create invalid chains.
            if chain_head[cj] != j:
                continue

            # Merge chain_j into chain_i
            for member in chains[cj]:
                chain_id[member] = ci
                chains[ci].append(member)
            chain_tail[ci] = chain_tail[cj]
            del chains[cj]
            del chain_tail[cj]
            del chain_head[cj]

        # Convert chains to assignments
        assignments: dict[int, list[tuple[int, Tracklet]]] = {}
        global_id = 0
        for chain_members in chains.values():
            # Sort by start frame
            sorted_members = sorted(chain_members, key=lambda idx: tracklets[idx].start_frame)
            assignments[global_id] = [
                (idx, tracklets[idx]) for idx in sorted_members
            ]
            global_id += 1

        if len(assignments) < self.n_animals:
            logger.warning(
                f"Optimizer produced fewer trajectories ({len(assignments)}) than "
                f"expected animals ({self.n_animals}); detection/tracklet quality is likely too low"
            )
        elif len(assignments) > self.n_animals * 3:
            logger.warning(
                f"Optimizer produced many fragmented trajectories ({len(assignments)}) for "
                f"{self.n_animals} expected animals; treat this as a fragmented baseline"
            )

        return assignments

    def _ilp_assignment(
        self,
        tracklets: list[Tracklet],
        cost_matrix: np.ndarray,
        confidence_matrix: np.ndarray,
    ) -> dict[int, list[tuple[int, Tracklet]]]:
        """
        Solve the DAG path-cover association problem as a binary ILP.

        Each valid edge is a binary decision. Every tracklet has at most one
        predecessor and one successor. The known animal count caps the number
        of selected links so the optimizer cannot collapse the result below
        the expected number of identities. This formulation is equivalent to
        min-cost flow on the bipartite DAG while keeping the dependency surface
        limited to SciPy.
        """
        from scipy.optimize import Bounds, LinearConstraint, milp

        n = len(tracklets)
        edges = [
            (i, j)
            for i in range(n)
            for j in range(n)
            if cost_matrix[i, j] < np.inf and confidence_matrix[i, j] > 0
        ]
        if not edges:
            return {
                global_id: [(idx, tracklet)]
                for global_id, (idx, tracklet) in enumerate(zip(range(n), tracklets))
            }

        constraints = np.zeros((2 * n + 1, len(edges)), dtype=float)
        for edge_idx, (source, target) in enumerate(edges):
            constraints[source, edge_idx] = 1.0
            constraints[n + target, edge_idx] = 1.0
            constraints[-1, edge_idx] = 1.0
        max_links = (
            max(0, n - min(self.n_animals, n))
            if self.config.closed_set_stitching
            else max(0, n - 1)
        )
        upper = np.concatenate([np.ones(2 * n), np.array([float(max_links)])])
        result = milp(
            c=np.array([-confidence_matrix[source, target] for source, target in edges]),
            integrality=np.ones(len(edges), dtype=int),
            bounds=Bounds(np.zeros(len(edges)), np.ones(len(edges))),
            constraints=LinearConstraint(constraints, np.zeros(2 * n + 1), upper),
            options={"disp": False},
        )
        if not result.success or result.x is None:
            logger.warning("ILP association failed; falling back to Hungarian assignment")
            return self._hungarian_assignment(tracklets, cost_matrix, confidence_matrix)

        selected = [edges[idx] for idx, value in enumerate(result.x) if value > 0.5]
        assignments = self._assignments_from_links(tracklets, selected)
        logger.info(
            "ILP selected %d/%d valid links and produced %d trajectories",
            len(selected),
            len(edges),
            len(assignments),
        )
        return assignments

    @staticmethod
    def _assignments_from_links(
        tracklets: list[Tracklet],
        links: list[tuple[int, int]],
    ) -> dict[int, list[tuple[int, Tracklet]]]:
        """Convert one-predecessor/one-successor links into trajectory chains."""
        n = len(tracklets)
        next_node = np.full(n, -1, dtype=int)
        prev_node = np.full(n, -1, dtype=int)
        for source, target in links:
            next_node[source] = target
            prev_node[target] = source

        visited = np.zeros(n, dtype=bool)
        chains: list[list[int]] = []
        for idx in range(n):
            if prev_node[idx] != -1 or visited[idx]:
                continue
            chain = []
            current = idx
            while current != -1 and not visited[current]:
                visited[current] = True
                chain.append(current)
                current = int(next_node[current])
            chains.append(chain)
        for idx in range(n):
            if not visited[idx]:
                chains.append([idx])

        assignments = {}
        for global_id, members in enumerate(chains):
            assignments[global_id] = [
                (idx, tracklets[idx])
                for idx in sorted(members, key=lambda value: tracklets[value].start_frame)
            ]
        return assignments

    def _hungarian_assignment(
        self,
        tracklets: list[Tracklet],
        cost_matrix: np.ndarray,
        confidence_matrix: np.ndarray,
    ) -> dict[int, list[tuple[int, Tracklet]]]:
        """
        Hungarian assignment: use scipy for optimal global one-to-one matching
        using a 2N x 2N bipartite formulation.
        """
        from scipy.optimize import linear_sum_assignment

        n = len(tracklets)
        if n == 0:
            return {}

        # 2N x 2N cost matrix
        # Upper left (N x N): link costs (1.0 - confidence) or a huge value M if invalid
        # Upper right (N x N): diagonal contains exit cost (0.5), other values are M
        # Lower left (N x N): diagonal contains entry cost (0.5), other values are M
        # Lower right (N x N): all 0.0 (dummy to dummy)
        M = 1e6
        C = np.full((2 * n, 2 * n), M)

        # Upper left: actual links
        for i in range(n):
            for j in range(n):
                if cost_matrix[i, j] < np.inf:
                    C[i, j] = cost_matrix[i, j]

        # Exit cost (upper right diagonal)
        exit_cost = 0.5
        for i in range(n):
            C[i, i + n] = exit_cost

        # Entry cost (lower left diagonal)
        entry_cost = 0.5
        for j in range(n):
            C[j + n, j] = entry_cost

        # Dummy to dummy (lower right)
        C[n:, n:] = 0.0

        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(C)

        # Reconstruct paths (chains) from matched links
        # A matched pair (row, col) represents a link row -> col if row < n and col < n
        next_node = np.full(n, -1, dtype=int)
        prev_node = np.full(n, -1, dtype=int)

        for r, c in zip(row_ind, col_ind):
            if r < n and c < n:
                # We have a valid link r -> c
                if C[r, c] < M:
                    next_node[r] = c
                    prev_node[c] = r

        # Reconstruct paths starting from heads (prev_node == -1)
        visited = np.zeros(n, dtype=bool)
        chains = []

        for i in range(n):
            if prev_node[i] == -1 and not visited[i]:
                # Start a new chain
                chain = []
                curr = i
                while curr != -1:
                    visited[curr] = True
                    chain.append(curr)
                    curr = next_node[curr]
                chains.append(chain)

        # Handle any cycles (DAG shouldn't have any, but be safe)
        for i in range(n):
            if not visited[i]:
                chain = []
                curr = i
                while not visited[curr]:
                    visited[curr] = True
                    chain.append(curr)
                    curr = next_node[curr]
                    if curr == -1 or curr == i:
                        break
                if chain:
                    chains.append(chain)

        # Convert chains to assignments format
        assignments: dict[int, list[tuple[int, Tracklet]]] = {}
        global_id = 0
        for chain_members in chains:
            sorted_members = sorted(chain_members, key=lambda idx: tracklets[idx].start_frame)
            assignments[global_id] = [
                (idx, tracklets[idx]) for idx in sorted_members
            ]
            global_id += 1

        if len(assignments) < self.n_animals:
            logger.warning(
                f"Hungarian produced fewer trajectories ({len(assignments)}) than "
                f"expected animals ({self.n_animals}); detection/tracklet quality is likely too low"
            )
        elif len(assignments) > self.n_animals:
            logger.info(
                f"Hungarian produced more trajectories ({len(assignments)}) than "
                f"expected animals ({self.n_animals}). Keeping them as distinct trajectories."
            )

        return assignments

    @staticmethod
    def _link_reason(confidence: float) -> str:
        """Generate human-readable reason for a link's confidence level."""
        if confidence >= 0.8:
            return "高置信连接：运动/外观/体型一致"
        elif confidence >= 0.5:
            return "中等置信：部分线索一致，建议抽查"
        elif confidence >= 0.2:
            return "低置信：遮挡恢复，多线索不确定"
        else:
            return "极低置信：可能的ID错误，需人工审查"
