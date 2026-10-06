"""
Self-supervised ReID training from high-confidence tracklets.

Core innovation: Uses zero human labels. Training data comes from:
- Positive pairs: different frames within the same high-confidence tracklet
- Negative pairs: ONLY same-frame different-individual crops
  (never use different-time tracklets as negatives — they might
   be the same animal after occlusion)

Per-video self-adaptation: the ReID model is fine-tuned on each
video independently, learning to distinguish the specific animals
in that particular experiment.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from densetrack.config import ReIDConfig
from densetrack.models import Tracklet
from densetrack.reid.features import ReIDFeatureExtractor

__all__ = ["SelfSupervisedReID"]

logger = logging.getLogger(__name__)

_torch_available = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _torch_available = True
except ImportError:
    pass


class SelfSupervisedReID:
    """
    Self-supervised ReID learning from high-confidence tracklets.

    The training loop for each video:
    1. Mine training triplets from high-confidence tracklets
    2. Train with triplet margin loss
    3. Compute all pairwise tracklet similarities
    4. (External) Global optimizer uses these similarities

    This can run multiple self-improvement iterations:
    Train ReID → better similarities → better global optimization →
    higher confidence tracklets → better ReID training data → ...
    """

    def __init__(self, config: ReIDConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.extractor: ReIDFeatureExtractor | None = None
        self._trained = False

    def _ensure_extractor(self) -> ReIDFeatureExtractor:
        """Lazy-load the feature extractor only when training can proceed."""
        if self.extractor is None:
            self.extractor = ReIDFeatureExtractor(
                feat_dim=self.config.feat_dim,
                backbone=self.config.backbone,
                device=self.device,
            )
        return self.extractor

    def mine_training_data(
        self,
        tracklets: list[Tracklet],
        frames_dict: dict[int, np.ndarray] | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Mine training triplets from high-confidence tracklets.

        CRITICAL NEGATIVE SAMPLING RULE:
        Only use same-frame different-individual crops as negatives.
        Never use different-time different-tracklet crops — they might
        be the same animal after occlusion, which would poison the model.

        Args:
            tracklets: All tracklets from the video.
            frames_dict: Optional dict mapping frame_idx -> BGR frame.

        Returns:
            List of (anchor_crop, positive_crop, negative_crop) triplets.
        """
        # Filter to high-confidence, clean tracklets
        high_conf = [
            t for t in tracklets
            if t.confidence > self.config.confidence_threshold
            and not t.had_occlusion
            and t.num_id_conflicts == 0
        ]

        if len(high_conf) < 2:
            logger.warning(
                f"Only {len(high_conf)} high-confidence tracklets "
                f"(need ≥2). Skipping ReID training."
            )
            return []

        # Build frame→tracklet index for negative mining
        frame_to_tracklets: dict[int, list[tuple[Tracklet, int]]] = {}
        for t in high_conf:
            for i, det in enumerate(t.detections):
                if getattr(det, "crop_bytes", None) is not None or (frames_dict is not None and det.frame_idx in frames_dict):
                    if det.frame_idx not in frame_to_tracklets:
                        frame_to_tracklets[det.frame_idx] = []
                    frame_to_tracklets[det.frame_idx].append((t, i))

        triplets = []
        for anchor_tracklet in high_conf:
            dets = anchor_tracklet.detections
            if len(dets) < 2:
                continue

            for i in range(0, len(dets) - 1, 2):
                anchor_det = dets[i]
                positive_det = dets[i + 1]

                # Check frames or crop bytes are available
                anchor_has_crop = getattr(anchor_det, "crop_bytes", None) is not None or (frames_dict is not None and anchor_det.frame_idx in frames_dict)
                positive_has_crop = getattr(positive_det, "crop_bytes", None) is not None or (frames_dict is not None and positive_det.frame_idx in frames_dict)

                if not anchor_has_crop or not positive_has_crop:
                    continue

                # Extract anchor and positive crops
                anchor_crop = self._extract_crop(
                    frames_dict[anchor_det.frame_idx] if frames_dict is not None else None,
                    anchor_det
                )
                positive_crop = self._extract_crop(
                    frames_dict[positive_det.frame_idx] if frames_dict is not None else None,
                    positive_det
                )

                if anchor_crop is None or positive_crop is None:
                    continue

                # Find negative: same frame as anchor, different tracklet
                negative_crop = self._find_same_frame_negative(
                    anchor_det.frame_idx,
                    anchor_tracklet.tracklet_id,
                    frame_to_tracklets,
                    frames_dict,
                )

                if negative_crop is not None:
                    triplets.append((anchor_crop, positive_crop, negative_crop))

        logger.info(f"Mined {len(triplets)} training triplets from {len(high_conf)} tracklets")
        return triplets

    def _find_same_frame_negative(
        self,
        frame_idx: int,
        exclude_tracklet_id: int,
        frame_to_tracklets: dict[int, list[tuple[Tracklet, int]]],
        frames_dict: dict[int, np.ndarray] | None = None,
    ) -> np.ndarray | None:
        """Find a crop from a DIFFERENT tracklet in the SAME frame."""
        candidates = frame_to_tracklets.get(frame_idx, [])
        for tracklet, det_idx in candidates:
            if tracklet.tracklet_id != exclude_tracklet_id:
                det = tracklet.detections[det_idx]
                crop = self._extract_crop(
                    frames_dict[frame_idx] if frames_dict is not None else None,
                    det
                )
                if crop is not None:
                    return crop
        return None

    @staticmethod
    def _extract_crop(frame: np.ndarray | None, det: Any) -> np.ndarray | None:
        """Extract and validate a detection crop from a frame or det.crop_bytes."""
        if getattr(det, "crop_bytes", None) is not None:
            try:
                arr = np.frombuffer(det.crop_bytes, dtype=np.uint8)
                crop = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if crop is not None and crop.shape[0] >= 10 and crop.shape[1] >= 10:
                    return crop
            except Exception as exc:
                logger.warning(f"Failed to decode crop_bytes for detection: {exc}")

        if frame is None:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        crop = frame[y1:y2, x1:x2].copy()
        return crop

    def train_on_video(
        self,
        tracklets: list[Tracklet],
        frames_dict: dict[int, np.ndarray] | None = None,
    ) -> bool:
        """
        Train ReID on a single video's tracklets.

        Args:
            tracklets: All tracklets from the video.
            frames_dict: Frame data for crop extraction.

        Returns:
            True if training succeeded, False if insufficient data.
        """
        if not _torch_available:
            logger.warning("PyTorch not available — skipping ReID training")
            return False

        triplets = self.mine_training_data(tracklets, frames_dict)

        if len(triplets) < self.config.min_triplets:
            logger.warning(
                f"Only {len(triplets)} triplets (need ≥{self.config.min_triplets}). "
                f"Skipping ReID training."
            )
            return False

        logger.info(f"Training ReID with {len(triplets)} triplets for {self.config.epochs} epochs")

        extractor = self._ensure_extractor()
        extractor.train_mode()
        optimizer = torch.optim.Adam(
            extractor.parameters,
            lr=self.config.learning_rate,
        )
        triplet_loss = nn.TripletMarginLoss(margin=self.config.margin)

        for epoch in range(self.config.epochs):
            total_loss = 0.0
            np.random.shuffle(triplets)

            for batch_start in range(0, len(triplets), 32):
                batch = triplets[batch_start:batch_start + 32]
                if len(batch) <= 1:
                    continue

                positives = torch.stack([
                    extractor.preprocess(p) for _, p, _ in batch
                ]).to(self.device)
                negatives = torch.stack([
                    extractor.preprocess(n) for _, _, n in batch
                ]).to(self.device)
                anchors = torch.stack([
                    extractor.preprocess(a) for a, _, _ in batch
                ]).to(self.device)

                a_feat = extractor._model(anchors)
                p_feat = extractor._model(positives)
                n_feat = extractor._model(negatives)

                loss = triplet_loss(a_feat, p_feat, n_feat)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / max(len(triplets) // 32, 1)
            logger.debug(f"  Epoch {epoch + 1}/{self.config.epochs}: loss={avg_loss:.4f}")

        extractor.eval_mode()
        self._trained = True
        logger.info("ReID training completed")
        return True

    def compute_all_similarities(
        self,
        tracklets: list[Tracklet],
        frames_dict: dict[int, np.ndarray] | None = None,
    ) -> dict[tuple[int, int], float]:
        """
        Compute pairwise ReID similarity for all tracklet pairs.

        For each tracklet, extracts features from multiple frames
        and computes average similarity.

        Returns:
            Dict mapping (tracklet_id_a, tracklet_id_b) -> similarity score.
        """
        # Extract representative features for each tracklet
        tracklet_features: dict[int, np.ndarray] = {}
        extractor = self._ensure_extractor()

        for t in tracklets:
            crops = []
            # Sample up to 5 crops per tracklet
            sample_indices = np.linspace(
                0, len(t.detections) - 1, min(5, len(t.detections))
            ).astype(int)

            for idx in sample_indices:
                det = t.detections[idx]
                if getattr(det, "crop_bytes", None) is not None or (frames_dict is not None and det.frame_idx in frames_dict):
                    crop = self._extract_crop(
                        frames_dict[det.frame_idx] if frames_dict is not None else None,
                        det
                    )
                    if crop is not None:
                        crops.append(crop)

            if crops:
                features = extractor.extract_batch(crops)
                tracklet_features[t.tracklet_id] = features.mean(axis=0)
            else:
                tracklet_features[t.tracklet_id] = np.zeros(self.config.feat_dim)

        # Compute all pairwise similarities
        similarities: dict[tuple[int, int], float] = {}
        ids = list(tracklet_features.keys())

        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                sim = extractor.compute_similarity(
                    tracklet_features[id_a],
                    tracklet_features[id_b],
                )
                similarities[(id_a, id_b)] = sim

        return similarities

    def run_self_improvement_loop(
        self,
        tracklets: list[Tracklet],
        frames_dict: dict[int, np.ndarray] | None = None,
    ) -> dict[tuple[int, int], float]:
        """
        Run the self-supervised improvement loop.

        Iteratively:
        1. Train ReID from high-confidence tracklets
        2. Compute all similarities (used by global optimizer externally)

        Returns:
            Final similarity dict after all iterations.
        """
        similarities: dict[tuple[int, int], float] = {}

        for iteration in range(self.config.max_iterations):
            logger.info(f"ReID self-improvement iteration {iteration + 1}/{self.config.max_iterations}")

            trained = self.train_on_video(tracklets, frames_dict)
            if not trained:
                logger.info("ReID training skipped — insufficient data")
                break

            similarities = self.compute_all_similarities(tracklets, frames_dict)
            logger.info(f"Computed {len(similarities)} pairwise similarities")

        return similarities
