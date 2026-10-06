"""
Grounding DINO open-vocabulary detection adapter.

Requires: pip install densetrack[grounding-dino]
License: Apache 2.0 (code); verify model card for weight distribution terms.
"""

from __future__ import annotations

import logging

import numpy as np

from densetrack.config import DetectionConfig
from densetrack.detection.base import BaseDetector
from densetrack.models import Detection, DetectionSource

__all__ = ["GroundingDINODetector"]

logger = logging.getLogger(__name__)

# Lazy import — only loaded when actually used
_groundingdino_available = False
try:
    import groundingdino  # noqa: F401

    _groundingdino_available = True
except ImportError:
    pass


class GroundingDINODetector(BaseDetector):
    """
    Open-vocabulary animal detector using Grounding DINO.

    Uses a text prompt (e.g., "top-view laboratory mouse") to detect
    animals without requiring any training data. This is the recommended
    primary detector for Phase 0 / Phase 1.

    Install:
        pip install densetrack[grounding-dino]
    """

    def __init__(self, config: DetectionConfig):
        super().__init__(config)
        self._model = None
        self._loaded = False

    @property
    def name(self) -> str:
        return "Grounding DINO"

    def validate(self) -> None:
        """Fail fast if the optional backend is not installed or configured."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """Lazy-load the model on first use."""
        if self._loaded:
            return

        if not _groundingdino_available:
            raise ImportError(
                "Grounding DINO is not installed. Install with:\n"
                "  pip install densetrack[grounding-dino]\n"
                "Or use the 'background_sub' backend instead."
            )

        try:
            from groundingdino.util.inference import load_model, predict

            self._predict_fn = predict

            model_path = self.config.model_path
            if model_path is None:
                # Default model config and weights
                logger.info("Loading Grounding DINO default model...")
                # The actual paths depend on the groundingdino package installation
                self._model = load_model(
                    "groundingdino/config/GroundingDINO_SwinB_cfg.py",
                    "weights/groundingdino_swinb_cogcoor.pth",
                )
            else:
                logger.info(f"Loading Grounding DINO from {model_path}")
                self._model = load_model(model_path, model_path)

            self._loaded = True
            logger.info("Grounding DINO model loaded successfully")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load Grounding DINO model: {e}\n"
                "Make sure the model weights are downloaded. "
                "See: https://github.com/IDEA-Research/GroundingDINO"
            ) from e

    def detect_frame(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        """
        Detect animals in a single frame using Grounding DINO.

        Args:
            frame: BGR image (H, W, 3).
            frame_idx: Frame index.

        Returns:
            List of Detection objects.
        """
        self._ensure_loaded()

        import cv2
        from PIL import Image

        # Convert BGR to RGB PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        h, w = frame.shape[:2]

        # Run inference
        boxes, logits, phrases = self._predict_fn(
            model=self._model,
            image=pil_image,
            caption=self.config.detection_prompt,
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
        )

        # Convert normalized boxes to pixel coordinates
        detections = []
        for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
            # Grounding DINO returns (cx, cy, w, h) normalized
            cx, cy, bw, bh = box.tolist()
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h

            # Clamp to image bounds
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            det = Detection(
                frame_idx=frame_idx,
                bbox_xyxy=(x1, y1, x2, y2),
                confidence=float(logit),
                source=DetectionSource.GROUNDING_DINO,
            )
            detections.append(det)

        return detections
