"""
ReID feature extraction backbone.

Uses a pretrained ResNet18 with a projection head for per-video
self-supervised re-identification learning.

Requires: torch, torchvision (included in base dependencies).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

__all__ = ["ReIDFeatureExtractor"]

logger = logging.getLogger(__name__)

_torch_available = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models
    import torchvision.transforms as T

    _torch_available = True
except ImportError:
    pass


class ReIDFeatureExtractor:
    """
    Feature extraction backbone for ReID.

    Architecture: ResNet18 (pretrained) → Linear(512, feat_dim) → BN → L2 normalize
    Input: BGR crop → resize 128x256 → ImageNet normalize
    Output: L2-normalized feature vector of dimension feat_dim.
    """

    # Standard ImageNet preprocessing
    CROP_SIZE = (128, 256)  # (width, height) — portrait orientation for animals

    def __init__(self, feat_dim: int = 128, backbone: str = "resnet18", device: str = "cpu"):
        self.feat_dim = feat_dim
        self.device = device
        self._model = None

        if not _torch_available:
            logger.warning(
                "PyTorch not available — ReID features will return zero vectors. "
                "Install with: pip install torch torchvision"
            )
            return

        # Build model
        if backbone == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = 512
        elif backbone == "resnet34":
            base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            in_features = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove the classification head
        layers = list(base.children())[:-1]  # Up to avgpool
        self._backbone = nn.Sequential(*layers)

        # Projection head
        self._head = nn.Sequential(
            nn.Linear(in_features, feat_dim),
            nn.BatchNorm1d(feat_dim),
        )

        self._model = nn.Sequential(self._backbone, nn.Flatten(), self._head)
        self._model.to(self.device)
        self._model.eval()

        # Preprocessing transform
        self._transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @property
    def parameters(self):
        """All trainable parameters."""
        if self._model is None:
            return []
        return self._model.parameters()

    def train_mode(self) -> None:
        """Switch to training mode."""
        if self._model:
            self._model.train()

    def eval_mode(self) -> None:
        """Switch to evaluation mode."""
        if self._model:
            self._model.eval()

    def preprocess(self, crop_bgr: np.ndarray) -> "torch.Tensor | None":
        """Preprocess a BGR crop for feature extraction."""
        if not _torch_available:
            return None

        # Resize to standard size
        resized = cv2.resize(crop_bgr, self.CROP_SIZE)
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Apply transforms
        tensor = self._transform(rgb)
        return tensor

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from a single BGR crop.

        Args:
            crop_bgr: BGR image crop of a detected animal.

        Returns:
            L2-normalized feature vector of shape (feat_dim,).
        """
        if not _torch_available or self._model is None:
            return np.zeros(self.feat_dim, dtype=np.float32)

        tensor = self.preprocess(crop_bgr)
        if tensor is None:
            return np.zeros(self.feat_dim, dtype=np.float32)

        with torch.no_grad():
            batch = tensor.unsqueeze(0).to(self.device)
            feat = self._model(batch)
            feat = F.normalize(feat, dim=1)
            return feat.cpu().numpy().flatten()

    def extract_batch(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """
        Extract features from multiple crops.

        Args:
            crops_bgr: List of BGR image crops.

        Returns:
            (N, feat_dim) array of L2-normalized features.
        """
        if not _torch_available or self._model is None or not crops_bgr:
            return np.zeros((len(crops_bgr), self.feat_dim), dtype=np.float32)

        tensors = []
        for crop in crops_bgr:
            t = self.preprocess(crop)
            if t is not None:
                tensors.append(t)

        if not tensors:
            return np.zeros((len(crops_bgr), self.feat_dim), dtype=np.float32)

        with torch.no_grad():
            batch = torch.stack(tensors).to(self.device)
            feats = self._model(batch)
            feats = F.normalize(feats, dim=1)
            return feats.cpu().numpy()

    @staticmethod
    def compute_similarity(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        """Cosine similarity between two feature vectors."""
        a = feat_a.flatten()
        b = feat_b.flatten()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
