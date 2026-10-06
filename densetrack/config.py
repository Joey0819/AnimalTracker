"""
Pydantic configuration models for DenseTrack-AI.

Loaded from YAML config files (e.g., configs/mice_topview.yaml).
All parameters have sensible defaults for the mouse top-view scenario.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------


class Species(str, Enum):
    MOUSE = "mouse"
    INSECT = "insect"
    ZEBRAFISH = "zebrafish"
    DOLPHIN = "dolphin"


# ---------------------------------------------------------------------------
# Detection Config
# ---------------------------------------------------------------------------


class DetectionConfig(BaseModel):
    """Configuration for the detection module."""

    backend: str = Field(
        default="background_sub",
        description="Detection backend: grounding_dino | background_sub | dark_object",
    )
    detection_prompt: str = Field(
        default="top-view laboratory mouse . white mouse . black mouse",
        description="Text prompt for open-vocabulary detection",
    )
    box_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    text_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    min_mask_area: float = Field(
        default=500,
        description="Minimum valid mask area in pixels²",
    )
    max_mask_area: float = Field(
        default=15000,
        description="Maximum valid mask area in pixels²",
    )
    redetect_interval: int = Field(
        default=25,
        description="Re-detect every N frames to correct SAM2 propagation drift",
    )
    model_path: str | None = Field(
        default=None,
        description="Path to detection model weights when required by the backend",
    )


class SAM2Config(BaseModel):
    """Configuration for SAM 2 video mask propagation."""

    enabled: bool = True
    model_variant: str = Field(
        default="sam2_hiera_large",
        description="SAM 2 model variant: sam2_hiera_tiny | sam2_hiera_small | sam2_hiera_base_plus | sam2_hiera_large",
    )
    points_per_side: int = Field(default=32, ge=8, le=128)
    model_path: str | None = None


# ---------------------------------------------------------------------------
# Pose Config
# ---------------------------------------------------------------------------


class PoseConfig(BaseModel):
    """Configuration for pose estimation."""

    backend: str = Field(
        default="geometry",
        description="Pose backend: dlc_superanimal | geometry",
    )
    superanimal_name: str = Field(
        default="superanimal_topviewmouse",
        description="DLC SuperAnimal model name",
    )
    model_name: str = Field(
        default="hrnet_w32",
        description="DLC backbone model",
    )
    video_adapt: bool = Field(
        default=True,
        description="Enable DLC video adaptation to reduce jitter",
    )
    fallback_to_geometry: bool = Field(
        default=True,
        description="Fall back to geometric features if keypoint model fails",
    )


# ---------------------------------------------------------------------------
# Tracking Config
# ---------------------------------------------------------------------------


class TrackletConfig(BaseModel):
    """Configuration for tracklet (short trajectory fragment) generation."""

    backend: str = Field(
        default="bytetrack",
        description="Tracklet generator: norfair | bytetrack | fixed_count | mac_adaptive | ocsort | udmt",
    )
    # ByteTrack thresholds (conservative: prefer breaking over wrong association)
    tau_high: float = Field(default=0.5, ge=0.0, le=1.0)
    tau_low: float = Field(default=0.1, ge=0.0, le=1.0)
    match_thresh: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="IoU threshold for matching (strict to avoid wrong links)",
    )
    track_buffer: int = Field(
        default=10,
        description="Max frames to keep a lost track before breaking (intentionally short)",
    )
    fixed_max_assignment_distance: float = Field(
        default=150.0,
        description="Max centroid distance for fixed-count tracker assignment",
    )
    fixed_prediction_confidence: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Base confidence for fixed-count predicted boxes during occlusion",
    )
    adaptive_appearance_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Appearance contribution for the Mac-native adaptive tracker",
    )
    adaptive_template_momentum: float = Field(
        default=0.12,
        ge=0.0,
        le=1.0,
        description="EMA momentum for high-confidence appearance template updates",
    )
    adaptive_template_update_cost: float = Field(
        default=0.72,
        ge=0.0,
        description="Maximum assignment cost allowed to update an appearance template",
    )
    adaptive_template_min_margin: float = Field(
        default=0.08,
        ge=0.0,
        description="Minimum assignment margin required before updating an appearance template",
    )
    adaptive_template_min_confidence: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Minimum detector confidence required before updating an appearance template",
    )
    adaptive_template_bank_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Maximum recent foreground appearance templates retained per identity",
    )
    adaptive_init_window: int = Field(
        default=120,
        ge=1,
        description="Candidate frames inspected near each video end for automatic initialization",
    )
    adaptive_consistency_distance: float = Field(
        default=90.0,
        gt=0.0,
        description="Pixel scale used when fusing forward and backward tracking passes",
    )
    adaptive_identity_smoothing: bool = Field(
        default=True,
        description="Run global identity-stable smoothing after bidirectional fusion",
    )
    adaptive_smoothing_motion_weight: float = Field(
        default=0.70,
        ge=0.0,
        description="Transition motion weight for identity-stable smoothing",
    )
    adaptive_smoothing_evidence_weight: float = Field(
        default=0.55,
        ge=0.0,
        description="Directional/fused evidence weight for identity-stable smoothing",
    )
    adaptive_smoothing_appearance_weight: float = Field(
        default=0.20,
        ge=0.0,
        description="Anchor appearance weight for identity-stable smoothing",
    )
    adaptive_discrete_geometry: bool = Field(
        default=True,
        description=(
            "Use HieDG-inspired two-stage discrete position, scale, and polar-velocity "
            "tokens during association"
        ),
    )
    adaptive_geometry_weight: float = Field(
        default=0.30,
        ge=0.0,
        description="Discrete-geometry contribution during directional association",
    )
    adaptive_geometry_coarse_bins: int = Field(
        default=8,
        ge=2,
        le=32,
        description="Coarse bins per geometric dimension (8x8 yields 64 codewords)",
    )
    adaptive_geometry_fine_bins: int = Field(
        default=8,
        ge=2,
        le=32,
        description="Residual bins per geometric dimension",
    )
    adaptive_geometry_knn: int = Field(
        default=5,
        ge=1,
        le=32,
        description="Historical neighbors used for inference-time velocity stabilization",
    )
    adaptive_velocity_window: int = Field(
        default=7,
        ge=2,
        le=60,
        description="Observed displacements used for robust median velocity estimation",
    )
    adaptive_reacquisition_min_margin: float = Field(
        default=0.035,
        ge=0.0,
        description="Minimum one-to-one assignment margin after an occlusion",
    )
    adaptive_reacquisition_distance_growth: float = Field(
        default=5.0,
        ge=0.0,
        description=(
            "Additional reacquisition radius per square root of missing frames; "
            "keeps uncertainty growth sublinear"
        ),
    )
    adaptive_max_reacquisition_distance: float = Field(
        default=220.0,
        gt=0.0,
        description=(
            "Hard centroid-distance ceiling for reacquisition, measured from the "
            "motion prediction"
        ),
    )
    adaptive_contact_distance: float = Field(
        default=70.0,
        gt=0.0,
        description=(
            "Center distance below which low-margin assignments are treated as "
            "contact ambiguity and deferred"
        ),
    )
    adaptive_contact_min_margin: float = Field(
        default=0.06,
        ge=0.0,
        description=(
            "Minimum assignment margin required to consume a detection during "
            "close contact"
        ),
    )
    adaptive_merged_area_ratio: float = Field(
        default=1.70,
        gt=1.0,
        description=(
            "Detection-to-single-animal area ratio above which a blob is kept "
            "anonymous instead of updating one identity"
        ),
    )
    adaptive_smoothing_geometry_weight: float = Field(
        default=0.50,
        ge=0.0,
        description="Discrete-geometry transition weight in full-video identity smoothing",
    )
    adaptive_smoothing_velocity_weight: float = Field(
        default=0.55,
        ge=0.0,
        description="Explicit constant-velocity prediction weight in full-video smoothing",
    )
    adaptive_smoothing_jump_penalty: float = Field(
        default=6.0,
        ge=0.0,
        description=(
            "Non-capped quadratic penalty for frame-to-frame motion beyond the "
            "consistency distance"
        ),
    )
    adaptive_smoothing_max_states: int = Field(
        default=128,
        ge=4,
        le=4096,
        description="Maximum assignment states retained per frame for scalable smoothing",
    )


class ContactEventConfig(BaseModel):
    """Configuration for the auditable contact-event layer.

    The layer turns contact/overlap episodes into explicit merge/split event
    records with top-K incoming-to-outgoing identity hypotheses and
    best-vs-second-best score margins for review prioritization.
    """

    enabled: bool = Field(
        default=True,
        description="Write contact_events.json/csv audit artifacts after tracking",
    )
    contact_distance_px: float = Field(
        default=75.0,
        gt=0.0,
        description="Centroid distance below which two boxes count as in contact",
    )
    iou_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Pairwise IoU above which two boxes count as in contact",
    )
    merge_iou: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description=(
            "IoU plus center containment above which two observed boxes count as "
            "a merged measurement"
        ),
    )
    recovery_window: int = Field(
        default=30,
        ge=1,
        description="Frames after an event to look for a split (recovered observations)",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=16,
        description="Number of incoming-to-outgoing identity hypotheses kept per event",
    )
    max_hypothesis_participants: int = Field(
        default=8,
        ge=2,
        le=16,
        description="Skip hypothesis enumeration when more participants share one event",
    )
    velocity_window: int = Field(
        default=7,
        ge=1,
        description="Observed pre-event steps used for the incoming velocity estimate",
    )
    motion_noise_px: float = Field(
        default=12.0,
        gt=0.0,
        description=(
            "Per-frame motion noise scale; the extrapolation gate grows with the "
            "square root of the extrapolated gap"
        ),
    )
    size_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of the squared log-area-ratio term in hypothesis cost",
    )
    hypothesis_temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="Softmax temperature converting hypothesis cost to relative score",
    )
    confident_margin: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Best-vs-second-best margin at or above which an event is confident",
    )
    ambiguous_margin: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Margin below which an event is uncertain instead of ambiguous",
    )
    episode_gap_frames: int = Field(
        default=120,
        ge=0,
        description=(
            "Maximum frame gap for chaining tracker-disagreement events with "
            "overlapping participants into one label-inversion episode"
        ),
    )


class UDMTConfig(BaseModel):
    """Configuration for external UDMT research tracking backend."""

    enabled: bool = Field(
        default=False,
        description="Enable UDMT external tracking backend (requires separate CUDA environment)",
    )
    udmt_home: str = Field(
        default="",
        description="Path to UDMT repository checkout",
    )
    python_executable: str = Field(
        default="python",
        description="Python executable in the UDMT conda/venv environment",
    )
    license_accepted: bool = Field(
        default=False,
        description="Confirm acceptance of UDMT's academic/non-profit research license",
    )
    cuda_visible_devices: str | None = Field(
        default=None,
        description="CUDA_VISIBLE_DEVICES override for UDMT",
    )
    auto_init: bool = Field(
        default=True,
        description="Use DenseTrack auto-init to provide seed points instead of manual clicks",
    )


class GlobalOptimizerConfig(BaseModel):
    """Configuration for global tracklet-to-trajectory optimization."""

    method: str = Field(
        default="greedy",
        description="Optimization method: greedy | hungarian | min_cost_flow | ilp | long_term",
    )
    max_temporal_gap: int = Field(
        default=250,
        description="Max frame gap between tracklets to consider linking (~10 sec at 25fps)",
    )
    min_link_confidence: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum confidence required to link two tracklets",
    )
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "motion": 0.25,
            "appearance": 0.20,
            "size": 0.15,
            "orientation": 0.10,
            "pose": 0.15,
            "temporal_gap": 0.15,
        },
        description="Weights for multi-factor similarity scoring",
    )
    long_term_geometry_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=0.75,
        description="HieDG-inspired discrete-geometry weight for long-term stitching",
    )
    long_term_velocity_window: int = Field(
        default=7,
        ge=2,
        le=60,
        description="Endpoint displacement window used to reject one-frame motion jitter",
    )
    long_term_coarse_bins: int = Field(default=8, ge=2, le=32)
    long_term_fine_bins: int = Field(default=8, ge=2, le=32)
    closed_set_stitching: bool = Field(
        default=True,
        description="Prefer a path cover with the known animal count when valid links permit it",
    )


# ---------------------------------------------------------------------------
# ReID Config
# ---------------------------------------------------------------------------


class ReIDConfig(BaseModel):
    """Configuration for self-supervised ReID learning."""

    enabled: bool = Field(
        default=True,
        description="Enable self-supervised ReID (requires sufficient high-conf tracklets)",
    )
    feat_dim: int = Field(default=128)
    backbone: str = Field(default="resnet18")
    confidence_threshold: float = Field(
        default=0.8,
        description="Minimum tracklet confidence to use for ReID training",
    )
    min_triplets: int = Field(
        default=50,
        description="Minimum number of triplets required to train ReID",
    )
    epochs: int = Field(default=10)
    learning_rate: float = Field(default=1e-4)
    margin: float = Field(default=0.3, description="Triplet loss margin")
    max_iterations: int = Field(
        default=3,
        description="Max self-improvement iterations per video",
    )


class RepresentationConfig(BaseModel):
    """Annotation-free visual SSL representation outputs inspired by BEAST3D."""

    enabled: bool = Field(
        default=True,
        description="Export unlabeled trajectory/pose SSL representations after tracking",
    )
    dim: int = Field(
        default=8,
        ge=2,
        le=64,
        description="Output embedding dimension for transfer features",
    )
    window_frames: int = Field(
        default=15,
        ge=1,
        description="Temporal smoothing window used for representation features",
    )
    sample_every: int = Field(
        default=3,
        ge=1,
        description="Sample every Nth tracked frame for SSL embeddings",
    )
    camera_manifest: str | None = Field(
        default=None,
        description="Optional multi-view camera/session manifest for BEAST3D-style readiness audit",
    )


class BioacousticSSLConfig(BaseModel):
    """Species-specific bioacoustic SSL outputs inspired by Dolph2Vec."""

    enabled: bool = Field(
        default=False,
        description="Analyze a WAV audio file for annotation-free communication behavior features",
    )
    audio_path: str | None = Field(
        default=None,
        description="Optional WAV file path to analyze alongside the video",
    )
    species: str = Field(
        default="dolphin",
        description="Species-specific acoustic domain label",
    )
    window_ms: float = Field(default=120.0, gt=0.0)
    hop_ms: float = Field(default=40.0, gt=0.0)
    n_bands: int = Field(default=16, ge=4, le=96)
    codebook_size: int = Field(default=12, ge=2, le=128)
    whistle_min_frequency_hz: float = Field(default=1000.0, ge=0.0)


# ---------------------------------------------------------------------------
# Experiment Template Config
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    """Behavior-study template and zone-metric export settings."""

    enabled: bool = Field(
        default=True,
        description="Export template-driven zone/time-bin experiment metrics after tracking",
    )
    template_id: str = Field(
        default="open_field",
        description=(
            "Experiment template: open_field | morris_water_maze | three_chamber_social | "
            "light_dark_box | elevated_plus_maze | cpp"
        ),
    )
    time_bin_s: float = Field(
        default=60.0,
        gt=0.0,
        description="Time-bin size for experiment metrics in seconds",
    )


# ---------------------------------------------------------------------------
# Confidence Config
# ---------------------------------------------------------------------------


class ConfidenceConfig(BaseModel):
    """Confidence thresholds and audit settings."""

    high_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    medium_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    low_threshold: float = Field(default=0.2, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Export Config
# ---------------------------------------------------------------------------


class ExportConfig(BaseModel):
    """Export format and filtering settings."""

    formats: list[str] = Field(
        default_factory=lambda: ["mot", "coco", "csv"],
        description="Export formats: mot | coco | dlc | sleap | csv",
    )
    min_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Minimum confidence to include in export (0.0 = all)",
    )
    overlay_video: bool = Field(
        default=True,
        description="Generate overlay visualization video",
    )
    overlay_fps: float | None = Field(
        default=None,
        description="Override output video FPS (None = match input)",
    )


# ---------------------------------------------------------------------------
# Top-Level Pipeline Config
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Top-level configuration for the full DenseTrack pipeline."""

    # Experiment metadata
    species: Species = Species.MOUSE
    n_animals: int = Field(
        default=4, ge=1, le=50,
        description="Number of animals in the video (used as constraint in global optimization)",
    )
    video_path: str | None = None
    output_dir: str = Field(default="results/")

    # Module configs
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    sam2: SAM2Config = Field(default_factory=SAM2Config)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    tracklet: TrackletConfig = Field(default_factory=TrackletConfig)
    global_optimizer: GlobalOptimizerConfig = Field(default_factory=GlobalOptimizerConfig)
    reid: ReIDConfig = Field(default_factory=ReIDConfig)
    representation: RepresentationConfig = Field(default_factory=RepresentationConfig)
    bioacoustic_ssl: BioacousticSSLConfig = Field(default_factory=BioacousticSSLConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    udmt: UDMTConfig = Field(default_factory=UDMTConfig)
    contact_events: ContactEventConfig = Field(default_factory=ContactEventConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    # Processing settings
    processing_resolution: int = Field(
        default=1080,
        description="Resize video to this height before processing (0 = no resize)",
    )
    device: str = Field(
        default="auto",
        description="Compute device: auto | cpu | cuda | mps",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(mode="json"),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    def resolve_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
