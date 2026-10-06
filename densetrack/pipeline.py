"""
Full pipeline orchestration for DenseTrack-AI.

Coordinates all modules:
  Detection → Pose → Tracklets → ReID → Global Optimization → Export

Each step produces intermediate results that feed into the next,
with confidence scoring throughout.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2

from densetrack.config import PipelineConfig
from densetrack.confidence.auditor import ConfidenceAuditor
from densetrack.confidence.report import ReportGenerator
from densetrack.detection import create_detector
from densetrack.export.coco_format import COCOExporter
from densetrack.export.dlc_format import DLCExporter
from densetrack.export.mot_format import MOTExporter
from densetrack.export.overlay_video import OverlayVideoGenerator
from densetrack.export.sleap_format import SLEAPExporter
from densetrack.models import Trajectory
from densetrack.pose import create_pose_estimator
from densetrack.reid.self_supervised import SelfSupervisedReID
from densetrack.representation.visual_ssl import (
    VisualSSLRepresentationAnalyzer,
    VisualSSLRepresentationConfig,
)
from densetrack.audio.bioacoustic_ssl import BioacousticSSLAnalyzer, BioacousticSSLConfig
from densetrack.tracking.global_optimizer import GlobalTrackletOptimizer
from densetrack.tracking.fixed_count_tracker import FixedCountTrackletGenerator
from densetrack.tracking.identity_continuity import write_identity_continuity_report
from densetrack.tracking.contact_events import write_contact_events_report
from densetrack.tracking.mac_adaptive_tracker import MacAdaptiveTrackletGenerator
from densetrack.tracking.norfair_tracker import NorfairTrackletGenerator
from densetrack.tracking.tracklet_generator import TrackletGenerator
from densetrack.utils.video import VideoReader, read_video_metadata

__all__ = ["Pipeline"]

logger = logging.getLogger(__name__)


class Pipeline:
    """
    DenseTrack-AI full processing pipeline.

    Usage:
        config = PipelineConfig.from_yaml("configs/mice_topview.yaml")
        config.video_path = "cage.mp4"
        config.n_animals = 5

        pipeline = Pipeline(config)
        results = pipeline.run()
    """

    def __init__(self, config: PipelineConfig, initialization: dict | None = None):
        self.config = config
        self.device = config.resolve_device()
        self.initialization = initialization or {}

        # Will be initialized lazily
        self._detector = None
        self._pose_estimator = None

    def run(self) -> dict:
        """
        Run the full AutoLabel MOT pipeline.

        Returns:
            Dict with:
            - trajectories: list of Trajectory objects
            - confidence_report: audit report dict
            - timings: dict of step durations
            - output_files: dict of generated file paths
        """
        if not self.config.video_path:
            raise ValueError("video_path must be set in config")

        video_path = self.config.video_path
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        meta = read_video_metadata(video_path)
        logger.info(f"Processing: {meta}")
        logger.info(f"Expected animals: {self.config.n_animals}")
        logger.info(f"Device: {self.device}")

        timings: dict[str, float] = {}
        output_files: dict[str, str] = {}

        # ━━━ Step 1+2: Detection + Pose (streaming) ━━━
        logger.info("═" * 50)
        logger.info("Step 1+2/6: Detection + Pose (streaming)")
        t0 = time.time()

        detector = create_detector(self.config.detection)
        detector.validate()
        pose_estimator = create_pose_estimator(self.config.pose)

        resize_h = self.config.processing_resolution if self.config.processing_resolution > 0 else 0

        detections_per_frame: dict[int, list] = {}
        poses_per_frame: dict[int, list] = {}
        frame_size: tuple[int, int] | None = None
        n_frames = 0

        # Stream frames: detect + pose in one pass, don't store raw pixels
        with VideoReader(video_path, resize_height=resize_h) as reader:
            for frame_idx, frame in reader:
                if frame_size is None:
                    frame_size = (frame.shape[1], frame.shape[0])

                # Detect
                dets = detector.detect_frame(frame, frame_idx)
                if dets:
                    detections_per_frame[frame_idx] = dets

                    # Extract crops for ReID if enabled
                    if self.config.reid.enabled or self.config.tracklet.backend.lower() == "mac_adaptive":
                        for det in dets:
                            h, w = frame.shape[:2]
                            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            if x2 - x1 >= 10 and y2 - y1 >= 10:
                                crop = frame[y1:y2, x1:x2]
                                success, encoded = cv2.imencode('.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                                if success:
                                    det.crop_bytes = encoded.tobytes()

                    # Pose
                    poses = pose_estimator.estimate_batch(frame, dets)
                    if poses:
                        poses_per_frame[frame_idx] = poses

                n_frames += 1
                if n_frames % 1000 == 0:
                    total_dets = sum(len(d) for d in detections_per_frame.values())
                    logger.info(
                        f"  ... {n_frames}/{meta.total_frames} frames, "
                        f"{total_dets} detections"
                    )

        timings["detection_and_pose"] = time.time() - t0
        total_dets = sum(len(d) for d in detections_per_frame.values())
        total_poses = sum(len(p) for p in poses_per_frame.values())
        fps_det = n_frames / max(timings["detection_and_pose"], 1e-6)
        logger.info(
            f"  → {total_dets} detections, {total_poses} poses "
            f"in {timings['detection_and_pose']:.1f}s ({fps_det:.1f} FPS)"
        )

        # ━━━ Step 3: Tracklet Generation ━━━
        logger.info("Step 3/6: Tracklet Generation")
        t0 = time.time()

        if self.config.tracklet.backend.lower() == "norfair":
            tracklet_gen = NorfairTrackletGenerator(self.config.tracklet)
            tracklets = tracklet_gen.generate(detections_per_frame)
        elif self.config.tracklet.backend.lower() == "fixed_count":
            tracklet_gen = FixedCountTrackletGenerator(
                self.config.tracklet,
                n_animals=self.config.n_animals,
                frame_size=frame_size,
            )
            tracklets = tracklet_gen.generate(detections_per_frame)
        elif self.config.tracklet.backend.lower() == "mac_adaptive":
            tracklet_gen = MacAdaptiveTrackletGenerator(
                self.config.tracklet,
                n_animals=self.config.n_animals,
                frame_size=frame_size,
                initialization_points=(
                    self.initialization.get("points", [])
                    if self.initialization.get("initialization_mode") == "manual_first_frame_click"
                    else None
                ),
                initialization_frame=self.initialization.get("selected_frame"),
            )
            tracklets = tracklet_gen.generate(
                detections_per_frame,
                total_frames=n_frames,
            )
        elif self.config.tracklet.backend.lower() == "udmt":
            tracklet_gen = None
            tracklets = self._run_udmt_tracking(
                video_path, detections_per_frame, n_frames
            )
        else:
            tracklet_gen = TrackletGenerator(self.config.tracklet)
            tracklets = tracklet_gen.generate(detections_per_frame)
        fixed_count_stats = getattr(tracklet_gen, "stats", None) if tracklet_gen is not None else None
        if isinstance(tracklet_gen, MacAdaptiveTrackletGenerator):
            output_files.update(tracklet_gen.write_identity_risk(output_dir))
        if fixed_count_stats is not None:
            logger.info(
                "  → Constrained tracking stats: observed %.2f/%s tracks/frame, "
                "%.1f%% frames with ≥%s observed tracks",
                fixed_count_stats.avg_observed_tracks_per_frame,
                self.config.n_animals,
                fixed_count_stats.mostly_observed_ratio * 100,
                max(1, self.config.n_animals - 1),
            )

        # Attach poses to tracklets
        for tracklet in tracklets:
            for det in tracklet.detections:
                frame_poses = poses_per_frame.get(det.frame_idx, [])
                for pose in frame_poses:
                    if pose.detection_id == det.detection_id:
                        tracklet.poses.append(pose)
                        break

        timings["tracklet"] = time.time() - t0
        logger.info(f"  → {len(tracklets)} tracklets in {timings['tracklet']:.1f}s")

        # ━━━ Step 4: Self-Supervised ReID ━━━
        reid_similarities = None
        if self.config.reid.enabled:
            logger.info("Step 4/6: Self-Supervised ReID")
            t0 = time.time()

            reid = SelfSupervisedReID(self.config.reid, device=self.device)
            reid_similarities = reid.run_self_improvement_loop(tracklets, None)

            timings["reid"] = time.time() - t0
            logger.info(
                f"  → {len(reid_similarities)} similarities in {timings['reid']:.1f}s"
            )
        else:
            logger.info("Step 4/6: ReID (skipped)")
            timings["reid"] = 0

        # ━━━ Step 5: Global Optimization ━━━
        logger.info("Step 5/6: Global Trajectory Optimization")
        t0 = time.time()

        optimizer = GlobalTrackletOptimizer(
            self.config.global_optimizer,
            n_animals=self.config.n_animals,
            frame_size=frame_size,
        )
        trajectories = optimizer.optimize(tracklets, reid_similarities)
        output_files.update(
            write_identity_continuity_report(
                output_dir,
                config=self.config,
                tracklets=tracklets,
                trajectories=trajectories,
                tracker_stats=fixed_count_stats,
                optimizer_stats=optimizer.stats,
            )
        )
        if self.config.contact_events.enabled:
            contact_outputs, contact_summary = write_contact_events_report(
                output_dir,
                config=self.config.contact_events,
                trajectories=trajectories,
                n_animals=self.config.n_animals,
                source_description="pipeline",
            )
            output_files.update(contact_outputs)
            logger.info(
                "  → Contact events: %s total, %s uncertain/unresolved, "
                "%s inversion episodes (%s raw disagreements)",
                contact_summary.get("total_events", 0),
                contact_summary.get("uncertainty_counts", {}).get("uncertain", 0)
                + contact_summary.get("unresolved_events", 0),
                contact_summary.get("inversion_episode_count", 0),
                contact_summary.get("tracker_disagreement_events", 0),
            )

        timings["optimization"] = time.time() - t0
        logger.info(
            f"  → {len(trajectories)} trajectories in {timings['optimization']:.1f}s"
        )

        # ━━━ Step 6: Confidence Audit + Export ━━━
        logger.info("Step 6/6: Confidence Audit + Export")
        t0 = time.time()

        # Confidence report
        auditor = ConfidenceAuditor(self.config.confidence)
        report = auditor.generate_report(trajectories)

        report_gen = ReportGenerator()
        report_path = output_dir / "confidence_report.json"
        report_gen.to_json(report, report_path)
        output_files["confidence_report"] = str(report_path)

        # Print summary
        logger.info("\n" + report_gen.to_text(report))

        # Export formats
        poses_by_detection = {
            pose.detection_id: pose
            for frame_poses in poses_per_frame.values()
            for pose in frame_poses
        }
        for fmt in self.config.export.formats:
            if fmt == "mot":
                exporter = MOTExporter()
                mot_path = output_dir / "tracks.mot"
                exporter.export(trajectories, mot_path, self.config.export.min_confidence)
                output_files["mot"] = str(mot_path)

            elif fmt == "coco":
                exporter = COCOExporter()
                coco_path = output_dir / "detections.coco.json"
                exporter.export(
                    trajectories,
                    video_metadata={"width": meta.width, "height": meta.height},
                    output_path=coco_path,
                    min_confidence=self.config.export.min_confidence,
                )
                output_files["coco"] = str(coco_path)

            elif fmt == "csv":
                exporter = DLCExporter()
                csv_path = output_dir / "poses.csv"
                exporter.export(
                    trajectories,
                    poses=poses_by_detection,
                    output_path=csv_path,
                )
                output_files["csv"] = str(csv_path)

            elif fmt == "sleap":
                exporter = SLEAPExporter()
                sleap_path = output_dir / "labels.sleap.json"
                exporter.export(
                    trajectories,
                    output_path=sleap_path,
                    video_path=video_path,
                    video_metadata={"width": meta.width, "height": meta.height, "fps": meta.fps},
                    min_confidence=self.config.export.min_confidence,
                )
                output_files["sleap"] = str(sleap_path)

        # Overlay video
        if self.config.export.overlay_video:
            overlay = OverlayVideoGenerator()
            overlay_path = output_dir / "overlay.mp4"
            overlay.generate(
                video_path, trajectories, overlay_path,
                fps=self.config.export.overlay_fps,
            )
            output_files["overlay_video"] = str(overlay_path)

        if self.config.representation.enabled and output_files.get("mot"):
            visual_json = output_dir / "visual_ssl_report.json"
            visual_csv = output_dir / "visual_ssl_embeddings.csv"
            visual_report = VisualSSLRepresentationAnalyzer(
                VisualSSLRepresentationConfig(
                    dim=self.config.representation.dim,
                    window_frames=self.config.representation.window_frames,
                    sample_every=self.config.representation.sample_every,
                    camera_manifest=self.config.representation.camera_manifest,
                )
            ).analyze_run(
                run_dir=output_dir,
                output_json=visual_json,
                output_csv=visual_csv,
            )
            output_files["visual_ssl_report"] = str(visual_json)
            output_files["visual_ssl_embeddings"] = str(visual_csv)
            logger.info(
                "  → Visual SSL representations: %s samples, %s-D",
                visual_report.get("summary", {}).get("num_samples", 0),
                visual_report.get("summary", {}).get("embedding_dim", 0),
            )

        if self.config.bioacoustic_ssl.enabled and self.config.bioacoustic_ssl.audio_path:
            audio_path = Path(self.config.bioacoustic_ssl.audio_path).expanduser()
            if audio_path.exists():
                audio_json = output_dir / "bioacoustic_ssl_report.json"
                audio_csv = output_dir / "bioacoustic_ssl_units.csv"
                audio_report = BioacousticSSLAnalyzer(
                    BioacousticSSLConfig(
                        species=self.config.bioacoustic_ssl.species,
                        window_ms=self.config.bioacoustic_ssl.window_ms,
                        hop_ms=self.config.bioacoustic_ssl.hop_ms,
                        n_bands=self.config.bioacoustic_ssl.n_bands,
                        codebook_size=self.config.bioacoustic_ssl.codebook_size,
                        whistle_min_frequency_hz=self.config.bioacoustic_ssl.whistle_min_frequency_hz,
                    )
                ).analyze_wav(
                    audio_path=audio_path,
                    output_json=audio_json,
                    output_csv=audio_csv,
                )
                output_files["bioacoustic_ssl_report"] = str(audio_json)
                output_files["bioacoustic_ssl_units"] = str(audio_csv)
                logger.info(
                    "  → Bioacoustic SSL: %s windows, %s whistle candidates",
                    audio_report.get("summary", {}).get("num_windows", 0),
                    audio_report.get("summary", {}).get("whistle_candidate_windows", 0),
                )
            else:
                logger.warning("Bioacoustic SSL audio path not found: %s", audio_path)

        timings["export"] = time.time() - t0

        # Total
        timings["total"] = sum(timings.values())
        fps = meta.total_frames / max(timings["total"], 1e-6)
        logger.info(
            f"\n{'═' * 50}\n"
            f"  Total: {timings['total']:.1f}s "
            f"({fps:.2f} FPS)\n"
            f"{'═' * 50}"
        )

        result = {
            "trajectories": trajectories,
            "confidence_report": report,
            "timings": timings,
            "output_files": output_files,
            "fixed_count_stats": fixed_count_stats,
            "video_metadata": {
                "path": meta.path,
                "width": meta.width,
                "height": meta.height,
                "fps": meta.fps,
                "total_frames": meta.total_frames,
            },
        }

        summary_path = output_dir / "run_summary.json"
        output_files["run_summary"] = str(summary_path)
        self._write_run_summary(
            summary_path=summary_path,
            result=result,
            fixed_count_stats=fixed_count_stats,
            started_at=None,
        )


        return result

    def _write_run_summary(
        self,
        summary_path: Path,
        result: dict,
        fixed_count_stats,
        started_at: str | None = None,
    ) -> None:
        """Write a compact machine-readable run summary and provenance."""
        stats_payload = (
            asdict(fixed_count_stats)
            if fixed_count_stats is not None and is_dataclass(fixed_count_stats)
            else None
        )
        summary = {
            "created_at": datetime.now().isoformat(),
            "started_at": started_at,
            "config": self.config.model_dump(mode="json"),
            "video_metadata": result.get("video_metadata", {}),
            "timings": result.get("timings", {}),
            "output_files": result.get("output_files", {}),
            "fixed_count_stats": stats_payload,
            "confidence_summary": result.get("confidence_report", {}).get("summary", {}),
            "provenance": {
                "generator": "DenseTrack-AI",
                "detection_backend": self.config.detection.backend,
                "pose_backend": self.config.pose.backend,
                "tracklet_backend": self.config.tracklet.backend,
                "udmt_enabled": self.config.udmt.enabled,
                "reid_enabled": self.config.reid.enabled,
                "optimizer": self.config.global_optimizer.method,
                "identity_profile": (
                    "research_long_term_identity"
                    if (
                        (
                            self.config.tracklet.backend.lower() == "mac_adaptive"
                            and self.config.tracklet.adaptive_discrete_geometry
                        )
                        or self.config.global_optimizer.method.lower() == "long_term"
                    )
                    else "configured_baseline"
                ),
                "hierarchical_discrete_geometry": (
                    self.config.tracklet.backend.lower() == "mac_adaptive"
                    and self.config.tracklet.adaptive_discrete_geometry
                ),
                "third_party_models_bundled": False,
            },
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Run summary saved to {summary_path}")

    def _run_udmt_tracking(
        self,
        video_path: str,
        detections_per_frame: dict[int, list],
        n_frames: int,
    ) -> list:
        """Run UDMT external tracker as the tracking backend.

        Uses DenseTrack's detector output to provide automatic
        first-frame initialization, replacing UDMT's manual clicks.
        """
        from densetrack.external.udmt import UDMTAdapter, UDMTExternalConfig

        udmt_cfg = self.config.udmt
        if not udmt_cfg.enabled:
            raise ValueError(
                "UDMT backend selected (tracklet.backend='udmt') but udmt.enabled is False. "
                "Set udmt.enabled=true and provide udmt.udmt_home."
            )

        adapter = UDMTAdapter(
            UDMTExternalConfig(
                udmt_home=udmt_cfg.udmt_home,
                python_executable=udmt_cfg.python_executable,
                license_accepted=udmt_cfg.license_accepted,
                cuda_visible_devices=udmt_cfg.cuda_visible_devices,
            )
        )
        adapter.raise_for_invalid(check_import=True)

        # Use DenseTrack detector output for automatic first-frame initialization
        init_positions = None
        if udmt_cfg.auto_init and detections_per_frame:
            first_frame_idx = min(detections_per_frame.keys())
            first_dets = detections_per_frame[first_frame_idx]
            init_positions = [
                det.centroid for det in first_dets[:self.config.n_animals]
            ]
            logger.info(
                f"UDMT auto-init: {len(init_positions)} seed points from frame {first_frame_idx}"
            )

        logger.info(
            f"Launching UDMT external tracker on {video_path} "
            f"(n_animals={self.config.n_animals})"
        )

        # The official UDMT workflow is GUI-centered. Keep execution external
        # until a Linux/CUDA headless runner contract is verified end to end.
        raise NotImplementedError(
            "UDMT direct pipeline execution requires a verified Linux/CUDA headless "
            "runner contract. / UDMT 直接管线执行仍需验证 Linux/CUDA 无头运行契约。\n"
            "Current workflow / 当前流程:\n"
            f"  1. densetrack udmt-handoff {video_path} --n-animals {self.config.n_animals}\n"
            "  2. Run the official UDMT GUI externally / 在外部运行官方 UDMT GUI\n"
            "  3. densetrack udmt-import <tracks.mot>\n"
            "  4. Or use the native backend / 或使用本地后端: --detector dark_object"
        )

    def _maybe_apply_sam2(
        self,
        video_path: str,
        frames: dict[int, np.ndarray],
        detections_per_frame: dict[int, list],
        detector,
    ) -> dict[int, list]:
        """
        Optionally refine detections with SAM 2 video propagation.

        This is deliberately conservative: if SAM 2 is unavailable, not
        initialized, or returns too little data, the original detector output is
        kept. A bad mask propagation pass should not erase usable detections.
        """
        if not self.config.sam2.enabled:
            return detections_per_frame

        try:
            from densetrack.detection.sam2_propagator import SAM2Propagator
        except Exception as exc:
            logger.warning(f"SAM 2 adapter unavailable, skipping mask propagation: {exc}")
            return detections_per_frame

        propagator = SAM2Propagator(self.config.sam2)
        if not propagator.available:
            logger.info("SAM 2 not installed; keeping detector-only results")
            return detections_per_frame

        initial_frame_idx = None
        initial_detections = []
        for frame_idx in sorted(detections_per_frame):
            dets = detections_per_frame[frame_idx]
            if dets:
                initial_frame_idx = frame_idx
                initial_detections = dets
                break

        if initial_frame_idx is None:
            logger.warning("SAM 2 skipped because no initial detections are available")
            return detections_per_frame

        logger.info(f"Applying SAM 2 mask propagation from frame {initial_frame_idx}")
        try:
            propagated = propagator.propagate_video(
                video_path=video_path,
                initial_detections=initial_detections,
                redetect_fn=detector.detect_frame,
                redetect_interval=self.config.detection.redetect_interval,
            )
        except Exception as exc:
            logger.warning(f"SAM 2 propagation failed, keeping detector results: {exc}")
            return detections_per_frame

        propagated_count = sum(len(dets) for dets in propagated.values())
        if propagated_count == 0 or len(propagated) < max(10, len(frames) // 10):
            logger.warning(
                "SAM 2 propagation produced insufficient detections "
                f"({propagated_count} objects across {len(propagated)} frames); "
                "keeping detector results"
            )
            return detections_per_frame

        logger.info(
            f"SAM 2 accepted: {propagated_count} propagated detections "
            f"across {len(propagated)} frames"
        )
        return propagated
