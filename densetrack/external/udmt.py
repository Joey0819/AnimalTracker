"""Optional adapter for the external UDMT research tracker.

UDMT is not vendored because its public repository is distributed for
academic/non-profit non-commercial research use. This module only validates an
external installation and imports tracking output into DenseTrack formats.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from densetrack.export.run_reexport import load_mot_rows, reexport_run

__all__ = [
    "UDMTAdapter",
    "UDMTEnvironmentError",
    "UDMTExternalConfig",
    "build_udmt_handoff_package",
]


class UDMTEnvironmentError(RuntimeError):
    """Raised when an external UDMT environment is not ready to run."""


@dataclass(frozen=True)
class UDMTExternalConfig:
    """Configuration for an external UDMT checkout/environment."""

    udmt_home: str | Path
    python_executable: str = "python"
    license_accepted: bool = False
    cuda_visible_devices: str | None = None


class UDMTAdapter:
    """Validate and bridge an external UDMT run into DenseTrack outputs."""

    def __init__(self, config: UDMTExternalConfig) -> None:
        self.config = config
        self.udmt_home = Path(config.udmt_home).expanduser().resolve()

    def launch_command(self) -> list[str]:
        """Return the canonical UDMT GUI launch command."""
        return [self.config.python_executable, "-m", "udmt.gui.launch_script"]

    def launch_environment(self) -> dict[str, str]:
        """Return environment additions needed when running from a source checkout."""
        env: dict[str, str] = {"PYTHONPATH": str(self.udmt_home)}
        if self.config.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices
        return env

    def validate(self, check_import: bool = True) -> dict[str, Any]:
        """Inspect whether the external UDMT installation is usable."""
        errors: list[str] = []
        warnings: list[str] = []

        if not self.config.license_accepted:
            errors.append(
                "UDMT license acceptance is required before use. The public "
                "repository is for academic/non-profit non-commercial research use."
            )

        if not self.udmt_home.exists():
            errors.append(f"UDMT_HOME does not exist: {self.udmt_home}")
        else:
            if not (self.udmt_home / "udmt").is_dir():
                errors.append(f"UDMT package directory not found: {self.udmt_home / 'udmt'}")
            if not (self.udmt_home / "LICENSE").exists():
                warnings.append(f"UDMT LICENSE file not found under: {self.udmt_home}")
            if not (self.udmt_home / "README.md").exists():
                warnings.append(f"UDMT README.md not found under: {self.udmt_home}")

        resolved_python = shutil.which(self.config.python_executable)
        python_path = Path(self.config.python_executable).expanduser()
        if resolved_python is None and not python_path.exists():
            errors.append(f"Python executable not found: {self.config.python_executable}")

        if platform.system().lower() != "linux":
            warnings.append(
                "UDMT upstream recommends Linux with NVIDIA CUDA; this host is "
                f"{platform.system()}."
            )

        if check_import and not errors:
            import_report = self._check_python_import()
            if import_report["returncode"] != 0:
                errors.append(
                    "UDMT could not be imported by the configured Python. "
                    f"stderr: {import_report['stderr'][:500]}"
                )

        return {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "udmt_home": str(self.udmt_home),
            "python_executable": self.config.python_executable,
            "launch_command": " ".join(self.launch_command()),
            "environment": self.launch_environment(),
        }

    def raise_for_invalid(self, check_import: bool = True) -> None:
        """Raise a readable exception when validation fails."""
        report = self.validate(check_import=check_import)
        if not report["ok"]:
            raise UDMTEnvironmentError("\n".join(report["errors"]))

    def import_mot_tracks(
        self,
        source_path: str | Path,
        output_dir: str | Path,
        video_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Import a UDMT-derived MOT-like track file into DenseTrack outputs.

        The current public UDMT project is GUI-centered. Until its stable output
        contract is pinned for this project, this bridge accepts MOT Challenge
        rows: frame,id,x,y,width,height,confidence,...
        """
        source = Path(source_path).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"UDMT track file not found: {source}")

        rows = load_mot_rows(source)
        if not rows:
            raise ValueError(f"No MOT rows could be read from: {source}")

        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        tracks_path = output / "tracks.mot"

        lines = [
            "# DenseTrack-AI imported external UDMT MOT tracks",
            f"# Generated: {datetime.now().isoformat()}",
            f"# Source: {source}",
            "# Format: frame,id,bb_left,bb_top,bb_width,bb_height,conf,-1,-1,-1",
        ]
        for row in sorted(rows, key=lambda item: (item.frame, item.track_id)):
            lines.append(
                f"{row.frame},{row.track_id},{row.x:.2f},{row.y:.2f},"
                f"{row.width:.2f},{row.height:.2f},{row.confidence:.4f},-1,-1,-1"
            )
        tracks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        frames = sorted({row.frame for row in rows})
        track_ids = sorted({row.track_id for row in rows})
        summary = {
            "created_at": datetime.now().isoformat(),
            "pipeline": "udmt_external_import",
            "external_backend": "udmt",
            "source_path": str(source),
            "video_metadata": video_metadata or {},
            "tracking_summary": {
                "num_rows": len(rows),
                "num_frames": len(frames),
                "frame_min": frames[0],
                "frame_max": frames[-1],
                "num_tracks": len(track_ids),
                "track_ids": track_ids,
            },
            "provenance": {
                "adapter": "densetrack.external.udmt",
                "license_boundary": "UDMT is external and is not bundled with DenseTrack-AI.",
            },
        }
        summary_path = output / "run_summary.json"
        _write_json(summary_path, summary)

        outputs = {
            "tracks_mot": str(tracks_path),
            "run_summary": str(summary_path),
            "coco": str(reexport_run(output, output / "detections.coco.json", "coco")),
            "csv": str(reexport_run(output, output / "poses.csv", "csv")),
            "sleap": str(reexport_run(output, output / "labels.sleap.json", "sleap")),
        }
        summary["output_files"] = outputs
        _write_json(summary_path, summary)
        return summary

    def _check_python_import(self) -> dict[str, Any]:
        env = os.environ.copy()
        env_updates = self.launch_environment()
        if "PYTHONPATH" in env and env["PYTHONPATH"]:
            env_updates["PYTHONPATH"] = env_updates["PYTHONPATH"] + os.pathsep + env["PYTHONPATH"]
        env.update(env_updates)

        try:
            proc = subprocess.run(
                [
                    self.config.python_executable,
                    "-c",
                    "import udmt; print(getattr(udmt, '__file__', 'udmt'))",
                ],
                cwd=self.udmt_home,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "returncode": 124,
                "stdout": exc.stdout or "",
                "stderr": "Timed out while importing udmt.",
            }

        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }


def build_udmt_handoff_package(
    video_path: str | Path,
    initialization_json: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write a file-based UDMT handoff without bundling the external runtime."""
    video = Path(video_path).expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(f"Video not found / 未找到视频: {video}")
    initialization = Path(initialization_json).expanduser().resolve()
    if not initialization.exists():
        raise FileNotFoundError(
            f"Initialization JSON not found / 未找到自动初始化 JSON: {initialization}"
        )

    payload = json.loads(initialization.read_text(encoding="utf-8"))
    if payload.get("format") != "densetrack_auto_initialization_points":
        raise ValueError(
            "Unsupported initialization format / 不支持的自动初始化格式: "
            f"{payload.get('format')}"
        )

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    canonical_json = output / "initialization_points.json"
    if initialization != canonical_json:
        shutil.copy2(initialization, canonical_json)

    canonical_image = output / "initialization_points.jpg"
    source_image = initialization.with_name("initialization_points.jpg")
    if source_image.exists() and source_image.resolve() != canonical_image:
        shutil.copy2(source_image, canonical_image)
    elif not source_image.exists():
        canonical_image.unlink(missing_ok=True)

    requested = int(payload.get("n_animals_requested", 0) or 0)
    points = payload.get("points", [])
    seed_count = len(points)
    auto_ready = payload.get("status") == "ok" and requested > 0 and seed_count == requested
    selected_frame = payload.get("selected_frame")
    initialization_mode = payload.get("initialization_mode", "automatic")
    manual_first_frame_click_used = bool(
        payload.get("provenance", {}).get("manual_first_frame_click_used", False)
    )
    direct_drop_in_ready = auto_ready and selected_frame == 0
    start_pos_path = output / "start_pos_array.txt"
    _write_udmt_start_pos(start_pos_path, points)
    workspace_relative = Path("tmp") / video.stem / "extracted-images" / "start_pos_array.txt"
    workspace_seed_root = output / "udmt_workspace_seed"
    if workspace_seed_root.exists():
        shutil.rmtree(workspace_seed_root)
    workspace_seed_path = workspace_seed_root / workspace_relative
    workspace_seed_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(start_pos_path, workspace_seed_path)
    manifest = {
        "created_at": datetime.now().isoformat(),
        "format": "densetrack_udmt_external_handoff",
        "format_version": "1.0",
        "source_video": str(video),
        "external_backend": "udmt",
        "initialization": {
            "status": payload.get("status", "unknown"),
            "status_display": (
                (
                    "首帧点击辅助已就绪 / first-frame click assist ready"
                    if manual_first_frame_click_used
                    else "自动初始化已就绪 / automatic initialization ready"
                )
                if auto_ready
                else "自动初始化需复核 / automatic initialization requires review"
            ),
            "selected_frame": selected_frame,
            "score": payload.get("score", 0.0),
            "requested_animals": requested,
            "seed_points": seed_count,
            "points": points,
            "mode": initialization_mode,
            "mode_display": payload.get("initialization_mode_display"),
        },
        "automation_boundary": {
            "manual_training_annotation_required": False,
            "manual_training_annotation_used": False,
            "manual_first_frame_click_required": not auto_ready,
            "manual_first_frame_click_used": manual_first_frame_click_used,
            "fallback": (
                "No manual click is required when the external UDMT workspace uses "
                "the generated start_pos_array.txt. / 外部 UDMT 工作区使用生成的 "
                "start_pos_array.txt 后，无需人工点击。"
                if auto_ready
                else "Review or correct seed points before the external UDMT run. "
                "/ 外部 UDMT 运行前需复核或修正 seed 点。"
            ),
        },
        "upstream_contract": {
            "official_workspace_relative_destination": str(workspace_relative),
            "start_pos_array_format": "one x,y centroid per line",
            "coordinate_basis": payload.get("video", {}),
            "seed_frame": selected_frame,
            "direct_first_frame_drop_in_ready": direct_drop_in_ready,
            "requires_seed_frame_alignment": auto_ready and selected_frame != 0,
            "note": (
                "Copy the generated start_pos_array.txt into the official UDMT workspace "
                "path after project creation. If seed_frame is not 0, align the external "
                "video start or use a runner extension. / 创建 UDMT 项目后，将生成的 "
                "start_pos_array.txt 复制到官方工作区路径。若 seed_frame 不是 0，"
                "需对齐外部视频起点或扩展 runner。"
            ),
        },
        "files": {
            "initialization_points": str(canonical_json),
            "initialization_preview": str(canonical_image) if canonical_image.exists() else None,
            "udmt_start_pos_array": str(start_pos_path),
            "udmt_workspace_seed": str(workspace_seed_path),
        },
        "return_path": {
            "expected_track_format": "MOT-like CSV rows: frame,id,x,y,width,height,confidence,...",
            "import_command": "densetrack udmt-import /path/to/udmt_tracks.mot --output results/udmt_import",
        },
        "provenance": {
            "generator": "densetrack.external.udmt.build_udmt_handoff_package",
            "runtime_boundary": "UDMT source and weights remain external to DenseTrack-AI.",
        },
    }
    manifest_path = output / "udmt_handoff.json"
    _write_json(manifest_path, manifest)
    readme_path = output / "README.md"
    readme_path.write_text(_handoff_readme(manifest), encoding="utf-8")
    manifest["files"]["manifest"] = str(manifest_path)
    manifest["files"]["instructions"] = str(readme_path)
    _write_json(manifest_path, manifest)
    return manifest


def _handoff_readme(manifest: dict[str, Any]) -> str:
    initialization = manifest["initialization"]
    boundary = manifest["automation_boundary"]
    contract = manifest["upstream_contract"]
    return f"""# DenseTrack-AI -> UDMT External Handoff / 外部衔接包

## Initialization / 自动初始化

- Video / 视频: `{manifest["source_video"]}`
- Status / 状态: {initialization["status_display"]}
- Selected frame / 选定帧: `{initialization["selected_frame"]}`
- Score / 得分: `{float(initialization["score"]):.4f}`
- Seed points / 个体种子点: `{initialization["seed_points"]}/{initialization["requested_animals"]}`
- Manual training annotation required / 是否需要训练标注: `{str(boundary["manual_training_annotation_required"]).lower()}`
- Manual training annotation used / 是否使用训练标注: `{str(boundary["manual_training_annotation_used"]).lower()}`
- Manual first-frame click required / 是否需要第一帧人工点击: `{str(boundary["manual_first_frame_click_required"]).lower()}`
- Manual first-frame click used / 是否使用第一帧人工点击: `{str(boundary["manual_first_frame_click_used"]).lower()}`
- Direct first-frame drop-in ready / 是否可直接写入 UDMT 首帧: `{str(contract["direct_first_frame_drop_in_ready"]).lower()}`

## External Run / 外部运行

After creating the official UDMT project, copy `start_pos_array.txt` into:

```text
{contract["official_workspace_relative_destination"]}
```

The source video is referenced by absolute path and is not copied into this package.
If the seed frame is not `0`, align the external video start or use a runner extension.

创建官方 UDMT 项目后，将 `start_pos_array.txt` 复制到上述路径。此衔接包仅引用视频绝对路径，
不复制视频。如果 seed 帧不是 `0`，需对齐外部视频起点或扩展 runner。

## Return Results / 回传结果

Return MOT-like rows and import them with:

```bash
{manifest["return_path"]["import_command"]}
```

UDMT source code and weights remain external to DenseTrack-AI.
UDMT 源码与权重保持为 DenseTrack-AI 之外的外部依赖。
"""


def _write_udmt_start_pos(path: Path, points: list[dict[str, Any]]) -> None:
    """Write UDMT's official start_pos_array.txt centroid format."""
    lines = [f"{float(point['x']):.2f},{float(point['y']):.2f}" for point in points]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
