"""Local bilingual web workspace for zero-click DenseTrack jobs."""

from __future__ import annotations

import json
import mimetypes
import re
import shutil
import threading
import uuid
import warnings
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import unquote, urlparse

import cv2

from densetrack.config import PipelineConfig
from densetrack.initialization.auto_seed import (
    create_manual_initialization_points,
    write_auto_init_outputs,
)
from densetrack.workflows.complete_local import run_complete_local

__all__ = ["JobRegistry", "render_index_html", "serve_webapp"]

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
TRACKING_PROFILES = {
    "research_long_term": "科研长时身份 / Research long-term identity",
    "fast_baseline": "快速基线 / Fast baseline",
}


class JobRegistry:
    """Thread-safe persistent registry for local processing jobs."""

    def __init__(self, workspace: str | Path, config_path: str | Path):
        self.workspace = Path(workspace).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.jobs_dir = self.workspace / "jobs"
        self.jobs_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._load_existing()

    def submit(
        self,
        source: BinaryIO,
        filename: str,
        n_animals: int,
        clusters: int,
        initialization_mode: str = "automatic",
        experiment_template: str = "open_field",
        tracking_profile: str = "research_long_term",
    ) -> dict[str, Any]:
        """Persist an upload and either schedule processing or await first-frame clicks."""
        from densetrack.experiments import get_experiment_template

        safe_name = Path(filename).name
        extension = Path(safe_name).suffix.lower()
        if extension not in VIDEO_EXTENSIONS:
            raise ValueError(
                "Unsupported video type / 不支持的视频类型: "
                f"{extension or '(none)'}. Allowed / 允许: {', '.join(sorted(VIDEO_EXTENSIONS))}"
            )

        job_id = uuid.uuid4().hex[:12]
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir()
        upload_path = job_dir / f"input{extension}"
        with upload_path.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)

        now = datetime.now().isoformat()
        click_assisted = initialization_mode == "manual_first_frame_click"
        if tracking_profile not in TRACKING_PROFILES:
            raise ValueError(
                f"Unsupported tracking profile / 不支持的追踪配置: {tracking_profile}"
            )
        template = get_experiment_template(experiment_template)
        job = {
            "id": job_id,
            "status": "awaiting_seed" if click_assisted else "queued",
            "status_display": "等待首帧点击 / awaiting first-frame clicks" if click_assisted else "排队中 / queued",
            "filename": safe_name,
            "n_animals": n_animals,
            "clusters": clusters,
            "experiment_template": template.template_id,
            "experiment_template_display": template.display_name,
            "tracking_profile": tracking_profile,
            "tracking_profile_display": TRACKING_PROFILES[tracking_profile],
            "initialization_mode": initialization_mode,
            "initialization_mode_display": (
                "首帧点击辅助 / first-frame click assist"
                if click_assisted
                else "零点击自动初始化 / zero-click automatic initialization"
            ),
            "created_at": now,
            "updated_at": now,
            "result_url": None,
            "error": None,
        }
        if click_assisted:
            cfg = PipelineConfig.from_yaml(self.config_path)
            preview_path = job_dir / "initialization" / "frame.jpg"
            _write_first_frame_preview(upload_path, preview_path, cfg.processing_resolution)
            job["initialization_url"] = f"/api/jobs/{job_id}/initialization"
        self._save(job)
        if not click_assisted:
            threading.Thread(target=self._run, args=(job_id, upload_path), daemon=True).start()
        return dict(job)

    def list(self) -> list[dict[str, Any]]:
        """Return newest jobs first."""
        with self._lock:
            return sorted(
                (dict(job) for job in self._jobs.values()),
                key=lambda job: job["created_at"],
                reverse=True,
            )

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def result_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id / "results"

    def safe_artifact_path(self, job_id: str, relative_path: str) -> Path:
        root = self.result_dir(job_id).resolve()
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise FileNotFoundError(relative_path) from exc
        if not candidate.is_file():
            raise FileNotFoundError(relative_path)
        return candidate

    def safe_job_artifact_path(self, job_id: str, relative_path: str) -> Path:
        root = (self.jobs_dir / job_id).resolve()
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise FileNotFoundError(relative_path) from exc
        if not candidate.is_file():
            raise FileNotFoundError(relative_path)
        return candidate

    def get_initialization(self, job_id: str) -> dict[str, Any]:
        """Return the first-frame picker contract for a click-assisted job."""
        job = self.get(job_id)
        if not job:
            raise FileNotFoundError(f"Job not found / 未找到任务: {job_id}")
        if job.get("initialization_mode") != "manual_first_frame_click":
            raise ValueError("Job does not use click-assisted initialization / 任务未启用点击辅助初始化")
        return {
            "job_id": job_id,
            "status": job["status"],
            "status_display": job["status_display"],
            "n_animals": int(job["n_animals"]),
            "selected_frame": 0,
            "preview_url": f"/jobs/{job_id}/initialization/frame.jpg",
            "initialization_mode": "manual_first_frame_click",
            "initialization_mode_display": "首帧点击辅助 / first-frame click assist",
            "training_annotation_required": False,
        }

    def start_click_assisted(self, job_id: str, points: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate first-frame clicks, persist them, and schedule tracking."""
        job = self.get(job_id)
        if not job:
            raise FileNotFoundError(f"Job not found / 未找到任务: {job_id}")
        if job.get("status") != "awaiting_seed":
            raise ValueError("Job is not waiting for first-frame clicks / 任务当前不在等待首帧点击")
        job_dir = self.jobs_dir / job_id
        upload_path = next(job_dir.glob("input.*"), None)
        if upload_path is None:
            raise FileNotFoundError("Uploaded video not found / 未找到已上传视频")
        cfg = PipelineConfig.from_yaml(self.config_path)
        result = create_manual_initialization_points(
            video_path=upload_path,
            n_animals=int(job["n_animals"]),
            points=points,
            resize_height=cfg.processing_resolution,
            selected_frame=0,
        )
        files = write_auto_init_outputs(result, job_dir / "initialization")
        self._update(
            job_id,
            status="queued",
            status_display="已接收首帧点击，排队中 / first-frame clicks received, queued",
            initialization_points=len(result.payload["points"]),
            initialization_files=files,
        )
        threading.Thread(
            target=self._run,
            args=(job_id, upload_path, result.payload),
            daemon=True,
        ).start()
        return self.get(job_id) or {}

    def evaluate_review(
        self,
        job_id: str,
        source: BinaryIO,
        filename: str,
        confirm_reviewed: bool,
    ) -> dict[str, Any]:
        """Persist researcher-reviewed sparse GT and publish a guarded metric report."""
        job = self.get(job_id)
        if not job:
            raise FileNotFoundError(f"Job not found / 未找到任务: {job_id}")
        if job.get("status") != "completed":
            raise ValueError("Tracking job is not completed / 追踪任务尚未完成")
        if Path(filename).suffix.lower() != ".mot":
            raise ValueError("Reviewed GT must be a .mot file / 已审核 GT 必须是 .mot 文件")
        if not confirm_reviewed:
            raise ValueError(
                "Confirm that GT was manually reviewed before evaluation / "
                "计算指标前请确认 GT 已经过人工审核"
            )

        package = self.result_dir(job_id) / "validation_review"
        if not (package / "validation_manifest.json").exists():
            raise FileNotFoundError("Validation package not found / 未找到验证审核包")
        target = package / "reviewed_gt.mot"
        temporary = package / ".reviewed_gt.upload"
        backup = package / ".reviewed_gt.previous"
        with temporary.open("wb") as uploaded:
            shutil.copyfileobj(source, uploaded, length=1024 * 1024)
        if temporary.stat().st_size == 0:
            temporary.unlink()
            raise ValueError("Reviewed GT upload is empty / 已审核 GT 上传文件为空")

        if backup.exists():
            backup.unlink()
        if target.exists():
            target.replace(backup)
        temporary.replace(target)
        try:
            from densetrack.reporting.static_dashboard import build_static_dashboard
            from densetrack.validation.review_package import evaluate_validation_review_package

            report = evaluate_validation_review_package(
                package_dir=package,
                ground_truth=target,
                confirm_reviewed=True,
            )
            auto_init_dir = self.result_dir(job_id) / "auto_init"
            build_static_dashboard(
                run_dir=self.result_dir(job_id),
                auto_init_dir=auto_init_dir if auto_init_dir.exists() else None,
            )
        except Exception:
            target.unlink(missing_ok=True)
            if backup.exists():
                backup.replace(target)
            raise
        backup.unlink(missing_ok=True)

        report_url = f"/jobs/{job_id}/results/validation_review/evaluation_report.md"
        report["report_url"] = report_url
        report["dashboard_url"] = f"/jobs/{job_id}/results/dashboard.html"
        self._update(
            job_id,
            validation_status="reviewed",
            validation_status_display="已审核并评估 / reviewed and evaluated",
            validation_report_url=report_url,
            validation_dashboard_url=report["dashboard_url"],
            validation_metrics=report["metrics"],
        )
        return report

    def _run(
        self,
        job_id: str,
        upload_path: Path,
        initialization_override: dict[str, Any] | None = None,
    ) -> None:
        self._update(job_id, status="running", status_display="处理中 / running")
        job = self.get(job_id)
        if not job:
            return
        try:
            cfg = PipelineConfig.from_yaml(self.config_path)
            cfg.n_animals = int(job["n_animals"])
            tracking_profile = str(job.get("tracking_profile", "research_long_term"))
            if tracking_profile == "research_long_term":
                cfg.tracklet.backend = "mac_adaptive"
                cfg.tracklet.adaptive_discrete_geometry = True
                cfg.tracklet.adaptive_identity_smoothing = True
                cfg.global_optimizer.method = "long_term"
                cfg.global_optimizer.closed_set_stitching = True
            elif tracking_profile == "fast_baseline":
                cfg.tracklet.adaptive_discrete_geometry = False
                cfg.tracklet.adaptive_identity_smoothing = False
                cfg.global_optimizer.method = "greedy"
            result = run_complete_local(
                video_path=upload_path,
                pipeline_config=cfg,
                output_dir=self.result_dir(job_id),
                motif_clusters=int(job["clusters"]),
                initialization_override=initialization_override,
                experiment_template=str(job.get("experiment_template", "open_field")),
            )
            self._update(
                job_id,
                status="completed",
                status_display="已完成 / completed",
                result_url=f"/jobs/{job_id}/results/dashboard.html",
                manifest=result["manifest"],
            )
        except Exception as exc:  # pragma: no cover - exercised through HTTP smoke tests
            self._update(
                job_id,
                status="failed",
                status_display="失败 / failed",
                error=str(exc),
            )

    def _load_existing(self) -> None:
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if job.get("status") in {"queued", "running"}:
                job["status"] = "failed"
                job["status_display"] = "中断 / interrupted"
                job["error"] = "Server restarted before processing completed / 服务器重启导致任务中断"
            self._jobs[str(job["id"])] = job

    def _save(self, job: dict[str, Any]) -> None:
        with self._lock:
            self._jobs[str(job["id"])] = dict(job)
            target = self.jobs_dir / str(job["id"]) / "job.json"
            target.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = dict(self._jobs[job_id])
        job.update(changes)
        job["updated_at"] = datetime.now().isoformat()
        self._save(job)


def serve_webapp(
    workspace: str | Path = "results/web_workspace",
    config_path: str | Path = "configs/mice_black_topview_mac_adaptive.yaml",
    host: str = "127.0.0.1",
    port: int = 8765,
    max_upload_mb: int = 2048,
) -> None:
    """Start the local DenseTrack research workspace."""
    registry = JobRegistry(workspace=workspace, config_path=config_path)
    handler = _make_handler(registry, max_upload_bytes=max_upload_mb * 1024 * 1024)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"DenseTrack-AI Web 工作台 / workspace: http://{host}:{port}")
    print(f"结果目录 / Results workspace: {registry.workspace}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _make_handler(registry: JobRegistry, max_upload_bytes: int):
    class DenseTrackRequestHandler(BaseHTTPRequestHandler):
        server_version = "DenseTrackLocal/0.1"

        def do_GET(self) -> None:  # noqa: N802
            path = unquote(urlparse(self.path).path)
            if path == "/":
                self._send_bytes(render_index_html().encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/health":
                self._send_json({"status": "ok", "status_display": "可用 / healthy"})
                return
            if path == "/api/jobs":
                self._send_json({"jobs": registry.list()})
                return
            initialization_match = re.fullmatch(r"/api/jobs/([a-f0-9]{12})/initialization", path)
            if initialization_match:
                try:
                    payload = registry.get_initialization(initialization_match.group(1))
                except (FileNotFoundError, ValueError) as exc:
                    self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                    return
                self._send_json(payload)
                return
            match = re.fullmatch(r"/api/jobs/([a-f0-9]{12})", path)
            if match:
                job = registry.get(match.group(1))
                if not job:
                    self._send_error(HTTPStatus.NOT_FOUND, "Job not found / 未找到任务")
                    return
                self._send_json(job)
                return
            artifact_match = re.fullmatch(r"/jobs/([a-f0-9]{12})/results/(.+)", path)
            if artifact_match:
                try:
                    artifact = registry.safe_artifact_path(
                        artifact_match.group(1),
                        artifact_match.group(2),
                    )
                except FileNotFoundError:
                    self._send_error(HTTPStatus.NOT_FOUND, "Artifact not found / 未找到结果文件")
                    return
                self._send_file(artifact)
                return
            initialization_artifact_match = re.fullmatch(
                r"/jobs/([a-f0-9]{12})/initialization/(.+)",
                path,
            )
            if initialization_artifact_match:
                try:
                    artifact = registry.safe_job_artifact_path(
                        initialization_artifact_match.group(1),
                        f"initialization/{initialization_artifact_match.group(2)}",
                    )
                except FileNotFoundError:
                    self._send_error(HTTPStatus.NOT_FOUND, "Artifact not found / 未找到结果文件")
                    return
                self._send_file(artifact)
                return
            self._send_error(HTTPStatus.NOT_FOUND, "Not found / 未找到页面")

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            validation_match = re.fullmatch(r"/api/jobs/([a-f0-9]{12})/validation", path)
            if validation_match:
                self._handle_validation_upload(validation_match.group(1))
                return
            initialization_match = re.fullmatch(r"/api/jobs/([a-f0-9]{12})/initialization", path)
            if initialization_match:
                self._handle_click_assisted_start(initialization_match.group(1))
                return
            if path != "/api/jobs":
                self._send_error(HTTPStatus.NOT_FOUND, "Not found / 未找到页面")
                return
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0 or length > max_upload_bytes:
                self._send_error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "Upload is empty or too large / 上传文件为空或超过限制",
                )
                return
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self._send_error(
                    HTTPStatus.BAD_REQUEST,
                    "Expected multipart upload / 需要 multipart 上传",
                )
                return
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    import cgi

                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": content_type,
                        "CONTENT_LENGTH": str(length),
                    },
                )
                video = form["video"]
                filename = str(video.filename or "")
                if not filename or not getattr(video, "file", None):
                    raise ValueError("Video file is required / 请选择视频文件")
                n_animals = _bounded_int(form.getfirst("n_animals", "4"), 1, 50, "n_animals")
                clusters = _bounded_int(form.getfirst("clusters", "6"), 1, 30, "clusters")
                initialization_mode = form.getfirst("initialization_mode", "automatic")
                if initialization_mode not in {"automatic", "manual_first_frame_click"}:
                    raise ValueError("Unsupported initialization mode / 不支持的初始化模式")
                experiment_template = form.getfirst("experiment_template", "open_field")
                tracking_profile = form.getfirst(
                    "tracking_profile",
                    "research_long_term",
                )
                job = registry.submit(
                    video.file,
                    filename,
                    n_animals,
                    clusters,
                    initialization_mode=initialization_mode,
                    experiment_template=experiment_template,
                    tracking_profile=tracking_profile,
                )
            except (KeyError, ValueError) as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_json(job, status=HTTPStatus.ACCEPTED)

        def _handle_click_assisted_start(self, job_id: str) -> None:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0 or length > 1024 * 1024:
                self._send_error(HTTPStatus.BAD_REQUEST, "Invalid click payload / 点击数据无效")
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                points = payload.get("points", [])
                if not isinstance(points, list):
                    raise ValueError("Points must be a list / 点击点必须是列表")
                job = registry.start_click_assisted(job_id, points)
            except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_json(job, status=HTTPStatus.ACCEPTED)

        def _handle_validation_upload(self, job_id: str) -> None:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0 or length > max_upload_bytes:
                self._send_error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    "Upload is empty or too large / 上传文件为空或超过限制",
                )
                return
            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                self._send_error(HTTPStatus.BAD_REQUEST, "Expected multipart upload / 需要 multipart 上传")
                return
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    import cgi

                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": content_type,
                        "CONTENT_LENGTH": str(length),
                    },
                )
                reviewed_gt = form["reviewed_gt"]
                filename = str(reviewed_gt.filename or "")
                if not filename or not getattr(reviewed_gt, "file", None):
                    raise ValueError("Reviewed GT file is required / 请选择已审核 GT 文件")
                confirm_reviewed = form.getfirst("confirm_reviewed", "") == "true"
                report = registry.evaluate_review(
                    job_id=job_id,
                    source=reviewed_gt.file,
                    filename=filename,
                    confirm_reviewed=confirm_reviewed,
                )
            except (KeyError, FileNotFoundError, ValueError) as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_json(report)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[web] {self.address_string()} - {fmt % args}")

        def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send_bytes(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def _send_error(self, status: HTTPStatus, message: str) -> None:
            self._send_json({"error": message}, status=status)

        def _send_bytes(
            self,
            payload: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _send_file(self, path: Path) -> None:
            size = path.stat().st_size
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            start, end = _parse_range(self.headers.get("Range"), size)
            status = HTTPStatus.PARTIAL_CONTENT if start != 0 or end != size - 1 else HTTPStatus.OK
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as source:
                source.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    remaining -= len(chunk)

    return DenseTrackRequestHandler


def _bounded_int(value: str, minimum: int, maximum: int, name: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer / {name} 必须是整数") from exc
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum} / {name} 超出范围")
    return number


def _parse_range(header: str | None, size: int) -> tuple[int, int]:
    if not header:
        return 0, max(0, size - 1)
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not match:
        return 0, max(0, size - 1)
    start = int(match.group(1) or 0)
    end = int(match.group(2) or size - 1)
    return min(start, size - 1), min(max(start, end), size - 1)


def _write_first_frame_preview(video_path: Path, output_path: Path, resize_height: int = 0) -> None:
    """Write the first frame used by the browser click picker."""
    capture = cv2.VideoCapture(str(video_path))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise ValueError("Could not read the first video frame / 无法读取视频第一帧")
    if resize_height > 0 and frame.shape[0] != resize_height:
        scale = resize_height / frame.shape[0]
        frame = cv2.resize(
            frame,
            (int(round(frame.shape[1] * scale)), resize_height),
            interpolation=cv2.INTER_AREA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame):
        raise OSError(f"Could not write initialization preview / 无法写入初始化预览: {output_path}")


def render_index_html() -> str:
    """Return the local workspace frontend."""
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DenseTrack-AI 本地研究工作台 / Local Research Workspace</title>
  <style>
    :root {
      --ink: #182126; --muted: #62727a; --line: #d9e1e3; --panel: #ffffff;
      --canvas: #f4f7f6; --teal: #167c80; --teal-dark: #0f666a; --green: #2f855a;
      --amber: #b76a12; --coral: #bc4e43;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; max-width: 100%; overflow-x: hidden; }
    body { background: var(--canvas); color: var(--ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing: 0; }
    header { background: #18343a; color: white; border-bottom: 4px solid #e4a33d; padding: 22px 28px 20px; }
    header h1 { margin: 0 0 4px; font-size: 25px; overflow-wrap: anywhere; }
    header p { margin: 0; color: #b8d1d2; font-size: 13px; }
    main { width: 100%; max-width: 1180px; margin: 0 auto; padding: 22px; }
    section { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; margin-bottom: 16px; min-width: 0; }
    .upload { display: grid; grid-template-columns: 1fr 1.1fr; gap: 26px; padding: 22px; }
    h2 { margin: 0 0 8px; font-size: 19px; }
    h3 { margin: 0; font-size: 14px; }
    p { color: var(--muted); font-size: 13px; line-height: 1.6; }
    .steps { margin: 18px 0 0; padding: 0; list-style: none; display: grid; gap: 8px; color: #38545a; font-size: 13px; }
    .steps strong { display: inline-flex; align-items: center; justify-content: center; width: 22px; height: 22px; margin-right: 8px; border-radius: 50%; background: #e4f0f0; color: var(--teal-dark); font-size: 11px; }
    form { display: grid; gap: 14px; }
    label { display: grid; gap: 6px; color: #42555b; font-size: 12px; font-weight: 700; }
    input { width: 100%; border: 1px solid #bfd0d3; border-radius: 4px; background: white; color: var(--ink); font: inherit; padding: 10px; }
    select { width: 100%; border: 1px solid #bfd0d3; border-radius: 4px; background: white; color: var(--ink); font: inherit; padding: 10px; }
    input[type=file] { padding: 8px; font-size: 12px; }
    .field-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    button, .result-link { border: 0; border-radius: 4px; background: var(--teal); color: white; cursor: pointer; display: inline-flex; align-items: center; justify-content: center; font: inherit; font-size: 13px; font-weight: 750; padding: 10px 13px; text-decoration: none; transition: background .16s ease; }
    button:hover, .result-link:hover { background: var(--teal-dark); }
    button:disabled { cursor: wait; opacity: .65; }
    .jobs { padding: 0; overflow: hidden; }
    .section-heading { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 16px 18px; border-bottom: 1px solid var(--line); }
    .section-heading p { margin: 3px 0 0; }
    .secondary { background: transparent; border: 1px solid #bdd0d2; color: var(--teal-dark); padding: 7px 10px; }
    .secondary:hover { background: #e8f2f1; }
    .job-list { display: grid; }
    .job-row { display: grid; grid-template-columns: minmax(170px, 1.4fr) 110px 120px minmax(170px, 1fr); gap: 12px; align-items: center; padding: 14px 18px; border-bottom: 1px solid var(--line); }
    .job-row:last-child { border-bottom: 0; }
    .job-file { min-width: 0; }
    .job-file strong, .job-file span { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .job-file span, .job-meta { color: var(--muted); font-size: 12px; }
    .status { color: var(--amber); font-size: 12px; font-weight: 750; }
    .status.completed { color: var(--green); }
    .status.failed { color: var(--coral); }
    .result-link { justify-self: end; padding: 8px 10px; font-size: 12px; }
    .empty { margin: 0; padding: 24px 18px; }
    .form-status { min-height: 20px; margin: 0; }
    .mode-label { gap: 8px; }
    .segmented { display: grid; grid-template-columns: 1fr 1fr; border: 1px solid #bdd0d2; border-radius: 4px; overflow: hidden; }
    .segmented label { cursor: pointer; display: flex; align-items: center; gap: 7px; padding: 10px; font-weight: 700; }
    .segmented label + label { border-left: 1px solid #bdd0d2; }
    .segmented input { accent-color: var(--teal); width: auto; }
    .validation-actions { grid-column: 1 / -1; border-top: 1px dashed var(--line); display: grid; gap: 8px; margin-top: 4px; padding-top: 12px; }
    .validation-actions p { margin: 0; }
    .validation-links { display: flex; flex-wrap: wrap; gap: 7px; }
    .validation-links a { color: var(--teal-dark); font-size: 12px; font-weight: 700; text-decoration: none; }
    .validation-form { display: grid; grid-template-columns: minmax(180px, 1fr) auto auto; align-items: end; gap: 9px; }
    .validation-confirm { align-items: center; display: flex; gap: 6px; white-space: nowrap; }
    .validation-confirm input { width: auto; }
    .validation-status { color: var(--green); font-size: 12px; min-height: 18px; }
    dialog { border: 0; border-radius: 6px; box-shadow: 0 18px 55px rgba(24, 52, 58, .28); max-height: calc(100vh - 28px); max-width: min(920px, calc(100vw - 28px)); overflow: hidden; padding: 0; width: 920px; }
    dialog::backdrop { background: rgba(17, 32, 36, .62); }
    .seed-panel { background: white; display: grid; gap: 12px; max-height: calc(100vh - 28px); padding: 16px; }
    .seed-heading { align-items: start; display: flex; justify-content: space-between; gap: 12px; }
    .seed-heading h2, .seed-heading p { margin: 0; }
    .icon-button { background: transparent; border: 1px solid var(--line); color: var(--ink); font-size: 21px; height: 34px; padding: 0; width: 34px; }
    .icon-button:hover { background: #edf4f4; }
    .seed-canvas-wrap { background: #101718; border-radius: 4px; display: flex; justify-content: center; line-height: 0; min-height: 0; overflow: auto; }
    #seed-canvas { cursor: crosshair; display: block; height: auto; max-height: min(68vh, 720px); max-width: 100%; width: auto; }
    .seed-actions { display: flex; flex-wrap: wrap; gap: 8px; }
    .seed-actions .secondary { background: white; }
    @media (max-width: 760px) {
      header { padding: 18px 20px 16px; }
      main { padding: 14px; }
      .upload { grid-template-columns: 1fr; gap: 18px; padding: 17px; }
      .job-row { grid-template-columns: 1fr auto; gap: 7px 10px; }
      .job-meta { grid-column: 1; }
      .result-link { grid-column: 2; grid-row: 1 / span 2; }
      .validation-form { grid-template-columns: 1fr; align-items: stretch; }
      .validation-confirm { white-space: normal; }
      .segmented { grid-template-columns: 1fr; }
      .segmented label + label { border-left: 0; border-top: 1px solid #bdd0d2; }
    }
  </style>
</head>
<body>
  <header>
    <h1>DenseTrack-AI 本地研究工作台 / Local Research Workspace</h1>
    <p>长时身份连续性、多动物追踪与科研审计 / Long-term identity continuity, multi-animal tracking, and research audit</p>
  </header>
  <main>
    <section class="upload">
      <div>
        <h2>提交视频 / Submit video</h2>
        <p>选择本地视频后，系统会自动选取初始化种子点，以分层离散几何、双向全视频平滑和闭集轨迹拼接保持长时身份。</p>
        <p>After upload, the workspace uses hierarchical discrete geometry, bidirectional full-video smoothing, and closed-set stitching to preserve long-term identity.</p>
        <ol class="steps">
          <li><strong>01</strong>自动初始化 / Auto initialization</li>
          <li><strong>02</strong>长时身份关联 / Long-term identity association</li>
          <li><strong>03</strong>HOTA / AssA / ID switch 科研审计</li>
        </ol>
      </div>
      <form id="job-form">
        <label>视频文件 / Video file
          <input data-testid="video-input" name="video" type="file" accept="video/*" required>
        </label>
        <div class="field-grid">
          <label>动物数量 / Animals
            <input data-testid="animal-count" name="n_animals" type="number" value="4" min="1" max="50" required>
          </label>
          <label>行为模体数 / Motifs
            <input data-testid="cluster-count" name="clusters" type="number" value="6" min="1" max="30" required>
          </label>
        </div>
        <label>追踪配置 / Tracking profile
          <select data-testid="tracking-profile" name="tracking_profile">
            <option value="research_long_term" selected>科研长时身份（推荐） / Research long-term identity (recommended)</option>
            <option value="fast_baseline">快速基线（对照） / Fast baseline (control)</option>
          </select>
        </label>
        <label>实验范式 / Experiment template
          <select data-testid="experiment-template" name="experiment_template">
            <option value="open_field">旷场实验 / Open field</option>
            <option value="morris_water_maze">Morris 水迷宫 / Morris water maze</option>
            <option value="three_chamber_social">三腔社交实验 / Three-chamber social test</option>
            <option value="light_dark_box">明暗箱实验 / Light-dark box</option>
            <option value="elevated_plus_maze">高架十字迷宫 / Elevated plus maze</option>
            <option value="cpp">条件性位置偏爱 / Conditioned place preference</option>
          </select>
        </label>
        <div class="mode-label">
          <span>初始化方式 / Initialization mode</span>
          <div class="segmented">
            <label><input name="initialization_mode" type="radio" value="automatic" checked>零点击自动 / Zero-click auto</label>
            <label><input name="initialization_mode" type="radio" value="manual_first_frame_click">首帧点击辅助 / First-frame click assist</label>
          </div>
        </div>
        <button data-testid="submit-job" type="submit">开始处理 / Start processing</button>
        <p class="form-status" id="form-status" role="status"></p>
      </form>
    </section>
    <section class="jobs">
      <div class="section-heading">
        <div>
          <h2>处理任务 / Processing jobs</h2>
          <p>完成后可打开双语研究面板并下载结果 / Open the bilingual dashboard and download artifacts when complete.</p>
        </div>
        <button class="secondary" id="refresh-jobs" type="button">刷新 / Refresh</button>
      </div>
      <div class="job-list" id="job-list"><p class="empty">正在读取任务 / Loading jobs...</p></div>
    </section>
  </main>
  <dialog id="seed-dialog">
    <div class="seed-panel">
      <div class="seed-heading">
        <div>
          <h2>首帧个体选择 / First-frame individual selection</h2>
          <p id="seed-progress">依次点击动物 / Click animals in ID order</p>
        </div>
        <button class="icon-button" id="close-seed-dialog" type="button" title="关闭 / Close" aria-label="关闭 / Close">×</button>
      </div>
      <div class="seed-canvas-wrap"><canvas id="seed-canvas"></canvas></div>
      <div class="seed-actions">
        <button class="secondary" id="undo-seed" type="button">撤销 / Undo</button>
        <button class="secondary" id="clear-seeds" type="button">清空 / Clear</button>
        <button id="submit-seeds" type="button">确认并开始追踪 / Confirm and start tracking</button>
      </div>
      <p class="form-status" id="seed-status" role="status"></p>
    </div>
  </dialog>
  <script>
    const form = document.querySelector("#job-form");
    const formStatus = document.querySelector("#form-status");
    const jobList = document.querySelector("#job-list");
    const submitButton = document.querySelector("[data-testid=submit-job]");
    const seedDialog = document.querySelector("#seed-dialog");
    const seedCanvas = document.querySelector("#seed-canvas");
    const seedContext = seedCanvas.getContext("2d");
    const seedProgress = document.querySelector("#seed-progress");
    const seedStatus = document.querySelector("#seed-status");
    const seedSubmit = document.querySelector("#submit-seeds");
    const seedState = {jobId:null, required:0, points:[], image:new Image()};
    const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
    const renderJob = job => {
      const link = job.result_url ? `<a class="result-link" href="${escapeHtml(job.result_url)}">打开结果 / Open result</a>` : "";
      const initializationAction = job.status === "awaiting_seed" ? `<button class="result-link" data-seed-job="${escapeHtml(job.id)}" type="button">指定首帧个体 / Select first-frame animals</button>` : "";
      const error = job.error ? `<span>${escapeHtml(job.error)}</span>` : "";
      const reportLink = job.validation_report_url ? `<a href="${escapeHtml(job.validation_report_url)}">查看稀疏指标 / View sparse metrics</a>` : "";
      const validation = job.status === "completed" ? `<div class="validation-actions">
        <p><strong>稀疏科学验证 / Sparse scientific validation</strong></p>
        <div class="validation-links">
          <a href="/jobs/${escapeHtml(job.id)}/results/validation_review/review.html">打开审核器 / Open reviewer</a>
          ${reportLink}
        </div>
        <form class="validation-form" data-job-id="${escapeHtml(job.id)}">
          <label>已审核 GT / Reviewed GT
            <input name="reviewed_gt" type="file" accept=".mot,text/plain" required>
          </label>
          <label class="validation-confirm"><input name="confirm_reviewed" type="checkbox" value="true" required>确认已经人工审核 / Confirm manually reviewed</label>
          <button type="submit">上传并计算指标 / Upload and evaluate</button>
        </form>
        <div class="validation-status" role="status">${escapeHtml(job.validation_status_display || "")}</div>
      </div>` : "";
      return `<article class="job-row">
        <div class="job-file"><strong>${escapeHtml(job.filename)}</strong><span>${escapeHtml(job.id)} · ${escapeHtml(job.created_at)}</span>${error}</div>
        <div class="job-meta">${escapeHtml(job.n_animals)} 只动物 / animals<br>${escapeHtml(job.tracking_profile_display || "科研长时身份 / Research long-term identity")}<br>${escapeHtml(job.experiment_template_display || "旷场实验 / Open field")}</div>
        <div class="status ${escapeHtml(job.status)}">${escapeHtml(job.status_display)}</div>
        ${link}
        ${initializationAction}
        ${validation}
      </article>`;
    };
    async function refreshJobs() {
      try {
        const response = await fetch("/api/jobs", {cache: "no-store"});
        const payload = await response.json();
        jobList.innerHTML = payload.jobs.length ? payload.jobs.map(renderJob).join("") : `<p class="empty">尚无任务 / No jobs yet</p>`;
      } catch (error) {
        jobList.innerHTML = `<p class="empty">暂时无法连接服务，请稍后刷新 / Service temporarily unavailable. Refresh shortly.</p>`;
      }
    }
    form.addEventListener("submit", async event => {
      event.preventDefault();
      submitButton.disabled = true;
      formStatus.textContent = "正在上传 / Uploading...";
      try {
        const response = await fetch("/api/jobs", {method: "POST", body: new FormData(form)});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Upload failed / 上传失败");
        formStatus.textContent = `已提交任务 / Submitted: ${payload.id}`;
        form.reset();
        await refreshJobs();
      } catch (error) {
        formStatus.textContent = error.message;
      } finally {
        submitButton.disabled = false;
      }
    });
    document.querySelector("#refresh-jobs").addEventListener("click", refreshJobs);
    function drawSeeds() {
      if (!seedState.image.naturalWidth) return;
      seedCanvas.width = seedState.image.naturalWidth;
      seedCanvas.height = seedState.image.naturalHeight;
      seedContext.drawImage(seedState.image, 0, 0);
      seedState.points.forEach((point, index) => {
        seedContext.beginPath();
        seedContext.arc(point.x, point.y, 10, 0, Math.PI * 2);
        seedContext.fillStyle = ["#f59e0b","#22c55e","#ef4444","#a855f7","#06b6d4","#f97316"][index % 6];
        seedContext.fill();
        seedContext.lineWidth = 3;
        seedContext.strokeStyle = "#ffffff";
        seedContext.stroke();
        seedContext.fillStyle = "#ffffff";
        seedContext.font = "700 17px system-ui";
        seedContext.fillText(`ID ${index}`, point.x + 14, Math.max(22, point.y - 12));
      });
      seedProgress.textContent = `依次点击动物 / Click animals in ID order: ${seedState.points.length} / ${seedState.required}`;
      seedSubmit.disabled = seedState.points.length !== seedState.required;
    }
    async function openSeedPicker(jobId) {
      seedStatus.textContent = "正在读取首帧 / Loading first frame...";
      const response = await fetch(`/api/jobs/${jobId}/initialization`, {cache:"no-store"});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Initialization failed / 初始化失败");
      seedState.jobId = jobId;
      seedState.required = payload.n_animals;
      seedState.points = [];
      seedState.image.onload = () => { seedStatus.textContent = ""; drawSeeds(); };
      seedState.image.src = `${payload.preview_url}?t=${Date.now()}`;
      seedDialog.showModal();
    }
    jobList.addEventListener("click", async event => {
      const button = event.target.closest("[data-seed-job]");
      if (!button) return;
      try { await openSeedPicker(button.dataset.seedJob); }
      catch (error) { formStatus.textContent = error.message; }
    });
    seedCanvas.addEventListener("click", event => {
      if (seedState.points.length >= seedState.required) return;
      const rect = seedCanvas.getBoundingClientRect();
      seedState.points.push({
        x:(event.clientX - rect.left) * seedCanvas.width / rect.width,
        y:(event.clientY - rect.top) * seedCanvas.height / rect.height,
      });
      drawSeeds();
    });
    document.querySelector("#undo-seed").addEventListener("click", () => { seedState.points.pop(); drawSeeds(); });
    document.querySelector("#clear-seeds").addEventListener("click", () => { seedState.points = []; drawSeeds(); });
    document.querySelector("#close-seed-dialog").addEventListener("click", () => seedDialog.close());
    seedSubmit.addEventListener("click", async () => {
      seedSubmit.disabled = true;
      seedStatus.textContent = "正在提交首帧点击 / Submitting first-frame clicks...";
      try {
        const response = await fetch(`/api/jobs/${seedState.jobId}/initialization`, {
          method:"POST",
          headers:{"Content-Type":"application/json"},
          body:JSON.stringify({points:seedState.points}),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Initialization failed / 初始化失败");
        seedDialog.close();
        formStatus.textContent = `已提交首帧点击 / First-frame clicks submitted: ${payload.id}`;
        await refreshJobs();
      } catch (error) {
        seedStatus.textContent = error.message;
      } finally {
        seedSubmit.disabled = seedState.points.length !== seedState.required;
      }
    });
    jobList.addEventListener("submit", async event => {
      const validationForm = event.target.closest(".validation-form");
      if (!validationForm) return;
      event.preventDefault();
      const status = validationForm.parentElement.querySelector(".validation-status");
      const button = validationForm.querySelector("button");
      button.disabled = true;
      status.textContent = "正在上传并计算 / Uploading and evaluating...";
      try {
        const response = await fetch(`/api/jobs/${validationForm.dataset.jobId}/validation`, {
          method: "POST",
          body: new FormData(validationForm),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Evaluation failed / 评估失败");
        status.textContent = "评估完成 / Evaluation complete";
        await refreshJobs();
      } catch (error) {
        status.textContent = error.message;
      } finally {
        button.disabled = false;
      }
    });
    refreshJobs();
    setInterval(refreshJobs, 3000);
  </script>
</body>
</html>
"""
