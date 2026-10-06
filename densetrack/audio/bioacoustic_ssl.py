"""Species-specific bioacoustic SSL features for animal communication.

This module is inspired by Dolph2Vec: use unlabeled, species-specific
vocalizations to learn reusable acoustic units. It is not a Wav2Vec2 training
implementation. It provides a deterministic codebook and whistle-candidate
detector for WAV recordings so communication behavior can be audited alongside
markerless tracking outputs.
"""

from __future__ import annotations

import csv
import json
import wave
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["BioacousticSSLAnalyzer", "BioacousticSSLConfig"]


@dataclass
class BioacousticSSLConfig:
    """Controls unlabeled audio representation export."""

    species: str = "dolphin"
    window_ms: float = 120.0
    hop_ms: float = 40.0
    n_bands: int = 16
    codebook_size: int = 12
    whistle_min_frequency_hz: float = 1000.0


class BioacousticSSLAnalyzer:
    """Build Dolph2Vec-inspired acoustic units from unlabeled WAV audio."""

    def __init__(self, config: BioacousticSSLConfig | None = None):
        self.config = config or BioacousticSSLConfig()

    def analyze_wav(
        self,
        audio_path: str | Path,
        output_json: str | Path | None = None,
        output_csv: str | Path | None = None,
    ) -> dict[str, Any]:
        path = Path(audio_path).expanduser().resolve()
        samples, sample_rate = _read_wav(path)
        if samples.size == 0:
            report = self._empty_report(path)
            self._write_outputs(report, [], output_json, output_csv)
            return report

        rows, feature_names = self._window_features(samples, sample_rate)
        if not rows:
            report = self._empty_report(path)
            self._write_outputs(report, [], output_json, output_csv)
            return report

        feature_matrix = np.asarray(
            [[float(row[name]) for name in feature_names] for row in rows],
            dtype=np.float64,
        )
        standardized, means, scales = _standardize(feature_matrix)
        labels, centers = _kmeans(standardized, self.config.codebook_size)
        embedding, explained = _pca(standardized, dim=2)
        for idx, row in enumerate(rows):
            row["codebook_unit"] = int(labels[idx])
            row["embedding_x"] = float(embedding[idx, 0])
            row["embedding_y"] = float(embedding[idx, 1])
            row["whistle_candidate"] = bool(
                row["peak_frequency_hz"] >= self.config.whistle_min_frequency_hz
                and row["spectral_entropy"] <= 0.62
                and row["spectral_bandwidth_hz"] <= max(sample_rate * 0.20, 1.0)
            )

        unit_summary = []
        for unit in sorted(set(int(label) for label in labels)):
            members = [row for row in rows if row["codebook_unit"] == unit]
            unit_summary.append(
                {
                    "unit": unit,
                    "count": len(members),
                    "ratio": len(members) / max(len(rows), 1),
                    "mean_peak_frequency_hz": float(np.mean([m["peak_frequency_hz"] for m in members])),
                    "whistle_candidate_ratio": float(np.mean([m["whistle_candidate"] for m in members])),
                }
            )

        whistle_rows = [row for row in rows if row["whistle_candidate"]]
        report = {
            "created_at": datetime.now().isoformat(),
            "format": "densetrack_bioacoustic_ssl_report",
            "format_version": "1.0",
            "source_audio": str(path),
            "annotation_required": False,
            "paper_signal": {
                "name": "Dolph2Vec",
                "arxiv": "2606.12503",
                "used_as": (
                    "Design signal for species-specific SSL acoustic units; this module "
                    "does not train Wav2Vec2/Dolph2Vec weights."
                ),
            },
            "config": asdict(self.config),
            "summary": {
                "species": self.config.species,
                "sample_rate_hz": sample_rate,
                "duration_s": len(samples) / max(sample_rate, 1),
                "num_windows": len(rows),
                "codebook_units": len(unit_summary),
                "whistle_candidate_windows": len(whistle_rows),
                "whistle_candidate_ratio": len(whistle_rows) / max(len(rows), 1),
            },
            "features": {
                "names": feature_names,
                "means": means.tolist(),
                "scales": scales.tolist(),
                "pca_explained_variance_ratio": explained.tolist(),
            },
            "codebook": unit_summary,
            "limitations": [
                "This is a lightweight species-specific SSL surrogate, not the Dolph2Vec neural model.",
                "Whistle candidates are unsupervised acoustic events and require domain review before scientific claims.",
            ],
        }
        self._write_outputs(report, rows, output_json, output_csv)
        return report

    def _window_features(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        window = max(16, int(round(sample_rate * self.config.window_ms / 1000.0)))
        hop = max(1, int(round(sample_rate * self.config.hop_ms / 1000.0)))
        if len(samples) < window:
            return [], []
        band_edges = np.linspace(0, sample_rate / 2.0, self.config.n_bands + 1)
        feature_names = [
            "log_energy",
            "spectral_centroid_hz",
            "spectral_bandwidth_hz",
            "peak_frequency_hz",
            "spectral_entropy",
        ] + [f"band_{idx}_log_energy" for idx in range(self.config.n_bands)]

        rows = []
        hann = np.hanning(window)
        freqs = np.fft.rfftfreq(window, d=1.0 / sample_rate)
        previous_centroid = 0.0
        for start in range(0, len(samples) - window + 1, hop):
            segment = samples[start : start + window] * hann
            power = np.abs(np.fft.rfft(segment)) ** 2
            total_power = float(np.sum(power) + 1e-12)
            probabilities = power / total_power
            centroid = float(np.sum(freqs * probabilities))
            bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * probabilities)))
            peak_frequency = float(freqs[int(np.argmax(power))])
            entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)) / np.log(len(probabilities)))
            band_values = []
            for lower, upper in zip(band_edges[:-1], band_edges[1:]):
                mask = (freqs >= lower) & (freqs < upper)
                band_values.append(float(np.log1p(np.sum(power[mask]))))
            row = {
                "start_s": start / sample_rate,
                "end_s": (start + window) / sample_rate,
                "log_energy": float(np.log1p(total_power)),
                "spectral_centroid_hz": centroid,
                "spectral_bandwidth_hz": bandwidth,
                "peak_frequency_hz": peak_frequency,
                "spectral_entropy": entropy,
                "centroid_delta_hz": centroid - previous_centroid,
            }
            previous_centroid = centroid
            for idx, value in enumerate(band_values):
                row[f"band_{idx}_log_energy"] = value
            rows.append(row)
        feature_names.append("centroid_delta_hz")
        return rows, feature_names

    @staticmethod
    def _write_outputs(
        report: dict[str, Any],
        rows: list[dict[str, Any]],
        output_json: str | Path | None,
        output_csv: str | Path | None,
    ) -> None:
        if output_json:
            target = Path(output_json)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if output_csv:
            target = Path(output_csv)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not rows:
                target.write_text("", encoding="utf-8")
                return
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    def _empty_report(self, path: Path) -> dict[str, Any]:
        return {
            "created_at": datetime.now().isoformat(),
            "format": "densetrack_bioacoustic_ssl_report",
            "format_version": "1.0",
            "source_audio": str(path),
            "annotation_required": False,
            "config": asdict(self.config),
            "summary": {
                "species": self.config.species,
                "num_windows": 0,
                "codebook_units": 0,
                "whistle_candidate_windows": 0,
            },
        }


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if sample_width == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width / 不支持的 WAV 采样宽度: {sample_width}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float64), sample_rate


def _standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(features, axis=0)
    scales = np.std(features, axis=0)
    scales[scales < 1e-6] = 1.0
    return np.clip((features - means) / scales, -8.0, 8.0), means, scales


def _pca(features: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    centered = features - np.mean(features, axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    output_dim = max(1, min(dim, vt.shape[0]))
    embedding = centered @ vt[:output_dim].T
    if output_dim < dim:
        embedding = np.column_stack([embedding, np.zeros((len(features), dim - output_dim))])
    variance = singular**2
    total = max(float(np.sum(variance)), 1e-12)
    explained = np.zeros(dim, dtype=float)
    explained[:output_dim] = variance[:output_dim] / total
    return embedding.astype(float), explained


def _kmeans(features: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n_clusters = max(1, min(k, len(features)))
    centers = [features[int(np.argmax(np.linalg.norm(features, axis=1)))]]
    while len(centers) < n_clusters:
        distances = np.min(
            np.stack([np.sum((features - center) ** 2, axis=1) for center in centers]),
            axis=0,
        )
        centers.append(features[int(np.argmax(distances))])
    center_matrix = np.asarray(centers, dtype=float)
    labels = np.zeros(len(features), dtype=int)
    for _ in range(60):
        distances = np.stack(
            [np.sum((features - center) ** 2, axis=1) for center in center_matrix],
            axis=1,
        )
        new_labels = np.argmin(distances, axis=1)
        new_centers = center_matrix.copy()
        for cluster_id in range(n_clusters):
            members = features[new_labels == cluster_id]
            if len(members):
                new_centers[cluster_id] = np.mean(members, axis=0)
        if np.array_equal(labels, new_labels) and np.allclose(center_matrix, new_centers):
            break
        labels = new_labels
        center_matrix = new_centers
    return labels.astype(int), center_matrix
