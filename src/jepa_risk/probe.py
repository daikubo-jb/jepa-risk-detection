from __future__ import annotations

from dataclasses import dataclass
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .cache import CacheError, read_feature_cache
from .evaluation import classification_metrics, frame_rows_for_records, select_f1_threshold


class ProbeError(ValueError):
    """Raised when probe training or its feature contract is invalid."""


@dataclass
class ProbeModel:
    scaler: StandardScaler
    classifier: LogisticRegression
    selected_c: float
    threshold: float | None
    threshold_f1: float | None
    history: list[dict[str, Any]]
    cache_key: str | None = None


def _as_feature_matrix(features: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(features)
    if matrix.ndim != 2 or matrix.shape[1] != 1024:
        raise ProbeError(f"{name} must have shape [N,1024], got {matrix.shape}")
    if matrix.shape[0] == 0 or not np.isfinite(matrix).all():
        raise ProbeError(f"{name} is empty or contains non-finite values")
    return matrix.astype(np.float32, copy=False)


def _fit_candidate(
    train_features: np.ndarray,
    train_labels: Sequence[int],
    c_value: float,
    *,
    max_iter: int = 2000,
) -> tuple[StandardScaler, LogisticRegression, list[str]]:
    if c_value <= 0:
        raise ProbeError("C must be positive")
    x_train = _as_feature_matrix(train_features, "train_features")
    y_train = np.asarray(train_labels, dtype=np.int64)
    if len(y_train) != len(x_train):
        raise ProbeError("train labels and features have different row counts")
    if len(np.unique(y_train)) < 2:
        raise ProbeError("train data must contain both classes")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    classifier = LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        max_iter=max_iter,
        solver="lbfgs",
        random_state=42,
    )
    convergence_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        classifier.fit(x_scaled, y_train)
    for warning in caught:
        if issubclass(warning.category, ConvergenceWarning):
            convergence_warnings.append(str(warning.message))
    return scaler, classifier, convergence_warnings


def predict_probe_scores(
    scaler: StandardScaler,
    classifier: LogisticRegression,
    features: np.ndarray,
) -> np.ndarray:
    x = _as_feature_matrix(features, "features")
    scaled = scaler.transform(x)
    classes = list(classifier.classes_)
    if 1 not in classes:
        raise ProbeError("classifier has no positive class")
    return classifier.predict_proba(scaled)[:, classes.index(1)].astype(np.float64, copy=False)


def _validation_frame_rows(
    validation_records: Iterable[dict[str, Any]],
    validation_windows: Sequence[dict[str, Any]],
    scores: np.ndarray,
    *,
    stride_sec: float,
) -> list[dict[str, Any]]:
    if len(validation_windows) != len(scores):
        raise ProbeError("validation windows and score rows have different lengths")
    by_video: dict[str, list[dict[str, Any]]] = {}
    for window, score in zip(validation_windows, scores):
        by_video.setdefault(str(window["video_id"]), []).append(
            {
                "window_id": window["window_id"],
                "video_id": window["video_id"],
                "available_at_s": window["available_at_s"],
                "score": float(score),
                "valid": True,
            }
        )
    return frame_rows_for_records(validation_records, by_video, stride_sec=stride_sec)


def train_probe_model(
    train_features: np.ndarray,
    train_labels: Sequence[int],
    validation_features: np.ndarray,
    validation_windows: Sequence[dict[str, Any]],
    validation_records: Iterable[dict[str, Any]],
    *,
    c_candidates: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    stride_sec: float = 0.5,
    cache_key: str | None = None,
) -> ProbeModel:
    """Fit train-only probes and select C by validation frame-level AP.

    The scaler and classifier are fit only on ``train_features``. Validation
    labels are used solely for C and threshold selection; train and validation
    are never recombined.
    """

    x_train = _as_feature_matrix(train_features, "train_features")
    x_validation = _as_feature_matrix(validation_features, "validation_features")
    candidates = sorted({float(value) for value in c_candidates})
    if not candidates or any(value <= 0 for value in candidates):
        raise ProbeError("c_candidates must contain positive values")
    validation_records_list = list(validation_records)
    history: list[dict[str, Any]] = []
    fitted: dict[float, tuple[StandardScaler, LogisticRegression, np.ndarray, list[str]]] = {}
    for c_value in candidates:
        scaler, classifier, convergence = _fit_candidate(x_train, train_labels, c_value)
        scores = predict_probe_scores(scaler, classifier, x_validation)
        frame_rows = _validation_frame_rows(
            validation_records_list,
            validation_windows,
            scores,
            stride_sec=stride_sec,
        )
        metrics = classification_metrics(frame_rows)
        average_precision = metrics.get("average_precision")
        history.append(
            {
                "C": c_value,
                "average_precision": average_precision,
                "valid_frames": metrics["valid_rows"],
                "convergence_warnings": convergence,
            }
        )
        fitted[c_value] = (scaler, classifier, frame_rows, convergence)
    usable = [row for row in history if row["average_precision"] is not None]
    if not usable:
        raise ProbeError("validation frame AP is unavailable (empty or single-class validation)")
    best = max(usable, key=lambda row: (float(row["average_precision"]), -float(row["C"])))
    selected_c = float(best["C"])
    scaler, classifier, validation_frame_rows, _ = fitted[selected_c]
    threshold_result = select_f1_threshold(validation_frame_rows)
    threshold = threshold_result["threshold"]
    return ProbeModel(
        scaler=scaler,
        classifier=classifier,
        selected_c=selected_c,
        threshold=float(threshold) if threshold is not None else None,
        threshold_f1=threshold_result["f1"],
        history=history,
        cache_key=cache_key,
    )


def score_window_rows(
    model: ProbeModel,
    features: np.ndarray,
    windows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(features) != len(windows):
        raise ProbeError("features and windows have different row counts")
    scores = predict_probe_scores(model.scaler, model.classifier, features)
    return [
        {
            "video_id": window["video_id"],
            "window_id": window["window_id"],
            "available_at_s": float(window["available_at_s"]),
            "score": float(score),
            "valid": True,
            "failure_reason": None,
        }
        for window, score in zip(windows, scores)
    ]


def save_probe_model(path: Path | str, model: ProbeModel) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model": model,
    }
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def load_probe_model(path: Path | str) -> ProbeModel:
    try:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError) as exc:
        raise ProbeError(f"cannot load probe model: {path}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ProbeError(f"incompatible probe model: {path}")
    model = payload.get("model")
    if not isinstance(model, ProbeModel):
        raise ProbeError(f"invalid probe model payload: {path}")
    return model


def load_cached_split(
    prepared_dir: Path | str,
    feature_dir: Path | str,
    *,
    split: str,
    cache_key: str | None = None,
    limit_videos: int | None = None,
    selection: str = "all",
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], str]:
    """Load one split while enforcing cache/window order and key agreement."""

    prepared_dir = Path(prepared_dir)
    feature_dir = Path(feature_dir)
    manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    windows_bundle = json.loads((prepared_dir / "windows.json").read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != windows_bundle.get("manifest_sha256"):
        raise ProbeError("manifest and windows have different manifest_sha256")
    summary_path = feature_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    resolved_key = cache_key or summary.get("cache_key")
    if not resolved_key:
        raise ProbeError("cache key is required (or provide features/summary.json)")
    records_by_video = {str(record["video_id"]): record for record in manifest.get("records", [])}
    windows_by_video: dict[str, list[dict[str, Any]]] = {}
    for window in windows_bundle.get("windows", []):
        if window.get("split") == split and window.get("valid", False):
            windows_by_video.setdefault(str(window["video_id"]), []).append(window)
    if selection != "all":
        selection_ids = manifest.get("splits", {}).get(f"{selection}_ids")
        if not isinstance(selection_ids, list):
            raise ProbeError(f"prepared manifest has no {selection}_ids")
        selection_set = {str(video_id) for video_id in selection_ids}
        ordered_ids = [str(video_id) for video_id in selection_ids if str(video_id) in windows_by_video]
        windows_by_video = {video_id: windows_by_video[video_id] for video_id in ordered_ids if video_id in selection_set}
    else:
        windows_by_video = {video_id: windows_by_video[video_id] for video_id in sorted(windows_by_video)}
    if limit_videos is not None:
        if limit_videos <= 0:
            raise ProbeError("limit_videos must be positive")
        windows_by_video = dict(list(windows_by_video.items())[:limit_videos])
    rows: list[np.ndarray] = []
    selected_windows: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    for video_id in sorted(windows_by_video):
        expected_windows = windows_by_video[video_id]
        cache_path = feature_dir / video_id / str(resolved_key)
        try:
            cached = read_feature_cache(cache_path, expected_key=str(resolved_key))
        except (CacheError, OSError) as exc:
            raise ProbeError(f"cannot load {split} cache for {video_id}: {exc}") from exc
        expected_ids = [str(window["window_id"]) for window in expected_windows]
        if cached["window_ids"] != expected_ids:
            raise ProbeError(f"cache/window order mismatch for {video_id}")
        rows.append(np.asarray(cached["global_features"], dtype=np.float32))
        selected_windows.extend(expected_windows)
        selected_records.append(records_by_video[video_id])
    if not rows:
        raise ProbeError(f"no valid cached windows for split: {split}")
    return np.concatenate(rows, axis=0), selected_windows, selected_records, str(resolved_key)
