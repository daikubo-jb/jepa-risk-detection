"""Experiment summary and full-run cost estimates."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from .evaluation import alarms_from_frames, frame_rows_for_records
from .manifest import write_json_atomic


class ReportError(ValueError):
    """Raised when report inputs do not satisfy the artifact contract."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"expected JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_object(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return _read_object(path)


def _hours(seconds: float) -> dict[str, float]:
    return {
        "seconds": float(seconds),
        "hours": float(seconds / 3600.0),
        "days": float(seconds / 86400.0),
    }


def _runtime_estimate(
    measurement: dict[str, Any] | None,
    total_windows: int,
    *,
    name: str,
) -> dict[str, Any] | None:
    if measurement is None:
        return None
    if "total_seconds" not in measurement and isinstance(measurement.get("videos"), list):
        videos = [video for video in measurement["videos"] if isinstance(video, dict)]
        measurement = {
            **measurement,
            "total_seconds": sum(float(video.get("seconds", 0.0)) for video in videos),
            "windows": sum(int(video.get("windows", 0)) for video in videos),
        }
    try:
        measured_windows = int(measurement.get("windows", 0))
        measured_seconds = float(measurement.get("total_seconds", 0.0))
    except (TypeError, ValueError) as exc:
        raise ReportError(f"invalid {name} measurement") from exc
    if measured_windows <= 0 or measured_seconds <= 0:
        raise ReportError(f"{name} measurement must contain positive windows and total_seconds")
    seconds_per_window = float(measurement.get("seconds_per_window", measured_seconds / measured_windows))
    estimate = {
        "source_windows": measured_windows,
        "source_seconds": measured_seconds,
        "seconds_per_window": seconds_per_window,
        "estimated_windows": total_windows,
        "estimated": _hours(seconds_per_window * total_windows),
        "source": measurement.get("prepared") or measurement.get("checkpoint"),
    }
    if "batch_size" in measurement:
        estimate["batch_size"] = measurement["batch_size"]
    if "device" in measurement:
        estimate["device"] = measurement["device"]
    return estimate


def _storage_estimate(measurement: dict[str, Any] | None, total_windows: int) -> dict[str, Any] | None:
    if measurement is None or "storage_bytes" not in measurement:
        return None
    try:
        source_windows = int(measurement["windows"])
        storage_bytes = int(measurement["storage_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReportError("invalid feature storage measurement") from exc
    if source_windows <= 0 or storage_bytes < 0:
        raise ReportError("feature storage measurement has invalid values")
    estimated_bytes = storage_bytes / source_windows * total_windows
    return {
        "source_bytes": storage_bytes,
        "source_mib": storage_bytes / 1024**2,
        "bytes_per_window": storage_bytes / source_windows,
        "estimated_bytes": estimated_bytes,
        "estimated_gib": estimated_bytes / 1024**3,
    }


def _score_failure_examples(
    score_payload: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    records_by_id: dict[str, dict[str, Any]],
    *,
    metric_key: str,
    max_examples: int = 100,
) -> dict[str, Any] | None:
    if score_payload is None:
        return None
    score_rows = score_payload.get("scores")
    if not isinstance(score_rows, list):
        raise ReportError("score artifact has no scores list")
    video_ids: list[str] = []
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        if not isinstance(row, dict) or "video_id" not in row:
            raise ReportError("score artifact contains an invalid score row")
        video_id = str(row["video_id"])
        by_video[video_id].append(row)
        if video_id not in video_ids:
            video_ids.append(video_id)
    records = [records_by_id[video_id] for video_id in video_ids if video_id in records_by_id]
    frame_rows = frame_rows_for_records(records, by_video, stride_sec=0.5)
    threshold: float | None = None
    if metrics:
        selected = metrics.get(metric_key) or metrics
        candidate = selected.get("threshold")
        if isinstance(candidate, dict):
            candidate = candidate.get("threshold")
        if candidate is not None:
            try:
                threshold = float(candidate)
            except (TypeError, ValueError) as exc:
                raise ReportError(f"invalid {metric_key} threshold") from exc
    if threshold is None:
        return {
            "video_count": len(video_ids),
            "score_rows": len(score_rows),
            "threshold": None,
            "examples": [],
            "reason": "threshold_unavailable",
        }

    alarms = alarms_from_frames(frame_rows, threshold)
    alarms_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for alarm in alarms:
        alarms_by_video[str(alarm["video_id"])].append(alarm)
    examples: list[dict[str, Any]] = []
    for record in records:
        video_id = str(record["video_id"])
        rows = [row for row in frame_rows if str(row["video_id"]) == video_id]
        video_alarms = alarms_by_video.get(video_id, [])
        intervals = [(float(start), float(end)) for start, end in record.get("anomaly_intervals_s", [])]
        matched: set[int] = set()
        for start, end in intervals:
            interval_rows = [row for row in rows if start <= float(row["time_s"]) < end]
            fully_valid = bool(interval_rows) and all(bool(row.get("valid")) for row in interval_rows)
            overlaps = [
                (index, alarm)
                for index, alarm in enumerate(video_alarms)
                if float(alarm["end_s"]) > start and float(alarm["start_s"]) < end
            ]
            matched.update(index for index, _ in overlaps)
            if not fully_valid:
                examples.append(
                    {
                        "type": "partial_event",
                        "video_id": video_id,
                        "start_s": start,
                        "end_s": end,
                    }
                )
            elif not overlaps:
                examples.append(
                    {
                        "type": "missed_event",
                        "video_id": video_id,
                        "start_s": start,
                        "end_s": end,
                    }
                )
        for index, alarm in enumerate(video_alarms):
            if index not in matched:
                examples.append({"type": "false_alarm", **alarm})
    return {
        "video_count": len(video_ids),
        "score_rows": len(score_rows),
        "threshold": threshold,
        "alarm_count": len(alarms),
        "examples": examples[:max_examples],
        "truncated": len(examples) > max_examples,
    }


def build_report(
    *,
    config_path: Path | str,
    prepared_dir: Path | str,
    feature_measurement_path: Path | str | None = None,
    prediction_measurement_path: Path | str | None = None,
    probe_metrics_path: Path | str | None = None,
    prediction_metrics_path: Path | str | None = None,
    probe_scores_path: Path | str | None = None,
    prediction_scores_path: Path | str | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    prepared_dir = Path(prepared_dir).expanduser().resolve()
    manifest = _read_object(prepared_dir / "manifest.json")
    windows_bundle = _read_object(prepared_dir / "windows.json")
    if manifest.get("manifest_sha256") != windows_bundle.get("manifest_sha256"):
        raise ReportError("manifest and windows have different manifest_sha256")
    records = manifest.get("records")
    windows = windows_bundle.get("windows")
    if not isinstance(records, list) or not isinstance(windows, list):
        raise ReportError("prepared artifacts must contain records and windows lists")
    valid_windows = [window for window in windows if isinstance(window, dict) and window.get("valid", False)]
    split_counts = Counter(str(window.get("split")) for window in valid_windows)
    label_counts = Counter(int(window.get("label", 0)) for window in valid_windows)
    records_by_id = {str(record["video_id"]): record for record in records if isinstance(record, dict)}
    feature_measurement = _optional_object(
        Path(feature_measurement_path).expanduser().resolve() if feature_measurement_path else None
    )
    prediction_measurement = _optional_object(
        Path(prediction_measurement_path).expanduser().resolve() if prediction_measurement_path else None
    )
    probe_metrics = _optional_object(Path(probe_metrics_path).expanduser().resolve() if probe_metrics_path else None)
    prediction_metrics = _optional_object(
        Path(prediction_metrics_path).expanduser().resolve() if prediction_metrics_path else None
    )
    probe_scores = _optional_object(Path(probe_scores_path).expanduser().resolve() if probe_scores_path else None)
    prediction_scores = _optional_object(
        Path(prediction_scores_path).expanduser().resolve() if prediction_scores_path else None
    )
    total_windows = len(valid_windows)
    report: dict[str, Any] = {
        "schema_version": 1,
        "config": {"path": str(config_path), "sha256": _sha256_file(config_path)},
        "prepared": {
            "path": str(prepared_dir),
            "manifest_sha256": manifest.get("manifest_sha256"),
            "windows_sha256": hashlib.sha256(
                json.dumps(windows, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
        "dataset": {
            "videos": len(records),
            "valid_windows": total_windows,
            "valid_windows_by_split": dict(sorted(split_counts.items())),
            "window_labels": {str(key): value for key, value in sorted(label_counts.items())},
            "annotated_events": sum(len(record.get("anomaly_intervals_s", [])) for record in records),
            "official_test_videos": sum(1 for record in records if record.get("official_split") == "test"),
        },
        "estimates": {
            "feature_extraction": _runtime_estimate(feature_measurement, total_windows, name="feature"),
            "predictor_scoring": _runtime_estimate(prediction_measurement, total_windows, name="predictor"),
            "feature_cache_storage": _storage_estimate(feature_measurement, total_windows),
        },
        "evaluations": {"probe": probe_metrics, "prediction": prediction_metrics},
        "failure_examples": {
            "probe": _score_failure_examples(probe_scores, probe_metrics, records_by_id, metric_key="b"),
            "prediction": _score_failure_examples(
                prediction_scores, prediction_metrics, records_by_id, metric_key="a"
            ),
        },
        "artifacts": {
            "feature_measurement": str(Path(feature_measurement_path).resolve()) if feature_measurement_path else None,
            "prediction_measurement": str(Path(prediction_measurement_path).resolve()) if prediction_measurement_path else None,
            "probe_metrics": str(Path(probe_metrics_path).resolve()) if probe_metrics_path else None,
            "prediction_metrics": str(Path(prediction_metrics_path).resolve()) if prediction_metrics_path else None,
        },
        "limitations": [],
    }
    if report["dataset"]["official_test_videos"] == 0:
        report["limitations"].append("official test videos are not available; current metrics are exploratory")
    measurement_prepared = str(feature_measurement.get("prepared")) if feature_measurement else ""
    if measurement_prepared and Path(measurement_prepared).name != prepared_dir.name:
        report["limitations"].append(
            "feature measurement was produced from an older prepared artifact; runtime estimate is approximate"
        )
    if prediction_metrics is not None and prediction_metrics.get("video_count", 0) < 20:
        report["limitations"].append("predictor evaluation covers only a small pilot subset")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    estimates = report["estimates"]
    lines = [
        "# V-JEPA 2 Risk Detection Experiment Report",
        "",
        "## Dataset",
        "",
        f"- Videos: {dataset['videos']}",
        f"- Valid windows: {dataset['valid_windows']}",
        f"- Windows by split: `{dataset['valid_windows_by_split']}`",
        f"- Annotated events: {dataset['annotated_events']}",
        f"- Official test videos: {dataset['official_test_videos']}",
        "",
        "## Full-run estimates",
        "",
        "| Path | Seconds/window | Estimated hours | Estimated days |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (("feature_extraction", "B feature extraction"), ("predictor_scoring", "A predictor scoring")):
        estimate = estimates.get(key)
        if estimate:
            lines.append(
                f"| {label} | {estimate['seconds_per_window']:.3f} | "
                f"{estimate['estimated']['hours']:.2f} | {estimate['estimated']['days']:.2f} |"
            )
    storage = estimates.get("feature_cache_storage")
    if storage:
        lines.extend(["", f"Estimated B feature cache: **{storage['estimated_gib']:.2f} GiB**."])
    lines.extend(["", "## Evaluation", ""])
    for key, label in (("probe", "B probe"), ("prediction", "A predictor")):
        metrics = report["evaluations"].get(key) or {}
        if key == "probe":
            frame = metrics.get("b", {}).get("frame_metrics") or metrics.get("frame_metrics")
        else:
            frame = metrics.get("a", {}).get("frame_metrics") or metrics.get("frame_metrics")
        if frame:
            lines.append(
                f"- {label}: ROC-AUC={frame.get('roc_auc')}, AP={frame.get('average_precision')}, "
                f"coverage={frame.get('coverage')}"
            )
    evaluation_source = report["evaluations"].get("probe") or report["evaluations"].get("prediction") or {}
    common = evaluation_source.get("common") or {}
    common_count = common.get("frame_count")
    common_a = common.get("a_frame_metrics") or {}
    common_b = common.get("b_frame_metrics") or {}
    if common_count is not None and (common_a or common_b):
        lines.extend(
            [
                "",
                "### Common frame comparison",
                "",
                f"- Common valid frames: {common_count}",
                f"- A: ROC-AUC={common_a.get('roc_auc')}, AP={common_a.get('average_precision')}",
                f"- B: ROC-AUC={common_b.get('roc_auc')}, AP={common_b.get('average_precision')}",
            ]
        )
        a_bootstrap = common.get("a_bootstrap_average_precision") or {}
        b_bootstrap = common.get("b_bootstrap_average_precision") or {}
        if a_bootstrap.get("ci95_percentile") or b_bootstrap.get("ci95_percentile"):
            lines.extend(
                [
                    f"- A AP bootstrap 95% CI: {a_bootstrap.get('ci95_percentile')}",
                    f"- B AP bootstrap 95% CI: {b_bootstrap.get('ci95_percentile')}",
                ]
            )
    a_metrics = evaluation_source.get("a") or {}
    b_metrics = evaluation_source.get("b") or {}
    a_events = a_metrics.get("event_metrics") or {}
    b_events = b_metrics.get("event_metrics") or {}
    if a_events or b_events:
        lines.extend(
            [
                "",
                "### Threshold event metrics",
                "",
                "| Path | Threshold | Detection rate | Mean delay (s) | False alarms |",
                "|---|---:|---:|---:|---:|",
                f"| A | {(a_metrics.get('threshold') or {}).get('threshold')} | {a_events.get('detection_rate')} | "
                f"{a_events.get('mean_delay_s')} | {a_events.get('false_alarm_count')} |",
                f"| B | {(b_metrics.get('threshold') or {}).get('threshold')} | {b_events.get('detection_rate')} | "
                f"{b_events.get('mean_delay_s')} | {b_events.get('false_alarm_count')} |",
            ]
        )
    lines.extend(["", "## Failure examples", ""])
    for key, label in (("probe", "B"), ("prediction", "A")):
        failures = report["failure_examples"].get(key) or {}
        lines.append(f"- {label}: {len(failures.get('examples', []))} examples")
    if report["limitations"]:
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], output_dir: Path | str) -> tuple[Path, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    markdown_path = output_dir / "report.md"
    write_json_atomic(report, json_path)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path
