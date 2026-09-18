"""Causal frame-level evaluation for window scores."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


class EvaluationError(ValueError):
    """Raised when evaluation inputs violate the score contract."""


def _finite_score(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def assign_window_scores_to_frames(
    frame_times_s: Sequence[float],
    score_rows: Iterable[dict[str, Any]],
    *,
    stride_sec: float = 0.5,
) -> list[float | None]:
    """Assign each score to its causal half-open interval ``[t, t+stride)``.

    Invalid or missing windows leave frames unscored. No previous score is
    carried across an invalid interval.
    """
    if stride_sec <= 0:
        raise EvaluationError("stride_sec must be positive")
    indexed: list[tuple[float, int, dict[str, Any]]] = []
    for index, row in enumerate(score_rows):
        try:
            available_at = float(row["available_at_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationError("score row has an invalid available_at_s") from exc
        indexed.append((available_at, index, row))
    indexed.sort(key=lambda item: (item[0], item[1]))

    assigned: list[float | None] = []
    for frame_time in frame_times_s:
        time_s = float(frame_time)
        candidates = [item for item in indexed if item[0] <= time_s < item[0] + stride_sec]
        if not candidates:
            assigned.append(None)
        else:
            latest = max(candidates, key=lambda item: (item[0], item[1]))[2]
            assigned.append(
                float(latest["score"])
                if bool(latest.get("valid", True)) and _finite_score(latest.get("score"))
                else None
            )
    return assigned


def frame_rows_for_record(
    record: dict[str, Any],
    score_rows: Iterable[dict[str, Any]],
    *,
    stride_sec: float = 0.5,
) -> list[dict[str, Any]]:
    """Create frame rows with labels and causally assigned scores for one video."""
    frame_times = [float(value) for value in record.get("frame_times_s", [])]
    frame_ids = [int(value) for value in record.get("frame_ids", [])]
    if len(frame_times) != len(frame_ids):
        raise EvaluationError(f"frame arrays do not match for {record.get('video_id')}")
    scores = assign_window_scores_to_frames(frame_times, score_rows, stride_sec=stride_sec)
    intervals = record.get("anomaly_intervals_s", [])
    rows: list[dict[str, Any]] = []
    for frame_id, time_s, score in zip(frame_ids, frame_times, scores):
        label = int(any(float(start) <= time_s < float(end) for start, end in intervals))
        rows.append(
            {
                "video_id": record["video_id"],
                "frame_id": frame_id,
                "time_s": time_s,
                "label": label,
                "score": score,
                "valid": score is not None,
                "failure_reason": None if score is not None else "unscored",
            }
        )
    return rows


def frame_rows_for_records(
    records: Iterable[dict[str, Any]],
    scores_by_video: dict[str, Iterable[dict[str, Any]]],
    *,
    stride_sec: float = 0.5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.extend(
            frame_rows_for_record(
                record,
                scores_by_video.get(record["video_id"], []),
                stride_sec=stride_sec,
            )
        )
    return rows


def common_valid_rows(
    rows_a: Iterable[dict[str, Any]],
    rows_b: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join A/B rows by video and frame, retaining only the common valid set."""
    key = lambda row: (row["video_id"], int(row["frame_id"]))
    indexed_b = {key(row): row for row in rows_b}
    common: list[dict[str, Any]] = []
    for row_a in rows_a:
        row_b = indexed_b.get(key(row_a))
        if row_b is None or not row_a.get("valid") or not row_b.get("valid"):
            continue
        if not _finite_score(row_a.get("score")) or not _finite_score(row_b.get("score")):
            continue
        if int(row_a["label"]) != int(row_b["label"]):
            raise EvaluationError(f"labels disagree at {key(row_a)}")
        common.append(
            {
                **row_a,
                "score_a": float(row_a["score"]),
                "score_b": float(row_b["score"]),
                "valid_a": True,
                "valid_b": True,
                "common_valid": True,
            }
        )
    return common


def _usable_rows(rows: Iterable[dict[str, Any]], score_field: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if bool(row.get("valid", True)) and _finite_score(row.get(score_field))
    ]


def classification_metrics(
    rows: Iterable[dict[str, Any]],
    *,
    score_field: str = "score",
) -> dict[str, Any]:
    """Return ROC-AUC/AP without fabricating values for empty/single classes."""
    rows_list = list(rows)
    usable = _usable_rows(rows_list, score_field)
    labels = np.asarray([int(row["label"]) for row in usable], dtype=np.int64)
    scores = np.asarray([float(row[score_field]) for row in usable], dtype=np.float64)
    result: dict[str, Any] = {
        "total_rows": len(rows_list),
        "valid_rows": len(usable),
        "coverage": len(usable) / len(rows_list) if rows_list else 0.0,
        "positive_rows": int(labels.sum()) if len(labels) else 0,
        "negative_rows": int((labels == 0).sum()) if len(labels) else 0,
        "roc_auc": None,
        "average_precision": None,
        "reason": None,
    }
    if not usable:
        result["reason"] = "no_valid_scores"
        return result
    if len(np.unique(labels)) < 2:
        result["reason"] = "single_class"
        return result
    result["roc_auc"] = float(roc_auc_score(labels, scores))
    result["average_precision"] = float(average_precision_score(labels, scores))
    return result


def constant_score_metrics(rows: Iterable[dict[str, Any]], value: float = 0.5) -> dict[str, Any]:
    rows_list = list(rows)
    constant_rows = [{**row, "score": value} for row in rows_list]
    return classification_metrics(constant_rows)


def select_f1_threshold(
    rows: Iterable[dict[str, Any]],
    *,
    score_field: str = "score",
) -> dict[str, Any]:
    """Select validation threshold by F1, breaking ties toward higher values."""
    usable = _usable_rows(rows, score_field)
    if not usable:
        return {"threshold": None, "f1": None, "reason": "no_valid_scores"}
    labels = np.asarray([int(row["label"]) for row in usable], dtype=np.int64)
    scores = np.asarray([float(row[score_field]) for row in usable], dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    group_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    true_positives = np.cumsum(sorted_labels, dtype=np.int64)[group_ends]
    predicted_positives = group_ends + 1
    false_positives = predicted_positives - true_positives
    false_negatives = int(labels.sum()) - true_positives
    denominators = 2 * true_positives + false_positives + false_negatives
    f1_values = np.divide(
        2.0 * true_positives,
        denominators,
        out=np.zeros_like(true_positives, dtype=np.float64),
        where=denominators != 0,
    )
    best_f1 = float(np.max(f1_values)) if len(f1_values) else 0.0
    if best_f1 <= 0.0:
        best_threshold = float("inf")
    else:
        # Scores are descending, so the first maximum implements the
        # higher-threshold tie break from the original candidate loop.
        best_threshold = float(sorted_scores[group_ends[int(np.argmax(f1_values))]])
        best_f1 = float(f1_score(labels, scores >= best_threshold, zero_division=0))
    return {"threshold": best_threshold, "f1": best_f1, "reason": None}


def _frame_step(rows: list[dict[str, Any]], default: float = 0.1) -> float:
    times = sorted(float(row["time_s"]) for row in rows)
    differences = [right - left for left, right in zip(times, times[1:]) if right > left]
    return float(np.median(differences)) if differences else default


def alarms_from_frames(
    rows: Iterable[dict[str, Any]],
    threshold: float,
    *,
    score_field: str = "score",
) -> list[dict[str, Any]]:
    """Build threshold alarms; invalid frames split alarms and are never filled."""
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[str(row["video_id"])].append(row)
    alarms: list[dict[str, Any]] = []
    for video_id, video_rows in by_video.items():
        ordered = sorted(video_rows, key=lambda row: float(row["time_s"]))
        step = _frame_step(ordered)
        current: dict[str, Any] | None = None
        previous_time: float | None = None
        for row in ordered:
            time_s = float(row["time_s"])
            active = bool(row.get("valid")) and _finite_score(row.get(score_field)) and float(row[score_field]) >= threshold
            scored = bool(row.get("valid")) and _finite_score(row.get(score_field))
            contiguous = previous_time is not None and time_s - previous_time <= step * 1.5
            if active and current is not None and contiguous:
                current["end_s"] = time_s + step
            elif active:
                if current is not None:
                    alarms.append(current)
                current = {"video_id": video_id, "start_s": time_s, "end_s": time_s + step}
            elif current is not None:
                alarms.append(current)
                current = None
            previous_time = time_s if scored else None
        if current is not None:
            alarms.append(current)
    return alarms


def event_metrics(
    records: Iterable[dict[str, Any]],
    rows: Iterable[dict[str, Any]],
    threshold: float,
    *,
    score_field: str = "score",
) -> dict[str, Any]:
    """Measure event detection only for fully scored annotated intervals."""
    rows_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_video[str(row["video_id"])].append(row)
    alarms = alarms_from_frames(rows, threshold, score_field=score_field)
    alarms_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for alarm in alarms:
        alarms_by_video[alarm["video_id"]].append(alarm)

    event_count = detected_count = partial_count = 0
    delays: list[float] = []
    false_alarm_count = 0
    valid_non_event_seconds = 0.0
    for record in records:
        video_id = str(record["video_id"])
        video_rows = rows_by_video.get(video_id, [])
        video_alarms = alarms_by_video.get(video_id, [])
        intervals = [(float(start), float(end)) for start, end in record.get("anomaly_intervals_s", [])]
        interval_alarms: set[int] = set()
        for start, end in intervals:
            event_count += 1
            annotation_rows = [row for row in video_rows if start <= float(row["time_s"]) < end]
            fully_valid = bool(annotation_rows) and all(
                bool(row.get("valid")) and _finite_score(row.get(score_field)) for row in annotation_rows
            )
            if not fully_valid:
                partial_count += 1
                continue
            overlaps = [
                (index, alarm)
                for index, alarm in enumerate(video_alarms)
                if float(alarm["end_s"]) > start and float(alarm["start_s"]) < end
            ]
            interval_alarms.update(index for index, _ in overlaps)
            if overlaps:
                detected_count += 1
                _, first_alarm = min(overlaps, key=lambda item: float(item[1]["start_s"]))
                delays.append(max(0.0, float(first_alarm["start_s"]) - start))
        for index, alarm in enumerate(video_alarms):
            if index not in interval_alarms:
                false_alarm_count += 1
        step = _frame_step(video_rows)
        valid_non_event_seconds += sum(
            step
            for row in video_rows
            if bool(row.get("valid"))
            and not any(start <= float(row["time_s"]) < end for start, end in intervals)
        )
    return {
        "event_count": event_count,
        "fully_scored_event_count": event_count - partial_count,
        "partial_event_count": partial_count,
        "detected_count": detected_count,
        "detection_rate": (
            detected_count / (event_count - partial_count) if event_count - partial_count else None
        ),
        "delays_s": delays,
        "mean_delay_s": float(np.mean(delays)) if delays else None,
        "false_alarm_count": false_alarm_count,
        "false_alarm_rate_per_valid_second": (
            false_alarm_count / valid_non_event_seconds if valid_non_event_seconds > 0 else None
        ),
    }


def bootstrap_classification_metrics(
    rows: Iterable[dict[str, Any]],
    *,
    score_field: str = "score",
    metric: str = "average_precision",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap a frame metric by sampling videos, keeping each video intact."""
    if n_bootstrap <= 0:
        raise EvaluationError("n_bootstrap must be positive")
    rows_list = list(rows)
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_list:
        by_video[str(row["video_id"])].append(row)
    videos = sorted(by_video)
    estimate = classification_metrics(rows_list, score_field=score_field).get(metric)
    if not videos:
        return {
            "metric": metric,
            "estimate": estimate,
            "ci95_percentile": None,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "valid_resamples": 0,
            "invalid_resamples": n_bootstrap,
        }
    samples: list[float] = []
    invalid = 0
    rng = np.random.default_rng(seed)
    for sampled_videos in rng.choice(videos, size=(n_bootstrap, len(videos)), replace=True):
        sampled_rows = [row for video_id in sampled_videos for row in by_video[str(video_id)]]
        value = classification_metrics(sampled_rows, score_field=score_field).get(metric)
        if value is None:
            invalid += 1
        else:
            samples.append(float(value))
    interval = [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))] if samples else None
    return {
        "metric": metric,
        "estimate": estimate,
        "ci95_percentile": interval,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "valid_resamples": len(samples),
        "invalid_resamples": invalid,
    }


def _score_rows_by_video(payload: dict[str, Any], name: str) -> dict[str, list[dict[str, Any]]]:
    score_rows = payload.get("scores")
    if not isinstance(score_rows, list):
        raise EvaluationError(f"{name} score artifact has no scores list")
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        if not isinstance(row, dict) or "video_id" not in row:
            raise EvaluationError(f"{name} score artifact contains an invalid score row")
        by_video[str(row["video_id"])].append(row)
    return by_video


def evaluate_score_payloads(
    records: Iterable[dict[str, Any]],
    probe_payload: dict[str, Any],
    prediction_payload: dict[str, Any],
    *,
    stride_sec: float = 0.5,
    n_bootstrap: int = 1000,
    seed: int = 42,
    split: str | None = None,
    selection: str = "all",
) -> dict[str, Any]:
    """Evaluate saved B/probe and A/predictor scores on one frame contract."""
    records_list = list(records)
    probe_by_video = _score_rows_by_video(probe_payload, "probe")
    prediction_by_video = _score_rows_by_video(prediction_payload, "prediction")
    selected_video_ids = sorted(set(probe_by_video) | set(prediction_by_video))
    records_by_id = {str(record["video_id"]): record for record in records_list}
    selected_records = [records_by_id[video_id] for video_id in selected_video_ids if video_id in records_by_id]
    if not selected_records:
        raise EvaluationError("score artifacts do not match any prepared records")
    rows_b = frame_rows_for_records(selected_records, probe_by_video, stride_sec=stride_sec)
    rows_a = frame_rows_for_records(selected_records, prediction_by_video, stride_sec=stride_sec)
    common = common_valid_rows(rows_a, rows_b)

    def metric_bundle(rows: list[dict[str, Any]], *, score_field: str = "score") -> dict[str, Any]:
        frame = classification_metrics(rows, score_field=score_field)
        baseline = constant_score_metrics(rows)
        bootstrap = bootstrap_classification_metrics(
            rows,
            score_field=score_field,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        threshold = select_f1_threshold(rows, score_field=score_field)
        threshold_value = threshold.get("threshold")
        events = (
            event_metrics(selected_records, rows, float(threshold_value), score_field=score_field)
            if threshold_value is not None
            else None
        )
        return {
            "frame_metrics": frame,
            "constant_baseline": baseline,
            "bootstrap_average_precision": bootstrap,
            "threshold": threshold,
            "event_metrics": events,
        }

    probe_metrics = metric_bundle(rows_b)
    prediction_metrics = metric_bundle(rows_a)
    common_a = classification_metrics(common, score_field="score_a")
    common_b = classification_metrics(common, score_field="score_b")
    row_key = lambda row: (str(row["video_id"]), int(row["frame_id"]))
    common_keys = {row_key(row) for row in common}
    a_valid_keys = {row_key(row) for row in rows_a if row.get("valid") and _finite_score(row.get("score"))}
    b_valid_keys = {row_key(row) for row in rows_b if row.get("valid") and _finite_score(row.get("score"))}
    common_a_bootstrap = (
        prediction_metrics["bootstrap_average_precision"]
        if a_valid_keys == common_keys
        else bootstrap_classification_metrics(
            common,
            score_field="score_a",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    )
    common_b_bootstrap = (
        probe_metrics["bootstrap_average_precision"]
        if b_valid_keys == common_keys
        else bootstrap_classification_metrics(
            common,
            score_field="score_b",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    )
    return {
        "schema_version": 1,
        "split": split,
        "selection": selection,
        "video_count": len(selected_records),
        "window_count": {
            "probe": sum(len(rows) for rows in probe_by_video.values()),
            "prediction": sum(len(rows) for rows in prediction_by_video.values()),
        },
        "frame_count": {"probe": len(rows_b), "prediction": len(rows_a), "common": len(common)},
        "a": prediction_metrics,
        "b": probe_metrics,
        "common": {
            "frame_count": len(common),
            "a_frame_metrics": common_a,
            "b_frame_metrics": common_b,
            "a_bootstrap_average_precision": common_a_bootstrap,
            "b_bootstrap_average_precision": common_b_bootstrap,
        },
        "cache_keys": {
            "a": prediction_payload.get("cache_key"),
            "b": probe_payload.get("cache_key"),
        },
    }
