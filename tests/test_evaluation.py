from jepa_risk.evaluation import (
    alarms_from_frames,
    assign_window_scores_to_frames,
    bootstrap_classification_metrics,
    classification_metrics,
    common_valid_rows,
    constant_score_metrics,
    event_metrics,
    evaluate_score_payloads,
    select_f1_threshold,
)


def _rows(scores, labels=(0, 1, 0, 1), video_id="v"):
    return [
        {
            "video_id": video_id,
            "frame_id": index,
            "time_s": float(index),
            "label": labels[index],
            "score": score,
            "valid": score is not None,
        }
        for index, score in enumerate(scores)
    ]


def test_assign_window_scores_uses_causal_half_open_intervals() -> None:
    score_rows = [
        {"available_at_s": 2.0, "score": 0.9, "valid": True},
        {"available_at_s": 2.5, "score": 0.1, "valid": True},
    ]

    assert assign_window_scores_to_frames([1.9, 2.0, 2.4, 2.5, 2.99, 3.0], score_rows) == [
        None,
        0.9,
        0.9,
        0.1,
        0.1,
        None,
    ]


def test_invalid_window_does_not_extend_previous_score() -> None:
    rows = [
        {"available_at_s": 2.0, "score": 0.9, "valid": True},
        {"available_at_s": 2.5, "score": 0.1, "valid": False, "failure_reason": "failed"},
    ]

    assert assign_window_scores_to_frames([2.4, 2.5, 2.9], rows) == [0.9, None, None]


def test_classification_metrics_and_constant_baseline() -> None:
    rows = _rows([0.1, 0.8, 0.2, 0.9])
    metrics = classification_metrics(rows)
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert constant_score_metrics(rows)["roc_auc"] == 0.5
    assert constant_score_metrics(rows)["average_precision"] == 0.5


def test_classification_metrics_reports_single_class_and_coverage() -> None:
    rows = _rows([0.1, None, 0.2, None], labels=(0, 0, 0, 0))
    metrics = classification_metrics(rows)
    assert metrics["coverage"] == 0.5
    assert metrics["roc_auc"] is None
    assert metrics["reason"] == "single_class"


def test_common_valid_rows_requires_both_streams() -> None:
    rows_a = _rows([0.1, 0.8, None, 0.9])
    rows_b = _rows([0.2, 0.7, 0.3, None])

    common = common_valid_rows(rows_a, rows_b)

    assert [(row["frame_id"], row["score_a"], row["score_b"]) for row in common] == [(0, 0.1, 0.2), (1, 0.8, 0.7)]


def test_threshold_prefers_higher_score_on_f1_tie() -> None:
    rows = _rows([0.2, 0.8, 0.4, 0.9])
    selected = select_f1_threshold(rows)
    assert selected["threshold"] == 0.8
    assert selected["f1"] == 1.0


def test_alarms_split_on_invalid_frame() -> None:
    rows = _rows([0.9, None, 0.9, 0.1])
    alarms = alarms_from_frames(rows, threshold=0.5)
    assert [(alarm["start_s"], alarm["end_s"]) for alarm in alarms] == [(0.0, 1.0), (2.0, 3.0)]


def test_event_metrics_reports_detection_delay_and_false_alarm() -> None:
    record = {
        "video_id": "v",
        "anomaly_intervals_s": [[2.0, 3.0]],
    }
    rows = _rows([0.9, 0.1, 0.9, 0.9, 0.1], labels=(0, 0, 1, 1, 0))

    metrics = event_metrics([record], rows, threshold=0.5)

    assert metrics["event_count"] == 1
    assert metrics["detected_count"] == 1
    assert metrics["detection_rate"] == 1.0
    assert metrics["mean_delay_s"] == 0.0
    assert metrics["false_alarm_count"] == 1


def test_evaluate_score_payloads_builds_a_b_and_common_metrics() -> None:
    record = {
        "video_id": "v",
        "split": "validation",
        "frame_ids": [0, 1, 2, 3],
        "frame_times_s": [0.0, 0.5, 1.0, 1.5],
        "anomaly_intervals_s": [[1.0, 1.5]],
    }
    scores = [
        {"video_id": "v", "available_at_s": 0.5, "score": 0.1, "valid": True},
        {"video_id": "v", "available_at_s": 1.0, "score": 0.9, "valid": True},
    ]

    result = evaluate_score_payloads(
        [record],
        {"cache_key": "b-key", "scores": scores},
        {"cache_key": "a-key", "scores": scores},
        split="validation",
        n_bootstrap=10,
    )

    assert result["video_count"] == 1
    assert result["common"]["frame_count"] == 2
    assert result["a"]["frame_metrics"]["average_precision"] == 1.0
    assert result["b"]["frame_metrics"]["roc_auc"] == 1.0
    assert result["cache_keys"] == {"a": "a-key", "b": "b-key"}


def test_bootstrap_is_video_level_and_reproducible() -> None:
    rows = _rows([0.1, 0.8, 0.2, 0.9], video_id="a") + _rows(
        [0.2, 0.7, 0.3, 0.6], video_id="b"
    )

    first = bootstrap_classification_metrics(rows, n_bootstrap=20, seed=42)
    second = bootstrap_classification_metrics(rows, n_bootstrap=20, seed=42)

    assert first == second
    assert first["valid_resamples"] == 20
