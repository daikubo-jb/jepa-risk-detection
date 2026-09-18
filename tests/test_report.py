from __future__ import annotations

import json
from pathlib import Path

from jepa_risk.report import build_report, write_report


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_report_estimates_cost_and_lists_failure_examples(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    config = tmp_path / "phase1.yaml"
    config.write_text("project: {name: test}\n", encoding="utf-8")
    records = [
        {
            "video_id": "negative",
            "official_split": "train",
            "split": "train",
            "frame_ids": [0, 1, 2, 3, 4],
            "frame_times_s": [0.0, 0.5, 1.0, 1.5, 2.0],
            "anomaly_intervals_s": [],
        },
        {
            "video_id": "positive",
            "official_split": "train",
            "split": "train",
            "frame_ids": [0, 1, 2, 3, 4],
            "frame_times_s": [0.0, 0.5, 1.0, 1.5, 2.0],
            "anomaly_intervals_s": [[1.0, 2.0]],
        },
    ]
    windows = [
        {"video_id": "negative", "split": "train", "valid": True, "label": 0},
        {"video_id": "positive", "split": "train", "valid": True, "label": 1},
    ]
    _write_json(prepared / "manifest.json", {"manifest_sha256": "same", "records": records})
    _write_json(prepared / "windows.json", {"manifest_sha256": "same", "windows": windows})
    measurement = {
        "prepared": "old-prepared",
        "windows": 2,
        "total_seconds": 10.0,
        "seconds_per_window": 5.0,
        "storage_bytes": 200,
    }
    _write_json(tmp_path / "feature-measurement.json", measurement)
    scores = {
        "scores": [
            {"video_id": "negative", "available_at_s": 0.0, "score": 0.9, "valid": True},
            {"video_id": "negative", "available_at_s": 0.5, "score": 0.9, "valid": True},
            {"video_id": "positive", "available_at_s": 0.0, "score": 0.1, "valid": True},
            {"video_id": "positive", "available_at_s": 0.5, "score": 0.1, "valid": True},
        ]
    }
    _write_json(tmp_path / "scores.json", scores)
    metrics = {"b": {"threshold": 0.5}, "a": {"threshold": {"threshold": 0.5}}}
    _write_json(tmp_path / "metrics.json", metrics)

    report = build_report(
        config_path=config,
        prepared_dir=prepared,
        feature_measurement_path=tmp_path / "feature-measurement.json",
        probe_metrics_path=tmp_path / "metrics.json",
        probe_scores_path=tmp_path / "scores.json",
    )

    assert report["dataset"]["videos"] == 2
    assert report["dataset"]["valid_windows"] == 2
    assert report["estimates"]["feature_extraction"]["estimated"]["seconds"] == 10.0
    assert report["estimates"]["feature_cache_storage"]["estimated_bytes"] == 200.0
    assert any(item["type"] == "false_alarm" for item in report["failure_examples"]["probe"]["examples"])
    assert report["limitations"]

    json_path, markdown_path = write_report(report, tmp_path / "report")
    assert json_path.is_file()
    assert markdown_path.is_file()
    assert "Full-run estimates" in markdown_path.read_text(encoding="utf-8")
