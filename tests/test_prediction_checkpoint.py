from __future__ import annotations

import pytest

from jepa_risk.cli import _read_prediction_checkpoint, _write_prediction_checkpoint


def _fixtures() -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    windows = [
        {"window_id": "video@1.000", "available_at_s": 1.0},
        {"window_id": "video@1.500", "available_at_s": 1.5},
    ]
    summary = {
        "video_id": "video",
        "status": "scored",
        "windows": 2,
        "batch_size": 1,
        "seconds": 1.234,
    }
    scores = [
        {
            "video_id": "video",
            "window_id": window["window_id"],
            "available_at_s": window["available_at_s"],
            "score": 0.1 + index,
            "valid": True,
            "failure_reason": None,
        }
        for index, window in enumerate(windows)
    ]
    return windows, summary, scores


def test_prediction_checkpoint_roundtrip(tmp_path) -> None:
    windows, summary, scores = _fixtures()
    _write_prediction_checkpoint(
        tmp_path,
        key="cache-key",
        split="validation",
        selection="pilot",
        video_id="video",
        summary=summary,
        scores=scores,
    )

    loaded_summary, loaded_scores = _read_prediction_checkpoint(
        tmp_path / "checkpoints" / "cache-key" / "video.json",
        key="cache-key",
        split="validation",
        selection="pilot",
        video_id="video",
        expected_windows=windows,
    )

    assert loaded_summary == summary
    assert loaded_scores == scores


def test_prediction_checkpoint_rejects_window_mismatch(tmp_path) -> None:
    windows, summary, scores = _fixtures()
    _write_prediction_checkpoint(
        tmp_path,
        key="cache-key",
        split="validation",
        selection="pilot",
        video_id="video",
        summary=summary,
        scores=scores,
    )

    with pytest.raises(ValueError, match="window order"):
        _read_prediction_checkpoint(
            tmp_path / "checkpoints" / "cache-key" / "video.json",
            key="cache-key",
            split="validation",
            selection="pilot",
            video_id="video",
            expected_windows=list(reversed(windows)),
        )
