from __future__ import annotations

import numpy as np
import pytest

from jepa_risk.probe import ProbeError, load_probe_model, save_probe_model, score_window_rows, train_probe_model


def _record(video_id: str) -> dict:
    return {
        "video_id": video_id,
        "frame_ids": [0, 1, 2, 3],
        "frame_times_s": [0.0, 0.5, 1.0, 1.5],
        "anomaly_intervals_s": [[1.0, 1.5]] if video_id.endswith("positive") else [],
    }


def _windows(video_id: str) -> list[dict]:
    return [
        {"video_id": video_id, "window_id": f"{video_id}@{time}", "available_at_s": time, "label": int(time >= 1.0)}
        for time in (0.5, 1.0, 1.5)
    ]


def test_probe_fits_scaler_on_train_only_and_scores_rows() -> None:
    train = np.zeros((4, 1024), dtype=np.float32)
    train[:, 0] = [0.0, 1.0, 9.0, 10.0]
    labels = [0, 0, 1, 1]
    validation = np.zeros((6, 1024), dtype=np.float32)
    validation[:, 0] = [0.0, 1.0, 9.0, 10.0, 8.0, 2.0]
    validation_windows = _windows("validation-positive") + _windows("validation-negative")
    # The negative record has no positive interval; its windows remain valid
    # and exercise the frame-level AP aggregation across videos.
    model = train_probe_model(
        train,
        labels,
        validation,
        validation_windows,
        [_record("validation-positive"), _record("validation-negative")],
        c_candidates=(0.01, 0.1),
    )
    assert model.scaler.mean_[0] == pytest.approx(5.0)
    assert model.selected_c in {0.01, 0.1}
    rows = score_window_rows(model, validation, validation_windows)
    assert len(rows) == 6
    assert all(row["valid"] and np.isfinite(row["score"]) for row in rows)


def test_probe_model_roundtrip_preserves_scores(tmp_path) -> None:
    train = np.zeros((4, 1024), dtype=np.float32)
    train[:, 0] = [0.0, 1.0, 9.0, 10.0]
    validation = np.zeros((6, 1024), dtype=np.float32)
    validation[:, 0] = [0.0, 1.0, 9.0, 10.0, 8.0, 2.0]
    windows = _windows("validation-positive") + _windows("validation-negative")
    model = train_probe_model(
        train,
        [0, 0, 1, 1],
        validation,
        windows,
        [_record("validation-positive"), _record("validation-negative")],
    )
    before = score_window_rows(model, validation, windows)
    path = save_probe_model(tmp_path / "probe.pkl", model)
    after = score_window_rows(load_probe_model(path), validation, windows)
    assert [row["score"] for row in before] == pytest.approx([row["score"] for row in after])


def test_probe_rejects_single_class_train() -> None:
    with pytest.raises(ProbeError, match="both classes"):
        train_probe_model(
            np.zeros((2, 1024), dtype=np.float32),
            [0, 0],
            np.zeros((2, 1024), dtype=np.float32),
            _windows("validation-positive")[:2],
            [_record("validation-positive")],
        )
