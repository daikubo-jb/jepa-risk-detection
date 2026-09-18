from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

import jepa_risk.manifest as manifest_module
from jepa_risk.manifest import MetadataError, assign_splits, load_manifest
from jepa_risk.windows import generate_windows


def test_load_manifest_normalizes_nexar_parquet(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "nexar"
    videos = root / "train"
    (videos / "positive").mkdir(parents=True)
    (videos / "negative").mkdir(parents=True)
    (videos / "positive" / "00001.mp4").write_bytes(b"positive")
    (videos / "negative" / "00002.mp4").write_bytes(b"negative")
    metadata_path = root / "metadata.parquet"
    video_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    table = pa.Table.from_arrays(
        [
            pa.array(
                [
                    {"bytes": None, "path": "hf://datasets/nexar/train/positive/00001.mp4"},
                    {"bytes": None, "path": "hf://datasets/nexar/train/negative/00002.mp4"},
                ],
                type=video_type,
            ),
            pa.array([5.0, None], type=pa.float64()),
            pa.array([3.5, None], type=pa.float64()),
            pa.array(["Normal", "Dark"]),
            pa.array(["Clear", "Rain"]),
            pa.array(["Urban", "Highway"]),
            pa.array([None, None], type=pa.float64()),
        ],
        names=[
            "video",
            "time_of_event",
            "time_of_alert",
            "light_conditions",
            "weather",
            "scene",
            "time_to_accident",
        ],
    )
    parquet.write_table(table, metadata_path)
    monkeypatch.setattr(manifest_module, "_probe_video", lambda path, fallback_fps: (30.0, 60))

    records = load_manifest(
        {
            "data": {
                "dataset": "nexar",
                "root": "nexar",
                "annotations": "nexar/metadata.parquet",
                "videos": "nexar/train",
                "fps": 30,
            }
        },
        tmp_path,
    )

    assert [record["video_id"] for record in records] == [
        "nexar_positive_00001",
        "nexar_negative_00002",
    ]
    positive, negative = records
    assert positive["label"] == 1
    assert positive["anomaly_intervals_s"] == [[3.5, 5.0]]
    assert positive["alert_time_s"] == 3.5
    assert positive["event_time_s"] == 5.0
    assert positive["fps"] == 30.0
    assert len(positive["frame_ids"]) == 60
    assert positive["data_path"].endswith("train/positive/00001.mp4")
    assert negative["label"] == 0
    assert negative["anomaly_intervals_s"] == []
    assert negative["event_time_s"] is None


def test_load_manifest_rejects_removed_dataset(tmp_path: Path) -> None:
    with pytest.raises(MetadataError, match="only Nexar is supported"):
        load_manifest({"data": {"dataset": "other"}}, tmp_path)


def test_assign_splits_keeps_source_groups_together() -> None:
    records = [
        {"video_id": "a_000001", "source_group_id": "a", "official_split": "train"},
        {"video_id": "a_000002", "source_group_id": "a", "official_split": "train"},
        {"video_id": "b_000001", "source_group_id": "b", "official_split": "train"},
        {"video_id": "c_000001", "source_group_id": "c", "official_split": "train"},
        {"video_id": "test_000001", "source_group_id": "test", "official_split": "test"},
    ]

    result = assign_splits(records, validation_fraction=0.2, seed=42)

    assert set(result["ids"]["train"]).isdisjoint(result["ids"]["validation"])
    assert "test_000001" in result["ids"]["test"]
    assignments = {record["video_id"]: record["split"] for record in records}
    assert assignments["a_000001"] == assignments["a_000002"]


def test_assign_splits_uses_seeded_selection_order_for_smoke_and_pilot() -> None:
    records = [
        {"video_id": f"video_{index:06d}", "source_group_id": f"group-{index}", "official_split": "train"}
        for index in range(10)
    ]
    first = assign_splits(records, validation_fraction=0.2, seed=42)
    second = assign_splits(records, validation_fraction=0.2, seed=42)
    different_seed = assign_splits(records, validation_fraction=0.2, seed=7)

    assert first["smoke20_ids"] == second["smoke20_ids"]
    assert first["selection"]["algorithm"].startswith("sort video IDs")
    assert first["smoke20_ids"] != sorted(first["ids"]["train"])[:20]
    assert first["smoke20_ids"] != different_seed["smoke20_ids"]


def test_assign_splits_excludes_train_group_overlapping_official_test() -> None:
    records = [
        {"video_id": "same_000001", "source_group_id": "same", "official_split": "train"},
        {"video_id": "same_000002", "source_group_id": "same", "official_split": "test"},
        {"video_id": "other_000001", "source_group_id": "other", "official_split": "train"},
        {"video_id": "third_000001", "source_group_id": "third", "official_split": "train"},
    ]

    result = assign_splits(records, validation_fraction=0.5, seed=42)

    assert result["excluded_overlap_ids"] == ["same_000001"]
    assert "same_000001" not in result["ids"]["train"]
    assert "same_000002" in result["ids"]["test"]
    assert next(record for record in records if record["video_id"] == "same_000001")["split"] == "excluded_overlap"


def test_generate_windows_uses_duplicate_sampling_and_end_label() -> None:
    record = {
        "video_id": "video_000001",
        "split": "train",
        "fps": 10,
        "frame_ids": list(range(50)),
        "frame_times_s": [index / 10 for index in range(50)],
        "anomaly_intervals_s": [[2.0, 3.1]],
    }

    windows = generate_windows(record)

    assert windows[0]["available_at_s"] == 2.0
    assert len(windows[0]["requested_times_s"]) == 64
    assert len(windows[0]["sampled_frame_ids"]) == 64
    assert len(set(windows[0]["sampled_frame_ids"])) < 64
    assert max(windows[0]["sampled_times_s"]) < windows[0]["available_at_s"]
    assert min(windows[0]["sampled_times_s"]) >= windows[0]["observation_start_s"]
    assert windows[0]["label"] == 1
    assert windows[2]["available_at_s"] == 3.0
    assert windows[2]["label"] == 1
    assert windows[3]["available_at_s"] == 3.5
    assert windows[3]["label"] == 0


def test_generate_windows_marks_known_missing_frame_files_invalid() -> None:
    record = {
        "video_id": "video_000001",
        "split": "train",
        "fps": 10,
        "frame_ids": list(range(50)),
        "frame_times_s": [index / 10 for index in range(50)],
        "available_frame_ids": [index for index in range(50) if index != 10],
        "anomaly_intervals_s": [],
    }

    windows = generate_windows(record)

    assert windows[0]["valid"] is False
    assert windows[0]["failure_reason"] == "missing_frame_files"
