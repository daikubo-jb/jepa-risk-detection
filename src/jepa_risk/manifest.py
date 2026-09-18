from __future__ import annotations

import hashlib
import json
import random
import subprocess
from fractions import Fraction
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable

from sklearn.model_selection import GroupShuffleSplit


class MetadataError(ValueError):
    """Raised when dataset metadata cannot be interpreted safely."""


def _resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base_dir / path).resolve()


def _parse_video_rate(value: Any, fallback: float) -> float:
    if isinstance(value, str) and value and value not in {"0/0", "N/A"}:
        try:
            rate = float(Fraction(value))
        except (ValueError, ZeroDivisionError):
            rate = 0.0
        if rate > 0:
            return rate
    return fallback


def _probe_video(path: Path, fallback_fps: float) -> tuple[float, int]:
    """Read the video rate and frame count without decoding the frames."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,nb_frames",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MetadataError("ffprobe is required to prepare Nexar videos") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown ffprobe error"
        raise MetadataError(f"cannot probe video {path}: {detail}")
    try:
        payload = json.loads(result.stdout)
        stream = (payload.get("streams") or [])[0]
        fallback = float(fallback_fps)
        if fallback <= 0:
            raise ValueError("fallback fps must be positive")
        fps = _parse_video_rate(stream.get("avg_frame_rate"), fallback)
        raw_num_frames = stream.get("nb_frames")
        num_frames = int(raw_num_frames) if raw_num_frames not in {None, "N/A"} else 0
        if num_frames <= 0:
            duration = float((payload.get("format") or {}).get("duration", 0.0))
            num_frames = round(duration * fps)
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MetadataError(f"invalid ffprobe result for {path}") from exc
    if fps <= 0 or num_frames <= 0:
        raise MetadataError(f"video has no usable fps/frame count: {path}")
    return fps, num_frames


def _nexar_metadata_path(data: dict[str, Any], config_dir: Path, root: Path) -> Path:
    configured = _resolve_path(data.get("annotations"), config_dir)
    if configured is not None:
        if not configured.is_file():
            raise FileNotFoundError(f"Nexar metadata file does not exist: {configured}")
        return configured
    candidates = sorted((root / "metadata").glob("*.parquet"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one Nexar metadata parquet under {root / 'metadata'}, found {len(candidates)}"
        )
    return candidates[0]


def _nexar_video_path(remote_path: str, videos_root: Path) -> tuple[str, str, Path]:
    marker = "/train/"
    if marker not in remote_path:
        raise MetadataError(f"Nexar video path is not under train/: {remote_path}")
    relative = remote_path.split(marker, 1)[1]
    parts = PurePosixPath(relative).parts
    if len(parts) != 2 or parts[0] not in {"positive", "negative"} or not parts[1].endswith(".mp4"):
        raise MetadataError(f"invalid Nexar video path: {remote_path}")
    class_name = parts[0]
    source_video_id = Path(parts[1]).stem
    local_path = (videos_root / class_name / parts[1]).resolve()
    return class_name, source_video_id, local_path


def _load_nexar_manifest(config: dict[str, Any], config_dir: Path) -> list[dict[str, Any]]:
    data = config.get("data") or {}
    root = _resolve_path(data.get("root"), config_dir)
    if root is None:
        raise MetadataError("data.root is required for Nexar")
    if not root.is_dir():
        raise FileNotFoundError(f"Nexar data root does not exist: {root}")
    metadata_path = _nexar_metadata_path(data, config_dir, root)
    videos_root = _resolve_path(data.get("videos"), config_dir) or (root / "train")
    fallback_fps = float(data.get("fps", 30.0))
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise MetadataError("Nexar manifest generation requires the pyarrow dependency") from exc
    rows = parquet.read_table(metadata_path).to_pylist()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise MetadataError("Nexar metadata rows must be objects")
        video_info = row.get("video")
        remote_path = video_info.get("path") if isinstance(video_info, dict) else None
        if not isinstance(remote_path, str):
            raise MetadataError("Nexar metadata row has no video.path")
        class_name, source_video_id, video_path = _nexar_video_path(remote_path, videos_root)
        video_id = f"nexar_{class_name}_{source_video_id}"
        if video_id in seen:
            raise MetadataError(f"duplicate Nexar video_id: {video_id}")
        if not video_path.is_file():
            raise FileNotFoundError(f"Nexar video does not exist: {video_path}")
        fps, num_frames = _probe_video(video_path, fallback_fps)
        label = int(class_name == "positive")
        event_time_raw = row.get("time_of_event")
        alert_time_raw = row.get("time_of_alert")
        event_time = float(event_time_raw) if event_time_raw is not None else None
        alert_time = float(alert_time_raw) if alert_time_raw is not None else None
        if label:
            if event_time is None or alert_time is None or not 0 <= alert_time < event_time:
                raise MetadataError(f"invalid Nexar alert/event times: {video_id}")
            anomaly_intervals = [[alert_time, event_time]]
        else:
            if event_time is not None or alert_time is not None:
                raise MetadataError(f"negative Nexar video has event times: {video_id}")
            anomaly_intervals = []
        raw_annotation = dict(row)
        raw_annotation.update(
            {
                "source_video_id": source_video_id,
                "class_name": class_name,
                "label": label,
                "metadata_path": str(metadata_path),
            }
        )
        records.append(
            {
                "video_id": video_id,
                "source_group_id": video_id,
                "official_split": "train",
                "split": "train",
                "fps": fps,
                "frame_ids": list(range(num_frames)),
                "frame_times_s": [round(frame_id / fps, 9) for frame_id in range(num_frames)],
                "anomaly_intervals_s": anomaly_intervals,
                "category": "collision_or_near_collision" if label else None,
                "data_path": str(video_path),
                "available_frame_ids": None,
                "label": label,
                "event_time_s": event_time,
                "alert_time_s": alert_time,
                "raw_annotation": raw_annotation,
                "annotation_contract": {
                    "dataset": "nexar_collision_prediction",
                    "metadata_path": str(metadata_path),
                    "positive_interval": "[time_of_alert, time_of_event)",
                    "time_unit": "seconds from video start",
                    "fps_source": "ffprobe avg_frame_rate",
                },
            }
        )
        seen.add(video_id)
    return records


def load_manifest(config: dict[str, Any], config_dir: Path) -> list[dict[str, Any]]:
    dataset_value = (config.get("data") or {}).get("dataset")
    if not isinstance(dataset_value, str) or not dataset_value.strip():
        raise MetadataError("data.dataset is required; only Nexar is supported")
    dataset = dataset_value.lower()
    if dataset != "nexar":
        raise MetadataError(f"unsupported dataset: {dataset}; only Nexar is supported")
    return _load_nexar_manifest(config, config_dir)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _seeded_order(ids: Iterable[str], seed: int) -> list[str]:
    ordered = sorted(str(video_id) for video_id in ids)
    random.Random(seed).shuffle(ordered)
    return ordered


def assign_splits(records: list[dict[str, Any]], *, validation_fraction: float = 0.2, seed: int = 42) -> dict[str, Any]:
    test_groups = {
        r["source_group_id"] or r["video_id"]
        for r in records
        if r["official_split"] == "test"
    }
    excluded_overlap_ids = {
        r["video_id"]
        for r in records
        if r["official_split"] == "train"
        and (r["source_group_id"] or r["video_id"]) in test_groups
    }
    train_records = sorted(
        (
            r
            for r in records
            if r["official_split"] == "train" and r["video_id"] not in excluded_overlap_ids
        ),
        key=lambda r: r["video_id"],
    )
    groups = [r["source_group_id"] or r["video_id"] for r in train_records]
    if train_records:
        splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed)
        train_indices, validation_indices = next(splitter.split(train_records, groups=groups))
        validation_ids = {train_records[index]["video_id"] for index in validation_indices}
    else:
        validation_ids = set()
    for record in records:
        if record["video_id"] in excluded_overlap_ids:
            record["split"] = "excluded_overlap"
        else:
            record["split"] = (
                "validation" if record["video_id"] in validation_ids else record["official_split"]
            )

    by_split = {
        split: sorted(r["video_id"] for r in records if r["split"] == split)
        for split in ("train", "validation", "test")
    }
    train_ids = by_split["train"]
    validation_ids_sorted = by_split["validation"]
    train_seeded_ids = _seeded_order(train_ids, seed)
    validation_seeded_ids = _seeded_order(validation_ids_sorted, seed)
    pilot_ids = train_seeded_ids[:160] + validation_seeded_ids[:40]
    return {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "group_key": "source_group_id or video_id",
        "official_split_counts": {
            split: sum(r["official_split"] == split for r in records)
            for split in ("train", "test")
        },
        "split_counts": {
            split: len(ids)
            for split, ids in by_split.items()
        },
        "excluded_overlap_ids": sorted(excluded_overlap_ids),
        "ids": by_split,
        "selection": {
            "algorithm": "sort video IDs then random.Random(seed).shuffle",
            "seed": seed,
            "train_order_sha256": _sha256_json(train_seeded_ids),
            "validation_order_sha256": _sha256_json(validation_seeded_ids),
        },
        "smoke20_ids": train_seeded_ids[:20],
        "pilot_ids": pilot_ids,
    }


def manifest_bundle(records: Iterable[dict[str, Any]], splits: dict[str, Any]) -> dict[str, Any]:
    records_list = list(records)
    return {
        "schema_version": 1,
        "records": records_list,
        "manifest_sha256": _sha256_json(records_list),
        "splits": splits,
        "splits_sha256": _sha256_json(splits),
    }


def write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)
