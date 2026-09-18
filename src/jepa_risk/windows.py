from __future__ import annotations

import bisect
from typing import Any


def _nearest_frame(
    frame_times: list[float],
    requested: float,
    *,
    start_s: float | None = None,
    end_s: float | None = None,
) -> int | None:
    """Return the nearest frame, optionally restricted to ``[start_s, end_s)``."""

    lower = bisect.bisect_left(frame_times, start_s) if start_s is not None else 0
    upper = bisect.bisect_left(frame_times, end_s) if end_s is not None else len(frame_times)
    if lower >= upper:
        return None
    index = bisect.bisect_left(frame_times, requested, lower, upper)
    candidates = [candidate for candidate in (index - 1, index) if lower <= candidate < upper]
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: (abs(frame_times[candidate] - requested), candidate))


def generate_windows(
    record: dict[str, Any],
    *,
    window_sec: float = 2.0,
    stride_sec: float = 0.5,
    input_frames: int = 64,
) -> list[dict[str, Any]]:
    if window_sec <= 0 or stride_sec <= 0 or input_frames <= 0:
        raise ValueError("window_sec, stride_sec, and input_frames must be positive")
    frame_ids = [int(value) for value in record["frame_ids"]]
    frame_times = [float(value) for value in record["frame_times_s"]]
    available_frame_ids = record.get("available_frame_ids")
    available_frame_id_set = set(int(value) for value in available_frame_ids) if available_frame_ids is not None else None
    if len(frame_ids) != len(frame_times) or not frame_ids:
        raise ValueError(f"invalid frame arrays for {record['video_id']}")
    duration_sec = len(frame_times) / float(record["fps"])
    windows: list[dict[str, Any]] = []
    index = 0
    while True:
        available_at = window_sec + index * stride_sec
        if available_at >= duration_sec - 1e-9:
            break
        observation_start = available_at - window_sec
        requested_times = [
            observation_start + (sample_index + 0.5) * window_sec / input_frames
            for sample_index in range(input_frames)
        ]
        sampled_indices = [
            _nearest_frame(frame_times, requested, start_s=observation_start, end_s=available_at)
            for requested in requested_times
        ]
        valid = all(sampled_index is not None for sampled_index in sampled_indices)
        sampled_indices_int = [int(sampled_index) for sampled_index in sampled_indices if sampled_index is not None]
        sampled_frame_ids = [frame_ids[sampled_index] for sampled_index in sampled_indices_int]
        sampled_times = [frame_times[sampled_index] for sampled_index in sampled_indices_int]
        if valid and available_frame_id_set is not None:
            valid = all(frame_id in available_frame_id_set for frame_id in sampled_frame_ids)
        label = any(
            float(interval[0]) <= available_at < float(interval[1])
            for interval in record.get("anomaly_intervals_s", [])
        )
        windows.append(
            {
                "window_id": f"{record['video_id']}@{available_at:.3f}",
                "video_id": record["video_id"],
                "split": record["split"],
                "observation_start_s": observation_start,
                "available_at_s": available_at,
                "requested_times_s": requested_times,
                "sampled_frame_ids": sampled_frame_ids,
                "sampled_times_s": sampled_times,
                "label": int(label),
                "valid": valid and len(sampled_frame_ids) == input_frames,
                "failure_reason": (
                    None
                    if valid
                    else "missing_frame_files"
                    if available_frame_id_set is not None
                    and not all(frame_id in available_frame_id_set for frame_id in sampled_frame_ids)
                    else "frame_selection_failed"
                ),
            }
        )
        index += 1
    return windows


def generate_all_windows(records: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, Any]]:
    return [window for record in records for window in generate_windows(record, **kwargs)]
