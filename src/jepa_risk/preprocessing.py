from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import av
import numpy as np
import torch
from PIL import Image


class PreprocessingError(ValueError):
    """Raised when a video window cannot be decoded or prepared safely."""


def decode_video_frames(video_path: Path | str, frame_ids: Sequence[int]) -> list[np.ndarray]:
    """Decode the requested zero-based frame indices as RGB uint8 arrays.

    PyAV is used instead of the removed ``torchvision.io.read_video`` API. The
    requested IDs may be duplicated or out of order; decoding is performed in
    ascending order and the result is returned in the original order.
    """

    requested = [int(frame_id) for frame_id in frame_ids]
    if not requested:
        raise PreprocessingError("at least one frame is required")
    if any(frame_id < 0 for frame_id in requested):
        raise PreprocessingError("frame IDs must be non-negative")
    unique_ids = sorted(set(requested))
    decoded: dict[int, np.ndarray] = {}
    try:
        container = av.open(str(video_path))
    except (OSError, av.error.FFmpegError) as exc:
        raise PreprocessingError(f"cannot open video: {video_path}") from exc
    try:
        stream = next((candidate for candidate in container.streams if candidate.type == "video"), None)
        if stream is None:
            raise PreprocessingError(f"video stream not found: {video_path}")
        wanted = set(unique_ids)
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                decoded[index] = frame.to_rgb().to_ndarray()
            if len(decoded) == len(wanted):
                break
    except (OSError, av.error.FFmpegError) as exc:
        raise PreprocessingError(f"cannot decode video: {video_path}") from exc
    finally:
        container.close()
    missing = [frame_id for frame_id in unique_ids if frame_id not in decoded]
    if missing:
        raise PreprocessingError(f"requested frames are missing from {video_path}: {missing[:8]}")
    return [decoded[frame_id] for frame_id in requested]


def letterbox_frame(frame: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Resize an RGB frame while preserving aspect ratio and add black padding."""

    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise PreprocessingError("frame must have shape [height, width, 3]")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise PreprocessingError("frame dimensions must be positive")
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating) and np.nanmin(frame) >= 0 and np.nanmax(frame) <= 1:
            frame = np.round(frame * 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    target_size = int(target_size)
    if target_size <= 0:
        raise PreprocessingError("target_size must be positive")

    height, width = frame.shape[:2]
    scale = min(target_size / width, target_size / height)
    resized_width = min(target_size, max(1, int(np.floor(width * scale + 0.5))))
    resized_height = min(target_size, max(1, int(np.floor(height * scale + 0.5))))
    image = Image.fromarray(frame)
    image = image.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    resized = np.asarray(image, dtype=np.uint8)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    left = (target_size - resized_width) // 2
    top = (target_size - resized_height) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas


def preprocess_video_frames(
    frames: Sequence[np.ndarray],
    processor: Any,
    *,
    target_size: int = 256,
) -> torch.Tensor:
    """Letterbox frames and apply the official V-JEPA2 processor normalization."""

    if not frames:
        raise PreprocessingError("at least one frame is required")
    letterboxed = np.stack([letterbox_frame(frame, target_size=target_size) for frame in frames], axis=0)
    encoded = processor(
        [letterboxed],
        do_resize=False,
        do_center_crop=False,
        input_data_format="channels_last",
        return_tensors="pt",
    )
    try:
        tensor = encoded["pixel_values_videos"]
    except (KeyError, TypeError) as exc:
        raise PreprocessingError("V-JEPA2 processor did not return pixel_values_videos") from exc
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 5:
        raise PreprocessingError("processor output must have shape [batch, time, channels, height, width]")
    expected = (1, len(frames), 3, target_size, target_size)
    if tuple(tensor.shape) != expected:
        raise PreprocessingError(f"unexpected processor shape {tuple(tensor.shape)}, expected {expected}")
    if not torch.isfinite(tensor).all():
        raise PreprocessingError("processor output contains non-finite values")
    return tensor


def load_video_window(
    video_path: Path | str,
    frame_ids: Sequence[int],
    processor: Any,
    *,
    target_size: int = 256,
) -> torch.Tensor:
    frames = decode_video_frames(video_path, frame_ids)
    return preprocess_video_frames(frames, processor, target_size=target_size)
