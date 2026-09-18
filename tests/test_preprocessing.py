from __future__ import annotations

import numpy as np
import torch

from jepa_risk.preprocessing import letterbox_frame, preprocess_video_frames


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls: dict[str, object] = {}

    def __call__(self, videos, **kwargs):
        self.calls = kwargs
        array = np.asarray(videos[0])
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2).unsqueeze(0).float()
        return {"pixel_values_videos": tensor / 255.0}


def test_letterbox_wide_preserves_content_and_pads_vertically() -> None:
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    frame[:, :, 0] = 255
    result = letterbox_frame(frame, target_size=8)
    assert result.shape == (8, 8, 3)
    assert np.all(result[:2] == 0)
    assert np.all(result[6:] == 0)
    assert np.all(result[2:6, :, 0] == 255)


def test_letterbox_tall_adds_left_and_right_padding() -> None:
    frame = np.zeros((4, 2, 3), dtype=np.uint8)
    frame[:, :, 1] = 200
    result = letterbox_frame(frame, target_size=8)
    assert np.all(result[:, :2] == 0)
    assert np.all(result[:, 6:] == 0)
    assert np.all(result[:, 2:6, 1] == 200)


def test_processor_is_called_without_second_resize_or_crop() -> None:
    processor = _FakeProcessor()
    frames = [np.full((3, 5, 3), 128, dtype=np.uint8) for _ in range(4)]
    result = preprocess_video_frames(frames, processor, target_size=8)
    assert tuple(result.shape) == (1, 4, 3, 8, 8)
    assert torch.isfinite(result).all()
    assert processor.calls["do_resize"] is False
    assert processor.calls["do_center_crop"] is False
    assert processor.calls["input_data_format"] == "channels_last"
