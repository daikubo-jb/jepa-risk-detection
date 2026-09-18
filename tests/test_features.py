from __future__ import annotations

import torch

from jepa_risk.features import extract_pooled_features


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def get_vision_features(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0]
        temporal = torch.arange(32, dtype=torch.float32).view(1, 32, 1, 1, 1)
        tokens = temporal.expand(batch, 32, 16, 16, 1024).reshape(batch, 8192, 1024)
        return tokens.to(device=inputs.device, dtype=inputs.dtype)


def test_pooled_features_use_temporal_and_spatial_axes() -> None:
    model = _DummyModel()
    result = extract_pooled_features(model, torch.zeros(2, 64, 3, 256, 256))
    assert tuple(result.global_features.shape) == (2, 1024)
    assert tuple(result.temporal_features.shape) == (2, 32, 1024)
    assert result.global_features.dtype == torch.float16
    assert result.temporal_features.dtype == torch.float16
    assert torch.all(result.temporal_features[0, :, 0] == torch.arange(32, dtype=torch.float16))
    assert torch.all(result.global_features == torch.tensor(15.5, dtype=torch.float16))
