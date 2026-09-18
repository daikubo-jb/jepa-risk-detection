from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import VJEPA2Model, VJEPA2VideoProcessor


class FeatureExtractionError(ValueError):
    """Raised when V-JEPA2 cannot produce the expected feature contract."""


@dataclass(frozen=True)
class PooledFeatures:
    global_features: torch.Tensor
    temporal_features: torch.Tensor


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype: str | torch.dtype = "float32") -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        resolved = getattr(torch, str(dtype).replace("torch.", ""))
    except AttributeError as exc:
        raise FeatureExtractionError(f"unsupported dtype: {dtype}") from exc
    if not isinstance(resolved, torch.dtype):
        raise FeatureExtractionError(f"unsupported dtype: {dtype}")
    return resolved


def load_vjepa2(
    model_id: str = "facebook/vjepa2-vitl-fpc64-256",
    *,
    device: str = "auto",
    dtype: str | torch.dtype = "float32",
    local_files_only: bool = False,
    revision: str | None = None,
) -> tuple[VJEPA2Model, VJEPA2VideoProcessor, torch.device]:
    """Load the pinned HF V-JEPA2 encoder and its official video processor."""

    kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if revision is not None:
        kwargs["revision"] = revision
    processor = VJEPA2VideoProcessor.from_pretrained(model_id, **kwargs)
    model = VJEPA2Model.from_pretrained(model_id, **kwargs)
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype)
    model = model.to(device=resolved_device)
    if resolved_dtype != torch.float32:
        model = model.to(dtype=resolved_dtype)
    model.eval()
    return model, processor, resolved_device


def extract_pooled_features(
    model: VJEPA2Model,
    pixel_values_videos: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> PooledFeatures:
    """Extract FP16 global and temporal pools from the official vision tokens."""

    if pixel_values_videos.ndim != 5:
        raise FeatureExtractionError("pixel_values_videos must have shape [batch, time, channels, height, width]")
    model_device = next(model.parameters()).device
    target_device = torch.device(device) if device is not None else model_device
    model_dtype = next(model.parameters()).dtype
    inputs = pixel_values_videos.to(device=target_device, dtype=model_dtype)
    with torch.inference_mode():
        tokens = model.get_vision_features(inputs)
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
        raise FeatureExtractionError("get_vision_features must return [batch, tokens, hidden]")
    batch, token_count, hidden = tokens.shape
    if token_count != 8192:
        raise FeatureExtractionError(f"expected 8192 vision tokens, got {token_count}")
    if not torch.isfinite(tokens).all():
        raise FeatureExtractionError("vision features contain non-finite values")
    token_grid = tokens.reshape(batch, 32, 16, 16, hidden).float()
    global_features = token_grid.mean(dim=(1, 2, 3)).to(dtype=output_dtype)
    temporal_features = token_grid.mean(dim=(2, 3)).to(dtype=output_dtype)
    if tuple(global_features.shape) != (batch, hidden) or tuple(temporal_features.shape) != (batch, 32, hidden):
        raise FeatureExtractionError("unexpected pooled feature shapes")
    return PooledFeatures(global_features=global_features, temporal_features=temporal_features)
