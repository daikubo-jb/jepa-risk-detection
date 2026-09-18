# The architecture follows the MIT-licensed Meta implementation at
# https://github.com/facebookresearch/vjepa2 (commit pinned in phase1.yaml).

"""Official V-JEPA 2 style masked prediction for the Phase 1 leak test.

The public Hugging Face model exposes ``context_mask`` only after the context
encoder has run.  This module keeps the Meta checkpoint parameter names while
applying the context token selection before every attention block.  It is
intentionally inference-only and implements the ViT-L/16 configuration used by
``artifacts/checkpoints/vitl.pt``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class PredictorError(ValueError):
    """Raised when the predictor contract or checkpoint is invalid."""


def _apply_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2 or mask.shape[0] != x.shape[0]:
        raise PredictorError(f"mask must have shape [B,K], got {tuple(mask.shape)} for {tuple(x.shape)}")
    indices = mask.to(device=x.device, dtype=torch.long).unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return torch.gather(x, dim=1, index=indices)


def _rotate(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Match Meta's RoPE implementation, including its published pair layout."""

    _, _, _, dim = x.shape
    if dim == 0 or dim % 2:
        raise PredictorError(f"rotary dimension must be a non-zero even value, got {dim}")
    omega = torch.arange(dim // 2, dtype=x.dtype, device=x.device)
    omega = 1.0 / (10000 ** (omega / (dim / 2.0)))
    freq = torch.einsum("...,f->...f", pos.to(dtype=x.dtype), omega)
    # This duplicated-frequency layout is retained for checkpoint compatibility.
    sin = freq.sin().repeat(*([1] * (freq.ndim - 1)), 2)
    cos = freq.cos().repeat(*([1] * (freq.ndim - 1)), 2)
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    rotated = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return x * cos + rotated * sin


class RoPEAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, *, grid_size: int = 16) -> None:
        super().__init__()
        if dim % num_heads:
            raise PredictorError("attention dimension must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.grid_size = grid_size
        # Meta splits each head into temporal, height, width rotary parts.
        self.d_dim = 2 * ((self.head_dim // 3) // 2)
        self.h_dim = 2 * ((self.head_dim // 3) // 2)
        self.w_dim = 2 * ((self.head_dim // 3) // 2)

    def _positions(
        self,
        ids: torch.Tensor,
        *,
        height: int | None,
        width: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height = self.grid_size if height is None else int(height)
        width = self.grid_size if width is None else int(width)
        per_frame = height * width
        frame = ids // per_frame
        within = ids - frame * per_frame
        row = within // width
        col = within - row * width
        return frame.float(), row.float() * self.grid_size / height, col.float() * self.grid_size / width

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if mask is None:
            per_frame = (self.grid_size if height is None else int(height)) * (
                self.grid_size if width is None else int(width)
            )
            ids = torch.arange(tokens, device=x.device) + 0
            # Full sequences are laid out as contiguous T,H,W tokens.
            if tokens % per_frame:
                raise PredictorError(f"full token count {tokens} is not divisible by spatial grid {per_frame}")
            ids = ids.unsqueeze(0).expand(batch, -1)
        else:
            ids = mask.to(device=x.device, dtype=torch.long)
        frame, row, col = self._positions(ids, height=height, width=width)
        frame = frame.unsqueeze(1).expand(-1, self.num_heads, -1)
        row = row.unsqueeze(1).expand(-1, self.num_heads, -1)
        col = col.unsqueeze(1).expand(-1, self.num_heads, -1)
        parts_q: list[torch.Tensor] = []
        parts_k: list[torch.Tensor] = []
        offset = 0
        for size, position in ((self.d_dim, frame), (self.h_dim, row), (self.w_dim, col)):
            parts_q.append(_rotate(q[..., offset : offset + size], position))
            parts_k.append(_rotate(k[..., offset : offset + size], position))
            offset += size
        if offset < self.head_dim:
            parts_q.append(q[..., offset:])
            parts_k.append(k[..., offset:])
        q = torch.cat(parts_q, dim=-1)
        k = torch.cat(parts_k, dim=-1)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(attended.transpose(1, 2).reshape(batch, tokens, channels))


class MLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, *, grid_size: int = 16) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = RoPEAttention(dim, num_heads, grid_size=grid_size)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim)

    def forward(self, x: torch.Tensor, *, mask: torch.Tensor | None = None, height: int = 16, width: int = 16) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask, height=height, width=width)
        return x + self.mlp(self.norm2(x))


class PatchEmbed3D(nn.Module):
    def __init__(self, *, embed_dim: int = 1024, patch_size: int = 16, tubelet_size: int = 2) -> None:
        super().__init__()
        self.proj = nn.Conv3d(3, embed_dim, kernel_size=(tubelet_size, patch_size, patch_size), stride=(tubelet_size, patch_size, patch_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise PredictorError(f"video input must be [B,C,T,H,W], got {tuple(x.shape)}")
        return self.proj(x).flatten(2).transpose(1, 2)


class MaskedVideoEncoder(nn.Module):
    """ViT-L/16 encoder with optional pre-attention token selection."""

    def __init__(self, *, depth: int = 24, embed_dim: int = 1024, num_heads: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed3D(embed_dim=embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor, *, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.patch_embed(x)
        batch, tokens, _ = x.shape
        height = width = 16
        if mask is not None:
            x = _apply_mask(x, mask)
        for block in self.blocks:
            x = block(x, mask=mask, height=height, width=width)
        return self.norm(x)


class MaskedVideoPredictor(nn.Module):
    """ViT-L predictor compatible with the official ``vitl.pt`` state dict."""

    def __init__(self, *, depth: int = 12, embed_dim: int = 1024, predictor_dim: int = 384, num_heads: int = 12) -> None:
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_dim, bias=True)
        self.mask_tokens = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, predictor_dim)) for _ in range(10)])
        self.predictor_blocks = nn.ModuleList([Block(predictor_dim, num_heads) for _ in range(depth)])
        self.predictor_norm = nn.LayerNorm(predictor_dim, eps=1e-6)
        self.predictor_proj = nn.Linear(predictor_dim, embed_dim, bias=True)

    def forward(
        self,
        context: torch.Tensor,
        context_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        mask_index: int = 1,
    ) -> torch.Tensor:
        batch = context.shape[0]
        context = self.predictor_embed(context)
        target = self.mask_tokens[mask_index % len(self.mask_tokens)].expand(batch, target_ids.shape[1], -1)
        joined = torch.cat([context, target], dim=1)
        ids = torch.cat([context_ids, target_ids], dim=1).to(device=joined.device, dtype=torch.long)
        order = torch.argsort(ids, dim=1)
        joined = torch.gather(joined, 1, order.unsqueeze(-1).expand(-1, -1, joined.shape[-1]))
        sorted_ids = torch.gather(ids, 1, order)
        for block in self.predictor_blocks:
            joined = block(joined, mask=sorted_ids, height=16, width=16)
        joined = self.predictor_norm(joined)
        # sorted_ids places target IDs after context IDs for the initial mask.
        target_mask = sorted_ids >= 24 * 16 * 16
        if not torch.all(target_mask.sum(dim=1) == target_ids.shape[1]):
            raise PredictorError("target token IDs do not match the configured context boundary")
        target_hidden = joined[target_mask].reshape(batch, target_ids.shape[1], -1)
        return self.predictor_proj(target_hidden)


@dataclass
class PredictorBundle:
    encoder: MaskedVideoEncoder
    target_encoder: MaskedVideoEncoder
    predictor: MaskedVideoPredictor
    checkpoint: str


def load_official_bundle(
    checkpoint: Path | str,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> PredictorBundle:
    """Load only encoder/predictor tensors from Meta's checkpoint safely."""

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise PredictorError(f"predictor checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("encoder"), dict)
        or not isinstance(payload.get("target_encoder"), dict)
        or not isinstance(payload.get("predictor"), dict)
    ):
        raise PredictorError("checkpoint must contain tensor dictionaries named encoder, target_encoder, and predictor")

    def strip_prefix(values: dict[str, Any]) -> dict[str, Any]:
        prefix = "module.backbone."
        if not all(str(key).startswith(prefix) for key in values):
            raise PredictorError("checkpoint state dict has an unexpected key prefix")
        return {str(key)[len(prefix) :]: value for key, value in values.items()}

    encoder = MaskedVideoEncoder()
    target_encoder = MaskedVideoEncoder()
    predictor = MaskedVideoPredictor()
    try:
        encoder.load_state_dict(strip_prefix(payload["encoder"]), strict=True)
        target_encoder.load_state_dict(strip_prefix(payload["target_encoder"]), strict=True)
        predictor.load_state_dict(strip_prefix(payload["predictor"]), strict=True)
    except (RuntimeError, KeyError) as exc:
        raise PredictorError(f"checkpoint does not match ViT-L predictor architecture: {exc}") from exc
    encoder = encoder.to(device=device, dtype=dtype).eval()
    target_encoder = target_encoder.to(device=device, dtype=dtype).eval()
    predictor = predictor.to(device=device, dtype=dtype).eval()
    return PredictorBundle(
        encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        checkpoint=str(checkpoint_path),
    )


def context_target_ids(*, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fixed 24/8 temporal-token mask for 64 frames."""

    ids = torch.arange(32 * 16 * 16, device=device, dtype=torch.long).reshape(32, 16 * 16)
    context = ids[:24].reshape(1, -1)
    target = ids[24:].reshape(1, -1)
    if torch.unique(torch.cat([context.reshape(-1), target.reshape(-1)])).numel() != 8192:
        raise PredictorError("context/target IDs do not cover all 8192 tokens")
    return context, target


def masked_prediction_error(bundle: PredictorBundle, pixel_values_videos: torch.Tensor) -> torch.Tensor:
    """Return one FP32 mean-L1 score per video window."""

    if pixel_values_videos.ndim != 5:
        raise PredictorError("pixel_values_videos must be [B,T,C,H,W]")
    target_device = next(bundle.encoder.parameters()).device
    model_dtype = next(bundle.encoder.parameters()).dtype
    videos = pixel_values_videos.permute(0, 2, 1, 3, 4).to(device=target_device, dtype=model_dtype)
    context_ids, target_ids = context_target_ids(device=target_device)
    context_ids = context_ids.expand(videos.shape[0], -1)
    target_ids = target_ids.expand(videos.shape[0], -1)
    with torch.inference_mode():
        full = bundle.target_encoder(videos)
        context = bundle.encoder(videos, mask=context_ids)
        predicted = bundle.predictor(context, context_ids, target_ids, mask_index=1)
        target = full.gather(1, target_ids.unsqueeze(-1).expand(-1, -1, full.shape[-1]))
        target = F.layer_norm(target, (target.shape[-1],))
        score = torch.abs(predicted.float() - target.float()).mean(dim=(1, 2))
    if not torch.isfinite(score).all():
        raise PredictorError("predictor produced non-finite scores")
    return score


def predict_masked_tokens(bundle: PredictorBundle, pixel_values_videos: torch.Tensor) -> torch.Tensor:
    """Return predictor output for the fixed target IDs (for leakage tests)."""

    if pixel_values_videos.ndim != 5:
        raise PredictorError("pixel_values_videos must be [B,T,C,H,W]")
    target_device = next(bundle.encoder.parameters()).device
    model_dtype = next(bundle.encoder.parameters()).dtype
    videos = pixel_values_videos.permute(0, 2, 1, 3, 4).to(device=target_device, dtype=model_dtype)
    context_ids, target_ids = context_target_ids(device=target_device)
    context_ids = context_ids.expand(videos.shape[0], -1)
    target_ids = target_ids.expand(videos.shape[0], -1)
    with torch.inference_mode():
        context = bundle.encoder(videos, mask=context_ids)
        predicted = bundle.predictor(context, context_ids, target_ids, mask_index=1)
    return predicted.float()
