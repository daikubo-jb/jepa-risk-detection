from __future__ import annotations

import torch

from jepa_risk.predictor import MaskedVideoPredictor, context_target_ids


def test_context_target_ids_cover_the_full_24_8_mask() -> None:
    context, target = context_target_ids()
    assert context.shape == (1, 24 * 16 * 16)
    assert target.shape == (1, 8 * 16 * 16)
    assert set(context.flatten().tolist()).isdisjoint(set(target.flatten().tolist()))
    assert torch.equal(context.flatten(), torch.arange(6144))
    assert torch.equal(target.flatten(), torch.arange(6144, 8192))


def test_predictor_returns_only_target_tokens_in_id_order() -> None:
    model = MaskedVideoPredictor(depth=1, embed_dim=24, predictor_dim=24, num_heads=3).eval()
    context = torch.randn(1, 2, 24)
    context_ids = torch.tensor([[0, 1]])
    target_ids = torch.tensor([[6144, 6145]])
    output = model(context, context_ids, target_ids, mask_index=1)
    assert output.shape == (1, 2, 24)
    assert torch.isfinite(output).all()
