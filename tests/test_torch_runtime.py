import pytest


@pytest.mark.gpu
def test_accelerator_can_allocate_tensor() -> None:
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        pytest.skip("CUDA or MPS is unavailable")

    tensor = torch.ones(1, device=device)
    assert tensor.device.type == device.type
