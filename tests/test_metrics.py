import torch

from litefno.metrics import vrmse, windowed_vrmse


def test_vrmse_zero():
    target = torch.ones(2, 4, 3)
    pred = target.clone()
    result = vrmse(pred, target)
    assert torch.allclose(result, torch.zeros_like(result))


def test_windowed_vrmse_keys():
    pred = torch.zeros(2, 40, 3)
    target = torch.ones(2, 40, 3)
    metrics = windowed_vrmse(pred, target)
    assert "vrmse_6_12" in metrics
    assert "vrmse_13_30" in metrics
