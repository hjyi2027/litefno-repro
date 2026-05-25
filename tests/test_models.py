import torch

from litefno.models import FNOS, LiteFNO


def test_litefno_forward_shape():
    model = LiteFNO(in_channels=2, out_channels=3, width=4, rank=2, layers=2)
    inputs = torch.randn(1, 2, 8, 8)
    outputs = model(inputs)
    assert outputs.shape == (1, 3, 8, 8)


def test_fno_s_forward_shape():
    model = FNOS(in_channels=2, out_channels=3, width=4, modes=4, layers=2)
    inputs = torch.randn(1, 2, 8, 8)
    outputs = model(inputs)
    assert outputs.shape == (1, 3, 8, 8)
