import pytest
import torch

from litefno.train import flatten_time, normalize_windows, unflatten_time


def test_flatten_unflatten_roundtrip():
    tensor = torch.randn(2, 3, 4, 5, 6)
    flattened = flatten_time(tensor)
    assert flattened.shape == (2, 18, 4, 5)
    restored = unflatten_time(flattened, time_steps=3, channels=6)
    assert torch.allclose(restored, tensor)


def test_normalize_windows_filters_by_output_steps():
    windows = [(0, 2), (6, 12)]
    normalized = normalize_windows(windows, output_steps=10)
    assert normalized == [(0, 2)]


def test_normalize_windows_requires_end_after_start():
    with pytest.raises(ValueError):
        normalize_windows([(5, 5)], output_steps=10)
