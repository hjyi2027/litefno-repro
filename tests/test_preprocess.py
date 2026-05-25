import numpy as np
import h5py

from litefno.preprocess import cap_trajectories, downsample_spatial, preprocess_h5_file


def test_preprocess_h5_file(tmp_path):
    input_path = tmp_path / "input.h5"
    output_path = tmp_path / "output.h5"
    data = np.random.rand(2, 10, 8, 8, 3).astype(np.float32)
    with h5py.File(input_path, "w") as handle:
        handle.create_dataset("data", data=data)

    preprocess_h5_file(
        input_path=input_path,
        output_path=output_path,
        dataset_key="data",
        factor=2,
        max_trajectories=1,
        max_steps=5,
    )

    with h5py.File(output_path, "r") as handle:
        processed = handle["data"][...]

    assert processed.shape == (1, 5, 4, 4, 3)


def test_downsample_spatial_block_mean():
    data = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4, 1)
    downsampled = downsample_spatial(data, factor=2)
    expected = np.array([[[[[2.5], [4.5]], [[10.5], [12.5]]]]], dtype=np.float32)
    assert downsampled.shape == (1, 1, 2, 2, 1)
    assert np.allclose(downsampled, expected)


def test_cap_trajectories_seeded_sampling():
    data = np.arange(10, dtype=np.float32).reshape(10, 1, 1, 1, 1)
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    sample_a = cap_trajectories(data, max_trajectories=4, rng=rng_a)
    sample_b = cap_trajectories(data, max_trajectories=4, rng=rng_b)
    assert np.array_equal(sample_a, sample_b)
    assert sample_a.shape[0] == 4
