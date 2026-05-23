import numpy as np
import h5py

from litefno.preprocess import preprocess_h5_file


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
