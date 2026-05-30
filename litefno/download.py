from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import load_config, resolve_path


def download_dataset(dataset: str, split: str, base_path: Path) -> None:
    if shutil.which("the-well-download") is None:
        raise RuntimeError("the-well-download is not available on PATH.")
    command = [
        "the-well-download",
        "--base-path",
        str(base_path),
        "--dataset",
        dataset,
        "--split",
        split,
    ]
    subprocess.run(command, check=True)


def download_from_config(config_path: Path) -> None:
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    dataset_name = dataset_cfg["name"]
    base_path = resolve_path(dataset_cfg, "input_dir")
    for split in dataset_cfg.get("splits", ["train", "valid", "test"]):
        download_dataset(dataset_name, split, base_path)
