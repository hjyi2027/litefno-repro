import pytest

from litefno.config import apply_overrides, load_config


def test_load_config_with_base(tmp_path):
    base_path = tmp_path / "base.yaml"
    child_path = tmp_path / "child.yaml"
    base_path.write_text(
        "dataset:\n  name: demo\ntraining:\n  epochs: 10\n",
        encoding="utf-8",
    )
    child_path.write_text(
        "base_config: base.yaml\ntraining:\n  epochs: 5\n",
        encoding="utf-8",
    )
    config = load_config(child_path)
    assert config["dataset"]["name"] == "demo"
    assert config["training"]["epochs"] == 5


def test_apply_overrides_nested_values():
    config = {"model": {"width": 64}, "training": {}}
    updated = apply_overrides(config, ["model.width=128", "training.epochs=20"])
    assert updated["model"]["width"] == 128
    assert updated["training"]["epochs"] == 20


def test_apply_overrides_requires_equals():
    with pytest.raises(ValueError):
        apply_overrides({}, ["model.width"])
