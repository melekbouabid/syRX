"""Optuna search spaces of the classifiers (src/models/classification/hpo.py), no PyTorch."""

from __future__ import annotations

import copy
from pathlib import Path

import optuna
import pytest

from src.models.classification import data as D
from src.models.classification import hpo

FAKE = Path(__file__).parent / "configs" / "classifier_fake.yaml"
COMMITTED = sorted((D.ROOT / "config" / "hpo").glob("classifier_*.yaml"))


@pytest.mark.parametrize("path", [FAKE, *COMMITTED], ids=lambda p: Path(p).stem)
def test_configs_are_valid_and_have_a_smoke_mode(path: Path) -> None:
    config = hpo.load_config(path)
    assert config["search"]["metric"].startswith("val_")
    smoke = hpo.apply_smoke(config)
    assert smoke["smoke"]["enabled"] is True
    assert smoke["search"]["n_trials"] == 2 and smoke["final"]["epochs"] == 1
    assert "timeout_hours" not in smoke["search"] and "max_hours" not in smoke["final"]
    for name in config["smoke"].get("fixed", {}):
        assert name not in smoke["search"]["space"]


@pytest.mark.parametrize("path", COMMITTED, ids=lambda p: Path(p).stem)
def test_committed_configs_search_the_hyperparameters_of_the_plan(path: Path) -> None:
    config = hpo.load_config(path)
    arch = path.stem.removeprefix("classifier_")
    assert config["arch"] == config["model"] == arch
    assert config["branch"] == f"experiment/classifier-{arch}"
    space, fixed = config["search"]["space"], config.get("fixed") or {}
    # lr, weight decay, unfrozen layers, image size, augmentation, imbalance (PLAN 2.3).
    for name in ("lr", "weight_decay", "randaugment", "mix_prob", "imbalance"):
        assert name in space, name
    assert "imgsz" in space or "imgsz" in fixed
    if config.get("pretrained", True):
        assert "unfreeze" in space
    assert set(space["imbalance"]["choices"]) == {"none", "class_weights", "focal"}
    assert float(config["search"]["timeout_hours"]) + float(config["final"]["max_hours"]) <= 4


def test_config_errors(tmp_path: Path) -> None:
    good = hpo.load_config(FAKE)
    cases = [
        ({"family": "keras"}, "family"),
        ({"search": {**good["search"], "metric": "test_top1"}}, "validation"),
        ({"fixed": {**good["fixed"], "lr": 0.1}}, "both fixed and searched"),
        ({"final": {"epochs": 0}}, "final"),
    ]
    for change, message in cases:
        bad = {**copy.deepcopy(good), **change}
        with pytest.raises(hpo.ConfigError, match=message):
            hpo.validate_config(bad)
    no_size = copy.deepcopy(good)
    no_size["search"]["space"].pop("imgsz")
    with pytest.raises(hpo.ConfigError, match="imgsz"):
        hpo.validate_config(no_size)
    wrong = copy.deepcopy(good)
    wrong["search"]["space"]["imbalance"]["choices"] = ["none", "oversample"]
    with pytest.raises(hpo.ConfigError, match="imbalance"):
        hpo.validate_config(wrong)


def test_trial_and_final_hyperparameters() -> None:
    config = hpo.load_config(FAKE)
    study = hpo.create_study(config)
    trial = study.ask()
    suggested = hpo.suggest(trial, config["search"]["space"])
    hp = hpo.trial_hyperparameters(config, suggested)
    assert hp["batch"] == 4 and hp["head_epochs"] == 1  # fixed values reach the training
    assert 1e-5 <= hp["lr"] <= 1e-1
    assert hpo.final_hyperparameters(config, hp)["head_epochs"] == 2
    assert isinstance(hpo.sampler(config), optuna.samplers.TPESampler)
