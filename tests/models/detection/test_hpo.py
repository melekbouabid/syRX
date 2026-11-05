"""Optuna search spaces (src/models/detection/hpo.py) and the configs of config/hpo/."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import optuna
import pytest

from src.models.detection import hpo

ROOT = Path(__file__).resolve().parents[3]
FAKE = Path(__file__).parent / "configs" / "detector_fake.yaml"
CONFIGS = sorted((ROOT / "config" / "hpo").glob("detector_*.yaml"))
# What step 2.2 asks every model's search to cover (PLAN: lr0, imgsz, batch, optimiser,
# augmentation).
REQUIRED_PARAMETERS = {"lr0", "imgsz", "batch", "optimizer"}
AUGMENTATIONS = {"mosaic", "mixup", "hsv_h", "hsv_s", "hsv_v", "fliplr", "flipud"}

optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture
def config() -> dict[str, Any]:
    return hpo.load_config(FAKE)


def sample(config: dict[str, Any], n: int = 6) -> list[dict[str, Any]]:
    study = hpo.create_study(config)
    out = []
    for _ in range(n):
        trial = study.ask()
        out.append(hpo.suggest(trial, config["search"]["space"]))
        study.tell(trial, 0.5)
    return out


def test_suggestions_stay_in_the_space(config: dict[str, Any]) -> None:
    for values in sample(config):
        assert list(values) == list(config["search"]["space"])  # YAML order
        assert values["optimizer"] in ("SGD", "AdamW")
        low, high = config["search"]["space"]["lr0"]["ranges"][values["optimizer"]]
        assert low <= values["lr0"] <= high
        assert values["imgsz"] in (64, 96)
        assert values["batch"] in (2, 4, 6, 8)
        assert 0.0 <= values["mosaic"] <= 1.0


def test_dependent_parameter_has_one_optuna_name_per_branch(config: dict[str, Any]) -> None:
    trial = optuna.trial.FixedTrial(
        {"optimizer": "AdamW", "lr0_adamw": 1e-3, "imgsz": 64, "batch": 4, "mosaic": 0.5}
    )
    values = hpo.suggest(trial, config["search"]["space"])
    assert values["lr0"] == 1e-3
    assert values["optimizer"] == "AdamW"


def test_the_seeded_sampler_is_reproducible(config: dict[str, Any]) -> None:
    assert sample(config) == sample(config)


def test_trial_hyperparameters_add_the_fixed_values(config: dict[str, Any]) -> None:
    hp = hpo.trial_hyperparameters(config, {"lr0": 0.01})
    assert hp == {"workers": 0, "lr0": 0.01}


def test_smoke_overrides(config: dict[str, Any]) -> None:
    smoke = hpo.apply_smoke(config)
    assert smoke["search"]["n_trials"] == 2
    assert smoke["search"]["epochs"] == 1
    assert smoke["final"]["epochs"] == 1
    assert smoke["smoke"]["enabled"] is True
    space = smoke["search"]["space"]
    assert "imgsz" not in space and "optimizer" not in space
    # lr0 followed the optimizer, now fixed to SGD: it keeps the SGD range.
    assert (space["lr0"]["low"], space["lr0"]["high"]) == (1e-3, 1e-1)
    assert smoke["fixed"] == {"workers": 0, "imgsz": 64, "optimizer": "SGD"}
    assert config["search"]["n_trials"] == 3  # the original is unchanged


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c.pop("weights"), "missing keys"),
        (lambda c: c.update(family="detr"), "family"),
        (lambda c: c["search"].update(metric="test_mAP50_95"), "validation"),
        (lambda c: c["search"].update(fraction=0), "fraction"),
        (lambda c: c["search"].update(n_trials=0), "n_trials"),
        (lambda c: c["search"]["space"].update(x={"type": "float", "low": 1, "high": 1}), "low"),
        (
            lambda c: c["search"]["space"].update(
                x={"type": "float", "low": 0, "high": 1, "log": True}
            ),
            "log",
        ),
        (lambda c: c["search"]["space"].update(x={"type": "list"}), "type"),
        (lambda c: c["search"]["space"].update(x={"type": "categorical"}), "choices"),
        (lambda c: c["search"]["space"]["lr0"]["ranges"].pop("AdamW"), "no range"),
        (lambda c: c["search"]["space"]["lr0"].update(depends_on="mosaic"), "categorical"),
        (lambda c: c["final"].update(epochs=0), "final"),
    ],
)
def test_invalid_configs_are_refused(config: dict[str, Any], change: Any, message: str) -> None:
    broken = copy.deepcopy(config)
    change(broken)
    with pytest.raises(hpo.ConfigError, match=message):
        hpo.validate_config(broken)


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_project_configs_cover_the_required_search(path: Path) -> None:
    """Every config/hpo/detector_<arch>.yaml is valid, searches lr0, imgsz, batch, the
    optimiser and at least one augmentation, and has a smoke section."""
    config = hpo.load_config(path)
    space = set(config["search"]["space"])
    assert path.stem == f"detector_{config['arch']}"
    assert config["branch"] == f"experiment/detector-{config['arch']}"
    assert space >= REQUIRED_PARAMETERS
    assert space & AUGMENTATIONS
    assert config["search"]["n_trials"] >= 8
    smoke = hpo.apply_smoke(config)
    assert smoke["search"]["n_trials"] <= 2
    assert sample(config, 3)
