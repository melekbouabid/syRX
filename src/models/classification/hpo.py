"""Hyperparameter search of a classifier with Optuna (config/hpo/classifier_<arch>.yaml).

One YAML file per experiment branch describes the model and its search, like the detectors
(src/models/detection/hpo.py, whose parameter types and `suggest` are reused):

    arch: resnet50                    # name of the runs, tags and Registry description
    model: resnet50                   # src/models/architecture.py ARCHITECTURES key
    family: torch                     # training code (src/models/classification/torch_backend)
    pretrained: true                  # ImageNet weights (transfer learning); baseline: false
    branch: experiment/classifier-resnet50
    seed: 0
    search:                           # each trial: a short training on a share of train
      n_trials: 10
      epochs: 6
      fraction: 0.4                   # share of the train crops of every class
      metric: val_top1                # maximised (validation split, never test)
      sampler: tpe                    # tpe | random, seeded with `seed`
      startup_trials: 4               # random trials before TPE models the results
      timeout_hours: 1.5              # the search stops in time (Kaggle GPU budget)
      space:
        lr: {type: float, low: 1.0e-4, high: 3.0e-3, log: true}
        unfreeze: {type: categorical, choices: [1, 2, 3, 5]}
        ...
    fixed: {batch: 64, workers: 4, head_epochs: 1}   # passed to every training as is
    final: {epochs: 30, patience: 8, head_epochs: 2, max_hours: 1.5}
    smoke: {n_trials: 2, epochs: 1, final_epochs: 1, images: {...}, fixed: {...}}

Hyperparameters the training understands (torch_backend.train): lr, weight_decay,
backbone_lr_mult, unfreeze (backbone blocks fine-tuned after the head-only epochs),
head_epochs, imgsz, batch, randaugment (magnitude, 0 = off), randaugment_ops, mix_prob
(share of the batches mixed by MixUp or CutMix), mixup_alpha, cutmix_alpha, imbalance (none |
class_weights | focal), focal_gamma, class_weight_power, label_smoothing, workers, and the
architecture options (baseline: width, dropout).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import optuna
import yaml

from src.models.detection.hpo import ConfigError, _check_param, suggest

__all__ = [
    "ConfigError",
    "apply_smoke",
    "create_study",
    "final_hyperparameters",
    "load_config",
    "sampler",
    "suggest",
    "trial_hyperparameters",
    "validate_config",
]

FAMILIES = ("torch",)
REQUIRED = ("arch", "model", "family", "seed", "search", "final")
SEARCH_REQUIRED = ("n_trials", "epochs", "metric", "space")
IMBALANCE = ("none", "class_weights", "focal")


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED if key not in config]
    if missing:
        raise ConfigError(f"missing keys: {missing}")
    if config["family"] not in FAMILIES:
        raise ConfigError(f"family must be one of {FAMILIES}, got {config['family']!r}")
    search = config["search"]
    missing = [key for key in SEARCH_REQUIRED if key not in search]
    if missing:
        raise ConfigError(f"search: missing keys {missing}")
    if int(search["n_trials"]) < 1 or int(search["epochs"]) < 1:
        raise ConfigError("search: n_trials and epochs must be at least 1")
    if not 0 < float(search.get("fraction", 1.0)) <= 1:
        raise ConfigError("search: fraction must be in (0, 1]")
    if not str(search["metric"]).startswith("val_"):
        raise ConfigError("search: the metric is measured on the validation split (val_*)")
    if search.get("sampler", "tpe") not in ("tpe", "random"):
        raise ConfigError("search: sampler must be tpe or random")
    space = search["space"]
    if not space:
        raise ConfigError("search: empty space")
    for name, spec in space.items():
        _check_param(name, spec, space)
    fixed = config.get("fixed") or {}
    overlap = sorted(set(fixed) & set(space))
    if overlap:
        raise ConfigError(f"{overlap} are both fixed and searched")
    if "imgsz" not in space and "imgsz" not in fixed:
        raise ConfigError("imgsz must be fixed or searched")
    choices = set(space.get("imbalance", {}).get("choices", [])) | (
        {fixed["imbalance"]} if "imbalance" in fixed else set()
    )
    if choices - set(IMBALANCE):
        raise ConfigError(f"imbalance must be one of {IMBALANCE}")
    final = config["final"]
    if int(final.get("epochs", 0)) < 1:
        raise ConfigError("final: epochs must be at least 1")


def apply_smoke(config: dict[str, Any]) -> dict[str, Any]:
    """The configuration of the local smoke test: `smoke` overrides, `smoke.enabled=True`."""
    out = copy.deepcopy(config)
    smoke = config.get("smoke") or {}
    search = out["search"]
    search["n_trials"] = int(smoke.get("n_trials", 2))
    search["epochs"] = int(smoke.get("epochs", 1))
    search["fraction"] = 1.0  # the subset is already tiny
    search["startup_trials"] = 1
    search.pop("timeout_hours", None)
    out["final"] = {
        **out["final"],
        "epochs": int(smoke.get("final_epochs", 1)),
        "patience": 1,
        "head_epochs": 0,
    }
    out["final"].pop("max_hours", None)
    fixed = smoke.get("fixed") or {}
    out["fixed"] = {**(out.get("fixed") or {}), **fixed}
    for name in fixed:  # parameters fixed by the smoke test are no longer searched
        search["space"].pop(name, None)
    out["smoke"] = {**smoke, "enabled": True}
    validate_config(out)
    return out


def sampler(config: dict[str, Any]) -> optuna.samplers.BaseSampler:
    seed = int(config["seed"])
    search = config["search"]
    if search.get("sampler", "tpe") == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    return optuna.samplers.TPESampler(
        seed=seed, n_startup_trials=int(search.get("startup_trials", 10))
    )


def create_study(
    config: dict[str, Any], storage: str | None = None, name: str | None = None
) -> optuna.Study:
    """Seeded study, maximising the search metric; resumed when `storage` already has it."""
    return optuna.create_study(
        study_name=name or f"classifier-{config['arch']}",
        storage=storage,
        sampler=sampler(config),
        direction="maximize",
        load_if_exists=storage is not None,
    )


def trial_hyperparameters(config: dict[str, Any], suggested: dict[str, Any]) -> dict[str, Any]:
    """Everything a trial's training receives: fixed values, then the trial's suggestion."""
    return {**(config.get("fixed") or {}), **suggested}


def final_hyperparameters(config: dict[str, Any], best: dict[str, Any]) -> dict[str, Any]:
    """The retraining: the best trial's values, with the `final` overrides (head_epochs)."""
    final = config["final"]
    out = dict(best)
    if "head_epochs" in final:
        out["head_epochs"] = int(final["head_epochs"])
    return out
