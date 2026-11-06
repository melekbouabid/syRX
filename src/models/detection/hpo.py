"""Hyperparameter search of a detector with Optuna (config/hpo/detector_<arch>.yaml).

One YAML file per experiment branch describes the model and its search:

    arch: yolov8n                  # name used in runs, tags and the Registry description
    family: yolo                   # yolo (Ultralytics) | fasterrcnn (torchvision)
    weights: yolov8n.pt            # pretrained weights (transfer learning)
    branch: experiment/detector-yolov8n
    seed: 0
    search:                        # Optuna: each trial is a short training on a share of train
      n_trials: 12
      epochs: 10
      fraction: 0.4                # share of the train images used by a trial
      metric: val_mAP50_95         # maximised (validation split, never test)
      sampler: tpe                 # tpe | random, seeded with `seed`
      space:
        lr0: {type: float, low: 1.0e-4, high: 1.0e-2, log: true}
        optimizer: {type: categorical, choices: [SGD, AdamW]}
        ...
    fixed: {workers: 4}            # passed to every training as is
    final: {epochs: 60, patience: 15}   # retraining of the best trial on the whole train set
    smoke:                         # local check: tiny subset, few epochs, 2 trials
      n_trials: 2
      ...

A float can depend on a categorical parameter: `{type: float, log: true, depends_on:
optimizer, ranges: {SGD: [1.0e-3, 2.0e-2], AdamW: [1.0e-5, 1.0e-3]}}`. Optuna then sees one
parameter per branch (`lr0_sgd`, `lr0_adamw`), and the training receives `lr0`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import optuna
import yaml

FAMILIES = ("yolo", "fasterrcnn")
TYPES = ("float", "int", "categorical")
REQUIRED = ("arch", "family", "weights", "seed", "search", "final")
SEARCH_REQUIRED = ("n_trials", "epochs", "metric", "space")


class ConfigError(ValueError):
    """The YAML file does not describe a valid search."""


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    validate_config(config)
    return config


def _check_param(name: str, spec: dict[str, Any], space: dict[str, Any]) -> None:
    kind = spec.get("type")
    if kind not in TYPES:
        raise ConfigError(f"{name}: type must be one of {TYPES}, got {kind!r}")
    if kind == "categorical":
        if not spec.get("choices"):
            raise ConfigError(f"{name}: a categorical parameter needs `choices`")
        return
    if "depends_on" in spec:
        parent = space.get(spec["depends_on"])
        if not parent or parent.get("type") != "categorical":
            raise ConfigError(f"{name}: depends_on must name a categorical parameter")
        ranges = spec.get("ranges") or {}
        missing = [c for c in parent["choices"] if c not in ranges]
        if missing:
            raise ConfigError(f"{name}: no range for {missing}")
        for choice, bounds in ranges.items():
            if len(bounds) != 2 or not bounds[0] < bounds[1]:
                raise ConfigError(f"{name}[{choice}]: a range is [low, high] with low < high")
        return
    low, high = spec.get("low"), spec.get("high")
    if low is None or high is None or not low < high:
        raise ConfigError(f"{name}: needs low < high")
    if spec.get("log") and low <= 0:
        raise ConfigError(f"{name}: a log scale needs low > 0")


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
    if int(config["final"].get("epochs", 0)) < 1:
        raise ConfigError("final: epochs must be at least 1")


def apply_smoke(config: dict[str, Any]) -> dict[str, Any]:
    """The configuration of the local smoke test: `smoke` overrides, `smoke=True`."""
    out = copy.deepcopy(config)
    smoke = config.get("smoke") or {}
    search = out["search"]
    search["n_trials"] = int(smoke.get("n_trials", 2))
    search["epochs"] = int(smoke.get("epochs", 1))
    search["fraction"] = 1.0  # the subset is already tiny
    out["final"] = {**out["final"], "epochs": int(smoke.get("final_epochs", 1)), "patience": 1}
    out["fixed"] = {**(out.get("fixed") or {}), **(smoke.get("fixed") or {})}
    # Parameters fixed by the smoke test are no longer searched.
    for name in smoke.get("fixed") or {}:
        search["space"].pop(name, None)
    fixed = smoke.get("fixed") or {}
    for spec in search["space"].values():
        parent = spec.get("depends_on")
        if parent in fixed:
            # The parameter it followed is fixed: keep the range of the fixed value.
            spec.pop("depends_on")
            ranges = spec.pop("ranges")
            low, high = ranges.get(fixed[parent], next(iter(ranges.values())))
            spec.update(low=low, high=high)
    out["smoke"] = {**smoke, "enabled": True}
    validate_config(out)
    return out


def _branch_name(name: str, choice: Any) -> str:
    return f"{name}_{str(choice).lower()}"


def suggest(trial: optuna.trial.BaseTrial, space: dict[str, Any]) -> dict[str, Any]:
    """Hyperparameters of one trial (dependent floats after the parameter they follow)."""
    values: dict[str, Any] = {}
    ordered = sorted(space.items(), key=lambda item: "depends_on" in item[1])
    for name, spec in ordered:
        kind = spec["type"]
        if kind == "categorical":
            values[name] = trial.suggest_categorical(name, list(spec["choices"]))
        elif "depends_on" in spec:
            choice = values[spec["depends_on"]]
            low, high = spec["ranges"][choice]
            values[name] = trial.suggest_float(
                _branch_name(name, choice), float(low), float(high), log=bool(spec.get("log"))
            )
        elif kind == "float":
            values[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log"))
            )
        else:
            values[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log")),
            )
    return {name: values[name] for name in space}  # the order of the YAML file


def sampler(config: dict[str, Any]) -> optuna.samplers.BaseSampler:
    seed = int(config["seed"])
    if config["search"].get("sampler", "tpe") == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed)


def create_study(
    config: dict[str, Any], storage: str | None = None, name: str | None = None
) -> optuna.Study:
    """Seeded study, maximising the search metric; resumed when `storage` already has it."""
    return optuna.create_study(
        study_name=name or f"detector-{config['arch']}",
        storage=storage,
        sampler=sampler(config),
        direction="maximize",
        load_if_exists=storage is not None,
    )


def trial_hyperparameters(config: dict[str, Any], suggested: dict[str, Any]) -> dict[str, Any]:
    """Everything a training receives: fixed values, then the trial's suggestion."""
    return {**(config.get("fixed") or {}), **suggested}
