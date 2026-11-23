"""Which registered version the API serves: the champion, or its optimised variant (step 2.4).

The alias `champion` always marks the best FP32 model: promote_if_better
(src/utils/mlflow_utils.py) compares new candidates with it on the FP32 test metric, so a
quantized copy must never hold it. The optimised variants of a champion (FP16, INT8) are
registered as further versions of the same registered model with the tags `precision` and
`variant_of` (the champion's version number); the one chosen for serving holds the alias
`served` (src/models/optimize.py). When FP32 is chosen, `served` is the champion itself.

`served_version` resolves what to publish: the `served` version when it is the champion or a
variant of the current champion; otherwise (a new champion was promoted since the last
optimisation) the champion itself, until `python -m src.models.optimize` runs again.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient

REGISTERED = {"classify": "tujya-pill-classifier", "detect": "tujya-pill-detector"}
CHAMPION = "champion"
SERVED = "served"
PRECISION_TAG = "precision"
VARIANT_OF_TAG = "variant_of"
SERVED_PRECISION_TAG = "served_precision"  # on the champion version: what is served


@dataclass(frozen=True)
class Resolved:
    version: str
    precision: str
    variant_of: str | None
    note: str


def _alias(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except mlflow.exceptions.MlflowException:
        return None


def served_version(name: str, client: MlflowClient | None = None) -> Resolved:
    """The version to serve for the registered model `name` (see the module docstring)."""
    client = client or MlflowClient()
    champion = _alias(client, name, CHAMPION)
    if champion is None:
        raise LookupError(f"no version of {name} has the alias {CHAMPION!r}")
    served = _alias(client, name, SERVED)
    if served is None:
        return Resolved(champion, "fp32", None, "no 'served' alias: the champion (FP32)")
    if served == champion:
        return Resolved(champion, "fp32", None, "'served' is the champion (FP32)")
    tags = client.get_model_version(name, served).tags or {}
    if tags.get(VARIANT_OF_TAG) == champion:
        precision = tags.get(PRECISION_TAG, "unknown")
        return Resolved(served, precision, champion, f"{precision} variant of the champion")
    return Resolved(
        champion,
        "fp32",
        None,
        f"'served' (v{served}) is a variant of v{tags.get(VARIANT_OF_TAG)}, not of the "
        f"champion v{champion}: the champion (FP32) until src.models.optimize runs again",
    )


def set_served(
    name: str, version: str, champion: str, precision: str, client: MlflowClient | None = None
) -> None:
    """Give `served` to `version` and record the served precision on the champion."""
    client = client or MlflowClient()
    client.set_registered_model_alias(name, SERVED, str(version))
    client.set_model_version_tag(name, str(champion), SERVED_PRECISION_TAG, precision)
