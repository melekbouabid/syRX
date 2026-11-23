"""MLflow Model Registry of the vision models: registration, aliases and publication.

    python -m src.models.registry publish                     # served variant of each champion
    python -m src.models.registry publish --exact             # the champions themselves (FP32)
    python -m src.models.registry publish --alias challenger  # another alias

- aliases: `champion` (best FP32 model), `challenger` (runner-up), `served` (the champion or
  its optimised variant, src/models/optimize.py);
- `register`, `alias_version`, `version_metric` and the champion rule `promote_if_better`;
- `served_version` / `set_served`: which version the API serves;
- `publish`: copies the served ONNX models into models/, where the API loads them.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import onnx
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

# --- Served version: the champion or its optimised variant ----------------------------------
# Which registered version the API serves: the champion, or its optimised variant (step 2.4).
#
# The alias `champion` always marks the best FP32 model: promote_if_better
# (below) compares new candidates with it on the FP32 test metric, so a
# quantized copy must never hold it. The optimised variants of a champion (FP16, INT8) are
# registered as further versions of the same registered model with the tags `precision` and
# `variant_of` (the champion's version number); the one chosen for serving holds the alias
# `served` (src/models/optimize.py). When FP32 is chosen, `served` is the champion itself.
#
# `served_version` resolves what to publish: the `served` version when it is the champion or a
# variant of the current champion; otherwise (a new champion was promoted since the last
# optimisation) the champion itself, until `python -m src.models.optimize` runs again.
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


# --- Model Registry: registration, aliases and the champion rule -------------------------------
CHALLENGER = "challenger"
# Tag set on registered versions by promote_if_better: champion / challenger / archived.
STATUS_TAG = "status"
ARCHIVED = "archived"


def register(
    model_uri: str,
    name: str,
    *,
    alias: str | None = None,
    tags: Mapping[str, str] | None = None,
    description: str | None = None,
    client: MlflowClient | None = None,
) -> ModelVersion:
    """Register `model_uri` as a new version of `name`, optionally under an alias."""
    client = client or MlflowClient()
    version = mlflow.register_model(model_uri, name, tags=dict(tags) if tags else None)
    if description:
        client.update_model_version(name, version.version, description=description)
    if alias:
        client.set_registered_model_alias(name, alias, version.version)
    return client.get_model_version(name, version.version)


def alias_version(name: str, alias: str, client: MlflowClient | None = None) -> str | None:
    """Version number holding `alias`, or None."""
    client = client or MlflowClient()
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except mlflow.exceptions.MlflowException:
        return None


def version_metric(
    name: str, version: str | int, metric: str, client: MlflowClient | None = None
) -> float | None:
    """Value of `metric` for a registered version: its run's metric, else its LoggedModel's."""
    client = client or MlflowClient()
    mv = client.get_model_version(name, str(version))
    if mv.run_id:
        value = client.get_run(mv.run_id).data.metrics.get(metric)
        if value is not None:
            return float(value)
    model_id = getattr(mv, "model_id", None)
    if model_id:
        for item in client.get_logged_model(model_id).metrics or []:
            if item.key == metric:
                return float(item.value)
    return None


@dataclass(frozen=True)
class Promotion:
    """Outcome of promote_if_better."""

    name: str
    version: str
    metric: str
    value: float
    promoted: bool  # the candidate is now the champion
    status: str  # champion / challenger / archived: alias given to the candidate
    champion_before: str | None = None
    champion_value: float | None = None
    reason: str = ""


def promote_if_better(
    name: str,
    version: str | int,
    metric: str,
    min_gain: float = 0.0,
    *,
    greater_is_better: bool = True,
    eligible: bool = True,
    ineligible_reason: str = "",
    client: MlflowClient | None = None,
) -> Promotion:
    """Give the alias `champion` to `version` only if it beats the champion by `min_gain`.

    - no champion yet: the candidate becomes champion;
    - candidate better by at least `min_gain` (and strictly better): it becomes champion and
      the former champion becomes challenger;
    - otherwise the champion stays; the candidate becomes challenger if there is none or if
      it beats the current challenger, else it is archived.

    The versions involved get the tag `status` (champion / challenger / archived) and the
    candidate a tag `promotion` explaining the decision, so the Registry shows the history.

    `eligible=False` (a guard the caller checked failed, `ineligible_reason` says which) keeps
    the candidate away from both aliases: it is archived whatever its score (D100).
    """
    client = client or MlflowClient()
    version = str(version)
    value = version_metric(name, version, metric, client)
    if value is None:
        raise ValueError(f"{name} version {version} has no metric {metric!r}")
    sign = 1.0 if greater_is_better else -1.0

    def gain_over(other: float) -> float:
        return sign * (value - other)

    champion = alias_version(name, CHAMPION, client)
    if not eligible:
        reason = (
            f"not eligible: {ineligible_reason or 'guard failed'} ({metric}={value:.4f}); archived"
        )
        client.set_model_version_tag(name, version, STATUS_TAG, ARCHIVED)
        client.set_model_version_tag(name, version, "promotion", reason)
        return Promotion(name, version, metric, value, False, ARCHIVED, champion, None, reason)
    if champion == version:
        return Promotion(
            name, version, metric, value, True, CHAMPION, champion, value, "already champion"
        )
    champion_value = (
        version_metric(name, champion, metric, client) if champion is not None else None
    )

    if champion is None or champion_value is None:
        reason = f"first champion ({metric}={value:.4f})"
        _set_status(client, name, version, CHAMPION)
        client.set_model_version_tag(name, version, "promotion", reason)
        return Promotion(name, version, metric, value, True, CHAMPION, champion, None, reason)

    gain = gain_over(champion_value)
    if gain > 0 and gain >= min_gain:
        reason = (
            f"promoted: {metric}={value:.4f} vs {champion_value:.4f} (v{champion}), "
            f"gain {gain:+.4f} >= {min_gain}"
        )
        _set_status(client, name, champion, CHALLENGER)
        _set_status(client, name, version, CHAMPION)
        client.set_model_version_tag(name, version, "promotion", reason)
        client.set_model_version_tag(name, version, "promoted_from", champion)
        return Promotion(
            name, version, metric, value, True, CHAMPION, champion, champion_value, reason
        )

    challenger = alias_version(name, CHALLENGER, client)
    challenger_value = (
        version_metric(name, challenger, metric, client) if challenger is not None else None
    )
    if challenger_value is None or gain_over(challenger_value) > 0:
        status = CHALLENGER
        _set_status(client, name, version, CHALLENGER)
    else:
        status = ARCHIVED
        client.set_model_version_tag(name, version, STATUS_TAG, ARCHIVED)
    reason = (
        f"not promoted: {metric}={value:.4f} vs champion {champion_value:.4f} (v{champion}), "
        f"gain {gain:+.4f} < {min_gain}; {status}"
    )
    client.set_model_version_tag(name, version, "promotion", reason)
    return Promotion(name, version, metric, value, False, status, champion, champion_value, reason)


def _set_status(client: MlflowClient, name: str, version: str, alias: str) -> None:
    """Move `alias` to `version`, archive the version that loses it, tag the new holder."""
    previous = alias_version(name, alias, client)
    client.set_registered_model_alias(name, alias, version)
    client.set_model_version_tag(name, version, STATUS_TAG, alias)
    if previous is not None and previous != version:
        holder = client.get_model_version(name, previous)
        # Only champion / challenger are statuses: `served` (step 2.4) does not keep one.
        if not set(holder.aliases or []) & {CHAMPION, CHALLENGER}:
            client.set_model_version_tag(name, previous, STATUS_TAG, ARCHIVED)


# --- Publication of the served models ----------------------------------------------------------
# Copy the deployment models into the directory the API loads them from.
#
#     python -m src.models.publish                          # served variant of each champion
#     python -m src.models.publish --exact                  # the champions themselves (FP32)
#     python -m src.models.publish --alias challenger       # another alias (MLflow Registry)
#     python -m src.models.publish --classifier-version 2 --detector-version 1
#     python -m src.models.publish --from-runs              # newest runs/*/weights/best.onnx
#
# Writes `detector.onnx`, `classifier.onnx` and `models.json` (origin and SHA-256 of each file)
# into `--out` (default `models/` at the repository root, git-ignored; the API reads it from
# TUJYA_MODELS_DIR, same default). The files are the ONNX exports registered by
# `src.models.export_onnx`, so the API serves exactly the versions shown in the Model Registry:
# by default the champions chosen by promote_if_better (above), in the
# precision chosen by `python -m src.models.optimize` (alias `served`: FP32, FP16 or INT8 variant
# of the champion, src/models/serving.py). `--exact` publishes the alias as it is.
#
# A classifier that carries its class list (ONNX metadata `class_keys`, the C3PI product of
# every output: the classifiers of step 2.3, D101) is published only if its `names` point, for
# every output, to the class_id of the same product in the API's class table
# (config/pill_classes.csv); otherwise the API would show the wrong medicine. The Ultralytics
# export of v1 has no `class_keys` and is published as before.

ROOT = Path(__file__).resolve().parents[2]  # repository root
RUNS = ROOT / "runs"  # Ultralytics training runs (src/models/train.py)
CLASS_TABLE = ROOT / "config" / "pill_classes.csv"  # what the API resolves class ids with

FILES = {"detect": "detector.onnx", "classify": "classifier.onnx"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_class_table(model: Path, table: Path = CLASS_TABLE) -> None:
    """Refuse a classifier whose outputs would name other products in the API's class table.

    `names` (output index -> class folder) is what backend/app/inference.py reads as the
    class_id of `table`; `class_keys` (output index -> C3PI class key, NDC9 + part) says which
    product each output really is. Both must agree for every output (D101).
    """
    props = onnx.load(str(model), load_external_data=False).metadata_props
    meta = {p.key: p.value for p in props}
    if "class_keys" not in meta:
        return  # Ultralytics export (v1): its names are the class ids of the table
    names = ast.literal_eval(meta.get("names", "{}"))
    keys = json.loads(meta["class_keys"])
    with table.open(newline="", encoding="utf-8") as handle:
        known = {int(row["class_id"]): row["class_key"] for row in csv.DictReader(handle)}
    wrong = [i for i in sorted(names) if known.get(int(names[i])) != keys.get(str(i))]
    if wrong:
        raise SystemExit(
            f"the classifier's class list ({meta.get('label_space', '?')}, "
            f"{len(names)} outputs) does not match {table.name}: {len(wrong)} outputs would "
            f"name another product (first: output {wrong[0]}, {keys.get(str(wrong[0]))}). "
            "Move the API's class table to the model's label space first "
            "(docs/module1/kaggle_classifieurs.md, decision D101)."
        )


def from_registry(
    task: str, version: int | None, work: Path, alias: str = CHAMPION, *, exact: bool = True
) -> tuple[Path, dict[str, Any]]:
    """The registered ONNX model of `task`: version `version`, else the one holding `alias`.

    With `exact=False` and the alias `champion`, the served variant of the champion
    (src/models/serving.py) instead of the champion itself.
    """
    import mlflow
    from mlflow import MlflowClient

    name = REGISTERED[task]
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise SystemExit(f"no registered versions of {name} on {mlflow.get_tracking_uri()}")
    note = None
    if version is None and alias == CHAMPION and not exact:
        try:
            resolved = served_version(name, client)
        except LookupError as error:
            raise SystemExit(str(error)) from None
        version, note = int(resolved.version), resolved.note
    if version is None:
        # search_model_versions does not return the aliases: ask for the alias directly.
        try:
            chosen = client.get_model_version_by_alias(name, alias)
        except mlflow.exceptions.MlflowException:
            raise SystemExit(f"no version of {name} has the alias {alias!r}") from None
    else:
        chosen = next((v for v in versions if int(v.version) == version), None)
        if chosen is None:
            raise SystemExit(f"{name} has no version {version}")
    target = work / task
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{name}/{chosen.version}", dst_path=str(target)
    )
    onnx_files = sorted(target.rglob("*.onnx"))
    if not onnx_files:
        raise SystemExit(f"{name} version {chosen.version} has no .onnx file")
    details = client.get_model_version(name, chosen.version)
    info: dict[str, Any] = {
        "source": "mlflow",
        "registered_model": name,
        "version": int(chosen.version),
        "aliases": sorted(details.aliases or []),
        "precision": (details.tags or {}).get(PRECISION_TAG, "fp32"),
        "run_id": chosen.run_id,
        "tracking_uri": mlflow.get_tracking_uri(),
    }
    if (details.tags or {}).get("arch"):
        info["arch"] = details.tags["arch"]
    if (details.tags or {}).get("variant_of"):
        info["variant_of"] = int(details.tags["variant_of"])
    info["metrics"] = served_metrics(client, name, details)
    if note:
        info["resolution"] = note
    return onnx_files[0], info


METRIC_PREFIXES = ("test_", "val_", "metrics/")
METRIC_KEYS = {"onnx_cpu_latency_ms", "onnx_cpu_latency_p95_ms", "model_size_mb", "accuracy_drop"}


def run_metrics(client: Any, run_id: str | None) -> dict[str, float]:
    """Evaluation metrics of a run (validation and test scores, latency, size), no losses."""
    if not run_id:
        return {}
    metrics = client.get_run(run_id).data.metrics
    return {
        key: float(value)
        for key, value in sorted(metrics.items())
        if (key.startswith(METRIC_PREFIXES) or key in METRIC_KEYS) and "loss" not in key
    }


def served_metrics(client: Any, name: str, details: Any) -> dict[str, float]:
    """Metrics of the published version; a variant adds its own to its champion's (val_*...)."""
    metrics: dict[str, float] = {}
    of = (details.tags or {}).get("variant_of")
    if of:
        metrics.update(run_metrics(client, client.get_model_version(name, of).run_id))
    metrics.update(run_metrics(client, details.run_id))
    return metrics


def from_runs(task: str) -> tuple[Path, dict[str, Any]]:
    runs = sorted(
        p
        for p in RUNS.glob(f"{task}-*/weights/best.onnx")
        if not p.parents[1].name.endswith("-test")
    )
    if not runs:
        raise SystemExit(f"no {task} run with weights/best.onnx under {RUNS}")
    path = runs[-1]
    return path, {"source": "runs", "run": path.parents[1].name}


def publish(
    out: Path,
    *,
    use_runs: bool,
    versions: dict[str, int | None],
    alias: str = CHAMPION,
    exact: bool = False,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"published_at": datetime.now(UTC).isoformat(timespec="seconds")}
    with tempfile.TemporaryDirectory() as tmp:
        for task, filename in FILES.items():
            if use_runs:
                path, info = from_runs(task)
            else:
                path, info = from_registry(task, versions[task], Path(tmp), alias, exact=exact)
            if task == "classify":
                check_class_table(path)
            # MLflow stores the weights as external data next to model.onnx; loading and
            # saving again gives one self-contained file (these models are far below 2 GB).
            onnx.save_model(onnx.load(str(path)), str(out / filename))
            manifest[task] = {"file": filename, "sha256": sha256(out / filename), **info}
    (out / "models.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def publish_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.registry publish")
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    parser.add_argument("--from-runs", action="store_true", help="skip MLflow, use runs/")
    parser.add_argument("--alias", default=CHAMPION, help="Model Registry alias to publish")
    parser.add_argument(
        "--exact",
        action="store_true",
        help="publish the alias itself, not the served variant of the champion (alias served)",
    )
    parser.add_argument("--classifier-version", type=int, default=None)
    parser.add_argument("--detector-version", type=int, default=None)
    args = parser.parse_args(argv)
    if not args.from_runs:
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    manifest = publish(
        args.out,
        use_runs=args.from_runs,
        versions={"classify": args.classifier_version, "detect": args.detector_version},
        alias=args.alias,
        exact=args.exact,
    )
    print(json.dumps(manifest, indent=2))


COMMANDS = {"publish": publish_main}


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    if not args or args[0] not in COMMANDS:
        sys.exit(f"usage: python -m src.models.registry {{{'|'.join(COMMANDS)}}} [options]")
    COMMANDS[args[0]](args[1:])


if __name__ == "__main__":
    main()
