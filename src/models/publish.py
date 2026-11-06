"""Copy the deployment models into the directory the API loads them from.

    python -m src.models.publish                          # served variant of each champion
    python -m src.models.publish --exact                  # the champions themselves (FP32)
    python -m src.models.publish --alias challenger       # another alias (MLflow Registry)
    python -m src.models.publish --classifier-version 2 --detector-version 1
    python -m src.models.publish --from-runs              # newest runs/*/weights/best.onnx

Writes `detector.onnx`, `classifier.onnx` and `models.json` (origin and SHA-256 of each file)
into `--out` (default `models/` at the repository root, git-ignored; the API reads it from
TUJYA_MODELS_DIR, same default). The files are the ONNX exports registered by
`src.models.export_onnx`, so the API serves exactly the versions shown in the Model Registry:
by default the champions chosen by promote_if_better (src/utils/mlflow_utils.py), in the
precision chosen by `python -m src.models.optimize` (alias `served`: FP32, FP16 or INT8 variant
of the champion, src/models/serving.py). `--exact` publishes the alias as it is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import onnx

from src.models.serving import CHAMPION, PRECISION_TAG, REGISTERED, served_version

ROOT = Path(__file__).resolve().parents[2]  # repository root
RUNS = ROOT / "runs"  # Ultralytics training runs (src/models/train.py)

FILES = {"detect": "detector.onnx", "classify": "classifier.onnx"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if (details.tags or {}).get("variant_of"):
        info["variant_of"] = int(details.tags["variant_of"])
    if note:
        info["resolution"] = note
    return onnx_files[0], info


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
            # MLflow stores the weights as external data next to model.onnx; loading and
            # saving again gives one self-contained file (these models are far below 2 GB).
            onnx.save_model(onnx.load(str(path)), str(out / filename))
            manifest[task] = {"file": filename, "sha256": sha256(out / filename), **info}
    (out / "models.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.publish")
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


if __name__ == "__main__":
    main()
