"""Copy the deployment models into the directory the API loads them from.

    python -m tujya_vision.publish                          # latest registered versions (MLflow)
    python -m tujya_vision.publish --classifier-version 2 --detector-version 1
    python -m tujya_vision.publish --from-runs              # newest vision/runs/*/best.onnx

Writes `detector.onnx`, `classifier.onnx` and `models.json` (origin and SHA-256 of each file)
into `--out` (default `models/` at the repository root, git-ignored; the API reads it from
SYNERGYRX_MODELS_DIR, same default). The files are the ONNX exports registered by
`tujya_vision.train`, so the API serves exactly the versions shown in the Model Registry.
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

from tujya_vision.train import REGISTERED, ROOT, RUNS

FILES = {"detect": "detector.onnx", "classify": "classifier.onnx"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def from_registry(task: str, version: int | None, work: Path) -> tuple[Path, dict[str, Any]]:
    import mlflow
    from mlflow import MlflowClient

    name = REGISTERED[task]
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise SystemExit(f"no registered versions of {name} on {mlflow.get_tracking_uri()}")
    chosen = (
        max(versions, key=lambda v: int(v.version))
        if version is None
        else next((v for v in versions if int(v.version) == version), None)
    )
    if chosen is None:
        raise SystemExit(f"{name} has no version {version}")
    target = work / task
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{name}/{chosen.version}", dst_path=str(target)
    )
    onnx_files = sorted(target.rglob("*.onnx"))
    if not onnx_files:
        raise SystemExit(f"{name} version {chosen.version} has no .onnx file")
    return onnx_files[0], {
        "source": "mlflow",
        "registered_model": name,
        "version": int(chosen.version),
        "run_id": chosen.run_id,
        "tracking_uri": mlflow.get_tracking_uri(),
    }


def from_runs(task: str) -> tuple[Path, dict[str, Any]]:
    runs = sorted(
        p
        for p in RUNS.glob(f"{task}-*/weights/best.onnx")
        if not p.parents[1].name.endswith("-test")
    )
    if not runs:
        raise SystemExit(f"no {task} run with weights/best.onnx under {RUNS}")
    path = runs[-1]
    return path, {"source": "vision/runs", "run": path.parents[1].name}


def publish(out: Path, *, use_runs: bool, versions: dict[str, int | None]) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"published_at": datetime.now(UTC).isoformat(timespec="seconds")}
    with tempfile.TemporaryDirectory() as tmp:
        for task, filename in FILES.items():
            if use_runs:
                path, info = from_runs(task)
            else:
                path, info = from_registry(task, versions[task], Path(tmp))
            # MLflow stores the weights as external data next to model.onnx; loading and
            # saving again gives one self-contained file (these models are far below 2 GB).
            onnx.save_model(onnx.load(str(path)), str(out / filename))
            manifest[task] = {"file": filename, "sha256": sha256(out / filename), **info}
    (out / "models.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tujya_vision.publish")
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    parser.add_argument("--from-runs", action="store_true", help="skip MLflow, use vision/runs")
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
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
