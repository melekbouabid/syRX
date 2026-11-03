"""Dataset statistics of the C3PI dataset v2, before / after, logged to MLflow.

    python -m src.data.spark.log_stats            # training venv (.venv-train: mlflow, pandas)

Reads the outputs of the Spark jobs (summaries, config/splits/c3pi_v2/classes.csv, the
split table) and the v1 lists (config/pill_classes.csv, config/splits/selected_images.csv),
then:

- writes data/processed/c3pi/bigdata/stats.json, the DVC metrics of the pipeline
  (`dvc metrics show`), with the before / after table (classes, images, GB);
- logs one MLflow run in the experiment `tujya-data` (server MLFLOW_TRACKING_URI, default
  http://127.0.0.1:5000, `docker compose up -d mlflow`): the `bigdata` parameters, the
  counts as metrics, the train / val / test tables as MLflow datasets (`mlflow.data`, with
  their digest and source), the class distribution (CSV and chart), the job summaries,
  params.yaml and dvc.lock, and the tags git_commit and dvc_rev.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
BIGDATA = ROOT / "data" / "processed" / "c3pi" / "bigdata"
SPLITS_V2 = ROOT / "config" / "splits" / "c3pi_v2"
SPLITS_V1 = ROOT / "config" / "splits" / "selected_images.csv"
CLASSES_V1 = ROOT / "config" / "pill_classes.csv"
IMAGES_DVC = ROOT / "data" / "raw" / "c3pi" / "images.dvc"
RAG_V2 = ROOT / "data" / "processed" / "rag" / "corpus_v2" / "summary.json"
STATS = BIGDATA / "stats.json"
EXPERIMENT = "tujya-data"
SUMMARIES = {
    "index": BIGDATA / "index" / "summary.json",
    "quality": BIGDATA / "quality" / "summary.json",
    "split": SPLITS_V2 / "summary.json",
    "augment": BIGDATA / "augmented" / "summary.json",
    "epillid": BIGDATA / "epillid" / "summary.json",
    "rag": RAG_V2,
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _git(*args: str) -> str:
    try:
        out = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607 - git from PATH
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def v1_stats() -> dict[str, Any]:
    """The dataset of steps 1.1-1.3: 150 classes, config/splits/selected_images.csv."""
    with SPLITS_V1.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    frame = pd.DataFrame(rows)
    consumer = frame[frame["image_type"] == "C3PI_Test"]
    per_class = consumer.groupby("class_id").size()
    return {
        "classes": int(frame["class_id"].nunique()),
        "images": len(frame),
        "consumer_photos": len(consumer),
        "bytes": int(frame["file_size"].astype("int64").sum()),
        "by_split": {k: int(v) for k, v in frame["split"].value_counts().sort_index().items()},
        "consumer_per_class_min": int(per_class.min()),
        "consumer_per_class_max": int(per_class.max()),
        "consumer_per_class_mean": round(float(per_class.mean()), 1),
    }


def v2_stats(split: pd.DataFrame, augmented: pd.DataFrame, classes: pd.DataFrame) -> dict:
    consumer = split[split["role"] == "consumer"]
    per_class = consumer.groupby("class_id").size()
    by_split = {k: int(v) for k, v in split["split"].value_counts().sort_index().items()}
    by_split["train_augmented"] = len(augmented)
    return {
        "classes": len(classes),
        "classes_kept_from_v1": int(classes["in_v1"].astype(str).eq("True").sum()),
        "images": len(split),
        "augmented_images": len(augmented),
        "consumer_photos": len(consumer),
        "by_role": {k: int(v) for k, v in split["role"].value_counts().sort_index().items()},
        "bytes": int(split["file_size"].sum()),
        "augmented_bytes": int(
            sum((ROOT / p).stat().st_size for p in augmented["path"] if (ROOT / p).is_file())
        ),
        "by_split": by_split,
        "consumer_per_class_min": int(per_class.min()),
        "consumer_per_class_max": int(per_class.max()),
        "consumer_per_class_mean": round(float(per_class.mean()), 1),
    }


def before_after(v1: dict[str, Any], v2: dict[str, Any]) -> list[dict[str, Any]]:
    gb = 1e9
    return [
        {"measure": "classes", "before": v1["classes"], "after": v2["classes"]},
        {"measure": "images (no augmentation)", "before": v1["images"], "after": v2["images"]},
        {
            "measure": "consumer photos",
            "before": v1["consumer_photos"],
            "after": v2["consumer_photos"],
        },
        {
            "measure": "consumer photos per class (mean)",
            "before": v1["consumer_per_class_mean"],
            "after": v2["consumer_per_class_mean"],
        },
        {"measure": "augmented train images", "before": 0, "after": v2["augmented_images"]},
        {
            "measure": "GB",
            "before": round(v1["bytes"] / gb, 2),
            "after": round((v2["bytes"] + v2["augmented_bytes"]) / gb, 2),
        },
    ]


def build_stats() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    split = pd.read_parquet(BIGDATA / "split" / "images")
    augmented = pd.read_parquet(BIGDATA / "augmented" / "manifest")
    augmented = augmented[augmented["ok"]]
    classes = pd.read_csv(SPLITS_V2 / "classes.csv", dtype={"class_key": str})
    v1, v2 = v1_stats(), v2_stats(split, augmented, classes)
    summaries = {name: _json(path) for name, path in SUMMARIES.items()}
    stats = {
        "before": v1,
        "after": v2,
        "before_after": before_after(v1, v2),
        "rejected": summaries["quality"].get("rejected"),
        "rejected_reasons": summaries["quality"].get("reasons", {}),
        "leakage": summaries["split"].get("leakage", {}),
        "epillid": {
            k: summaries["epillid"].get(k)
            for k in ("pill_types", "pill_types_in_c3pi", "v2_classes_in_epillid")
        },
        "rag": {k: summaries["rag"].get(k) for k in ("chunks", "documents", "ingredients")},
    }
    return stats, split, classes


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    else:
        out[prefix] = value


def log_to_mlflow(stats: dict[str, Any], split: pd.DataFrame, classes: pd.DataFrame) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT)
    params: dict[str, Any] = {}
    bigdata = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))["bigdata"]
    _flatten("bigdata", bigdata, params)
    with mlflow.start_run(run_name="c3pi-v2-big-data") as run:
        mlflow.set_tags(
            {
                "git_commit": _git("rev-parse", "HEAD"),
                "dvc_rev": _git("log", "-1", "--format=%H", "--", "dvc.lock"),
                "dataset": "c3pi-v2",
                "step": "1.4 big data (PySpark)",
            }
        )
        mlflow.log_params({k: str(v) for k, v in params.items()})
        metrics: dict[str, Any] = {}
        for side in ("before", "after"):
            for key, value in stats[side].items():
                if isinstance(value, dict):
                    for sub, number in value.items():
                        metrics[f"{side}_{key}_{sub}"] = number
                else:
                    metrics[f"{side}_{key}"] = value
        metrics["rejected"] = stats["rejected"] or 0
        for key, value in stats["leakage"].items():
            metrics[f"leakage_{key}"] = value
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})

        source = "data/processed/c3pi/bigdata/split/images"
        columns = ["path", "class_id", "class_key", "image_type", "role", "split", "group_id"]
        for name in ("train", "val", "test"):
            part = split.loc[split["split"] == name, columns].reset_index(drop=True)
            dataset = mlflow.data.from_pandas(
                part, source=source, name=f"c3pi-v2-{name}", targets="class_id"
            )
            mlflow.log_input(dataset, context=name)

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
            counts = (
                split[split["role"] == "consumer"]
                .pivot_table(index="class_id", columns="split", values="path", aggfunc="count")
                .fillna(0)
                .astype(int)
            )
            counts.to_csv(folder / "consumer_photos_per_class.csv")
            fig, ax = plt.subplots(figsize=(10, 4))
            totals = counts.sum(axis=1).sort_values(ascending=False).to_numpy()
            ax.bar(range(len(totals)), totals, width=1.0, color="#3b6ea5")
            ax.set_xlabel("class (sorted)")
            ax.set_ylabel("consumer photos")
            ax.set_title(f"C3PI v2: {len(totals)} classes, {int(totals.sum())} consumer photos")
            fig.tight_layout()
            fig.savefig(folder / "class_distribution.png", dpi=120)
            plt.close(fig)
            mlflow.log_artifacts(tmp, artifact_path="stats")
        mlflow.log_artifact(str(SPLITS_V2 / "classes.csv"), artifact_path="splits")
        for name, path in SUMMARIES.items():
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path=f"summaries/{name}")
        for path in (ROOT / "params.yaml", ROOT / "dvc.lock"):
            if path.is_file():
                mlflow.log_artifact(str(path), artifact_path="reproducibility")
        return str(run.info.run_id)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.data.spark.log_stats")
    parser.add_argument("--no-mlflow", action="store_true", help="only write stats.json")
    args = parser.parse_args(argv)
    stats, split, classes = build_stats()
    if not args.no_mlflow:
        stats["mlflow_run_id"] = log_to_mlflow(stats, split, classes)
    STATS.parent.mkdir(parents=True, exist_ok=True)
    STATS.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats["before_after"], indent=2))
    print(f"run: {stats.get('mlflow_run_id', 'not logged')}")


if __name__ == "__main__":
    main()
