"""SparkSession, paths and parameters shared by the jobs of src/data/spark."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[3]  # repository root (/opt/tujia in the containers)
PARAMS = ROOT / "params.yaml"
C3PI_RAW = ROOT / "data" / "raw" / "c3pi"
BIGDATA = ROOT / "data" / "processed" / "c3pi" / "bigdata"
SPLITS_V2 = ROOT / "config" / "splits" / "c3pi_v2"


def spark_session(app: str, master: str | None = None) -> SparkSession:
    """A SparkSession named tujya-<app>.

    The master comes from the argument, then TUJYA_SPARK_MASTER, then local[*]. The
    executors get the repository root on their PYTHONPATH, so the functions of this package
    that run inside UDFs import there too.
    """
    master = master or os.environ.get("TUJYA_SPARK_MASTER", "local[*]")
    builder = (
        SparkSession.builder.appName(f"tujya-{app}")
        .master(master)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", os.environ.get("TUJYA_SPARK_PARTITIONS", "16"))
        .config("spark.executorEnv.PYTHONPATH", os.environ.get("PYTHONPATH", str(ROOT)))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
    )
    return builder.getOrCreate()


def load_params(path: Path = PARAMS) -> dict[str, Any]:
    """The `bigdata` section of params.yaml."""
    with path.open(encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle)["bigdata"])


def write_rows(rows: Iterable[Mapping[str, Any]], path: Path, columns: list[str]) -> int:
    """Write a small table as one CSV file (LF line ends, UTF-8); returns the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
            n += 1
    return n


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
