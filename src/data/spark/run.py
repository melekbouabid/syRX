"""Run one Big Data job in this Python process: `python -m src.data.spark.run <job>`.

Needs pyspark and Java 17+ (the Spark image of docker-compose.yml has both; on the Windows
host use `python -m src.data.spark.submit <job>`, which runs this file in the cluster).
The master comes from --master, then TUJYA_SPARK_MASTER, then local[*].
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

# spark-submit runs this file as a script: put the repository root on sys.path first.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.spark.session import load_params, spark_session

JOBS = {
    "index": "src.data.spark.index_job",
    "quality": "src.data.spark.quality_job",
    "split": "src.data.spark.split_job",
    "augment": "src.data.spark.augment_job",
    "epillid": "src.data.spark.epillid_job",
    "rag": "src.data.spark.rag_job",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.data.spark.run")
    parser.add_argument("job", choices=sorted(JOBS))
    parser.add_argument("--master", default=None, help="Spark master URL (default local[*])")
    args = parser.parse_args(argv)
    module = importlib.import_module(JOBS[args.job])
    spark = spark_session(args.job, args.master)
    start = time.monotonic()
    try:
        summary = module.run(spark, load_params())
    finally:
        spark.stop()
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"job {args.job}: {time.monotonic() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
