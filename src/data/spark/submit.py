"""Submit a Big Data job to the Spark cluster of docker-compose.yml, from the host.

Run: `python -m src.data.spark.submit <job>` (standard library only; the DVC stage command).
Spark UI on http://localhost:4040 while it runs, master UI on http://localhost:8090.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

SERVICES = ["spark-master", "spark-worker"]
MASTER_URL = "spark://spark-master:7077"
CONTAINER_ROOT = "/opt/tujia"


def spark_submit_command(job: str, *, driver_memory: str, executor_memory: str) -> list[str]:
    """The spark-submit command line run inside the master container (client mode)."""
    return [
        "docker",
        "compose",
        "--profile",
        "spark",
        "exec",
        "-T",
        "spark-master",
        "/opt/spark/bin/spark-submit",
        "--master",
        MASTER_URL,
        "--driver-memory",
        driver_memory,
        "--executor-memory",
        executor_memory,
        "--conf",
        "spark.driver.host=spark-master",
        "--conf",
        f"spark.executorEnv.PYTHONPATH={CONTAINER_ROOT}:/opt/spark/python",
        f"{CONTAINER_ROOT}/src/data/spark/run.py",
        job,
        "--master",
        MASTER_URL,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.data.spark.submit")
    parser.add_argument("job")
    parser.add_argument("--driver-memory", default="3g")
    parser.add_argument("--executor-memory", default="6g")
    args = parser.parse_args(argv)
    up = ["docker", "compose", "--profile", "spark", "up", "-d", "--wait", *SERVICES]
    subprocess.run(up, check=True)  # noqa: S603 - fixed command line
    command = spark_submit_command(
        args.job, driver_memory=args.driver_memory, executor_memory=args.executor_memory
    )
    return subprocess.run(command, check=False).returncode  # noqa: S603


if __name__ == "__main__":
    sys.exit(main())
