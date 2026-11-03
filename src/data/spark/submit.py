"""Submit a Big Data job to the Spark cluster of docker-compose.yml, from the host.

    python -m src.data.spark.submit <job>          # index, quality, split, augment, epillid, rag

Standard library only, so it runs from any venv (it is the command of the DVC stages).
It starts the cluster if needed (`docker compose --profile spark up -d`), then runs
`spark-submit` in the master container, where the driver lives; the executors run in the
worker container. The repository is mounted on /opt/tujia in both, so the job reads and
writes the same files as the host. While the job runs, its UI is on http://localhost:4040;
the master UI (http://localhost:8090) keeps the list of finished applications.
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
