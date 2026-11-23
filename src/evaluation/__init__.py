"""Retrieval and generation benchmark: question set, scripts and results
(docs/module2/results.md)."""

import os

# No usage data leaves the PC from the benchmark scripts (MLflow telemetry, RAGAS analytics);
# their background threads also deadlocked on Python's import lock with the MLflow judges.
for _name in ("MLFLOW_DISABLE_TELEMETRY", "DO_NOT_TRACK", "RAGAS_DO_NOT_TRACK"):
    os.environ.setdefault(_name, "true")
