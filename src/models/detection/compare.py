"""Comparison table and figure of the detectors of step 2.2, read from MLflow.

    python -m src.models.detection.compare        # training venv, MLflow server running

Reads the experiment `tujya-vision`: the parent runs of the candidates (tag `stage=compare`)
and the re-evaluation of the baseline (tag `stage=compare-eval`), smoke runs excluded. Writes
- docs/module1/assets/detecteurs_comparaison.png: mAP@[.5:.95] on the comparison split
  against the ONNX CPU latency (one point per model, the champion highlighted);
- the Markdown table between the markers of docs/module1/kaggle_detecteurs.md.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils import mlflow_utils as mu
from src.utils.console import console_utf8

ROOT = mu.ROOT
DOC = ROOT / "docs" / "module1" / "kaggle_detecteurs.md"
FIGURE = ROOT / "docs" / "module1" / "assets" / "detecteurs_comparaison.png"
BEGIN, END = "<!-- comparison:begin -->", "<!-- comparison:end -->"
# Colours of the reference palette (validated pair, direct labels on every point).
CANDIDATE, CHAMPION = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


@dataclass(frozen=True)
class Row:
    arch: str
    run_id: str
    val_map: float | None
    test_map: float | None
    test_map50: float | None
    precision: float | None
    recall: float | None
    latency_ms: float | None
    parameters: float | None
    status: str  # champion / challenger / archived / baseline / ""


def _get(metrics: dict[str, float], key: str) -> float | None:
    value = metrics.get(key)
    return None if value is None else float(value)


def row_from_run(run: Any, status: str = "") -> Row:
    metrics, tags = run.data.metrics, run.data.tags
    return Row(
        arch=tags.get("arch", "?"),
        run_id=run.info.run_id,
        val_map=_get(metrics, "val_mAP50_95"),
        test_map=_get(metrics, "test_v2_mAP50_95"),
        test_map50=_get(metrics, "test_v2_mAP50"),
        precision=_get(metrics, "test_v2_precision"),
        recall=_get(metrics, "test_v2_recall"),
        latency_ms=_get(metrics, "onnx_cpu_latency_ms"),
        parameters=_get(metrics, "parameters"),
        status=status,
    )


def collect(experiment: str = "tujya-vision") -> list[Row]:
    from mlflow import MlflowClient

    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return []
    status_by_run: dict[str, str] = {}
    for mv in client.search_model_versions("name='tujya-pill-detector'"):
        status_by_run[mv.run_id] = mv.tags.get("status", "")
    runs = client.search_runs(
        [exp.experiment_id],
        "tags.stage IN ('compare', 'compare-eval') and tags.smoke = 'false'",
        order_by=["attributes.start_time DESC"],
    )
    rows: dict[str, Row] = {}
    for run in runs:
        if run.data.tags.get("stage") == "compare-eval":
            evaluated = run.data.tags.get("evaluated_run_id", "")
            status = status_by_run.get(evaluated, "") or "baseline"
            row = row_from_run(run, status)
            row = Row(**{**row.__dict__, "arch": f"{row.arch} (baseline v1)"})
        else:
            row = row_from_run(run, status_by_run.get(run.info.run_id, ""))
        rows.setdefault(row.arch, row)  # newest run of each model
    return sorted(rows.values(), key=lambda r: -(r.test_map or 0))


def _fmt(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def markdown_table(rows: list[Row]) -> str:
    lines = [
        "| Modèle | mAP@[.5:.95] val | mAP@[.5:.95] test | mAP50 test | Précision | Rappel "
        "| Latence ONNX CPU (ms) | Paramètres (M) | Statut | Run MLflow |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        params = "" if r.parameters is None else f"{r.parameters / 1e6:.1f}"
        lines.append(
            f"| {r.arch} | {_fmt(r.val_map)} | {_fmt(r.test_map)} | {_fmt(r.test_map50)} "
            f"| {_fmt(r.precision, 3)} | {_fmt(r.recall, 3)} | {_fmt(r.latency_ms, 1)} "
            f"| {params} | {r.status} | `{r.run_id[:8]}` |"
        )
    return "\n".join(lines) + "\n"


def replace_between_markers(text: str, table: str) -> str:
    pattern = re.compile(re.escape(BEGIN) + ".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"markers {BEGIN} / {END} not found")
    return pattern.sub(lambda _: f"{BEGIN}\n{table}{END}", text)


def plot(rows: list[Row], path: Path) -> Path:
    """mAP@[.5:.95] (comparison split) against ONNX CPU latency, champion in orange."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = [r for r in rows if r.test_map is not None and r.latency_ms is not None]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    for r in points:
        champion = r.status == "champion"
        ax.scatter(
            r.latency_ms,
            r.test_map,
            s=70 if champion else 50,
            marker="D" if champion else "o",
            color=CHAMPION if champion else CANDIDATE,
            edgecolors="#fcfcfb",
            linewidths=2,
            zorder=3,
        )
        label = f"{r.arch} ({r.status})" if r.status else r.arch
        ax.annotate(
            label,
            (r.latency_ms, r.test_map),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=8,
            color=INK,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Latence ONNX Runtime CPU par image (ms, échelle log)", color=MUTED)
    ax.set_ylabel("mAP@[.5:.95], split de comparaison", color=MUTED)
    ax.set_title("Détecteurs de comprimés, C3PI v2 (étape 2.2)", color=INK, loc="left")
    ax.grid(True, color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="#fcfcfb")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.detection.compare")
    parser.add_argument("--doc", type=Path, default=DOC)
    parser.add_argument("--figure", type=Path, default=FIGURE)
    args = parser.parse_args(argv)
    console_utf8()
    mu.configure()
    rows = collect()
    table = markdown_table(rows)
    print(table)
    plot(rows, args.figure)
    text = args.doc.read_text(encoding="utf-8")
    args.doc.write_bytes(replace_between_markers(text, table).encode("utf-8"))


if __name__ == "__main__":
    main()
