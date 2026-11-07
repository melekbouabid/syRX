"""Comparison table and figure of the detectors of step 2.2, read from MLflow.

    python -m src.models.detection.compare        # training venv, MLflow server running

Reads the experiment `tujya-vision`: the parent runs of the candidates (tag `stage=compare`)
and the re-evaluation of the baseline (tag `stage=compare-eval`), smoke runs excluded (the
MLflow search has no `IN` for tags: `tags.stage LIKE 'compare%'`). The latency, size and
parameter figures are those logged by src/models/detection/compare_latency.py (every model
timed together, D96 / D98). Writes
- docs/module1/assets/detecteurs_comparaison.png: mAP@[.5:.95] on the comparison split
  against the ONNX CPU latency and against the size of the ONNX file (one point per model,
  the champion highlighted);
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
    test_all_map: float | None = None  # whole v2 test split
    latency_p95_ms: float | None = None
    size_mb: float | None = None
    version: str = ""  # Registry version


def _get(metrics: dict[str, float], key: str) -> float | None:
    value = metrics.get(key)
    return None if value is None else float(value)


def row_from_run(run: Any, status: str = "", version: str = "") -> Row:
    metrics, tags = run.data.metrics, run.data.tags
    # One counting method for every model: the weights of the ONNX graph, else the training's.
    parameters = _get(metrics, "onnx_parameters")
    if parameters is None:
        parameters = _get(metrics, "parameters")
    return Row(
        arch=tags.get("arch", "?"),
        run_id=run.info.run_id,
        val_map=_get(metrics, "val_mAP50_95"),
        test_map=_get(metrics, "test_v2_mAP50_95"),
        test_map50=_get(metrics, "test_v2_mAP50"),
        precision=_get(metrics, "test_v2_precision"),
        recall=_get(metrics, "test_v2_recall"),
        latency_ms=_get(metrics, "onnx_cpu_latency_ms"),
        parameters=parameters,
        status=status,
        test_all_map=_get(metrics, "test_v2_all_mAP50_95"),
        latency_p95_ms=_get(metrics, "onnx_cpu_latency_p95_ms"),
        size_mb=_get(metrics, "model_size_mb"),
        version=version,
    )


COMPARED = "tags.stage LIKE 'compare%' and tags.smoke = 'false'"


def final_value(client: Any, run_id: str, key: str, latest: float | None) -> float | None:
    """The score of the retained model (step 0), not of the last epoch.

    The import logs the final metrics at step 0 and the per-epoch history at steps 1..n; a
    history key with the same name (Faster R-CNN's `val_mAP50_95`) would otherwise show the
    last epoch, which is not the checkpoint kept.
    """
    at_zero = [m for m in client.get_metric_history(run_id, key) if m.step == 0]
    return float(max(at_zero, key=lambda m: m.timestamp).value) if at_zero else latest


def collect(
    experiment: str = "tujya-vision", model: str = "tujya-pill-detector", client: Any = None
) -> list[Row]:
    from mlflow import MlflowClient

    client = client or MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return []
    status_by_run: dict[str, str] = {}
    version_by_run: dict[str, str] = {}
    for mv in client.search_model_versions(f"name='{model}'"):
        if "precision" in (mv.tags or {}):
            continue  # FP16 / INT8 variants (step 2.4) share the run of no candidate
        status_by_run[mv.run_id] = (mv.tags or {}).get("status", "")
        version_by_run[mv.run_id] = str(mv.version)
    runs = client.search_runs(
        [exp.experiment_id], COMPARED, order_by=["attributes.start_time DESC"]
    )
    rows: dict[str, Row] = {}
    for run in runs:
        stage = run.data.tags.get("stage")
        if stage == "compare-eval":
            evaluated = run.data.tags.get("evaluated_run_id", "")
            status = status_by_run.get(evaluated, "") or "baseline"
            row = row_from_run(run, status, version_by_run.get(evaluated, ""))
            row = Row(**{**row.__dict__, "arch": f"{row.arch} (baseline v1)"})
        elif stage == "compare":
            run_id = run.info.run_id
            row = row_from_run(run, status_by_run.get(run_id, ""), version_by_run.get(run_id, ""))
        else:
            continue  # another stage starting with "compare"
        if row.arch not in rows and row.val_map is not None:
            row = Row(
                **{
                    **row.__dict__,
                    "val_map": final_value(client, row.run_id, "val_mAP50_95", row.val_map),
                }
            )
        rows.setdefault(row.arch, row)  # newest run of each model
    return sorted(rows.values(), key=lambda r: -(r.test_map or 0))


def _fmt(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def markdown_table(rows: list[Row]) -> str:
    lines = [
        "| Modèle | mAP@[.5:.95] val | mAP@[.5:.95] test | mAP50 test | Précision | Rappel "
        "| mAP@[.5:.95] test v2 complet | Latence ONNX CPU moy. (ms) | p95 (ms) | Taille ONNX (Mo) "
        "| Paramètres (M) | Version | Statut | Run MLflow |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for r in rows:
        params = "" if r.parameters is None else f"{r.parameters / 1e6:.1f}"
        version = f"v{r.version}" if r.version else ""
        lines.append(
            f"| {r.arch} | {_fmt(r.val_map)} | {_fmt(r.test_map)} | {_fmt(r.test_map50)} "
            f"| {_fmt(r.precision, 3)} | {_fmt(r.recall, 3)} | {_fmt(r.test_all_map)} "
            f"| {_fmt(r.latency_ms, 1)} | {_fmt(r.latency_p95_ms, 1)} | {_fmt(r.size_mb, 1)} "
            f"| {params} | {version} | {r.status} | `{r.run_id[:8]}` |"
        )
    return "\n".join(lines) + "\n"


def replace_between_markers(text: str, table: str) -> str:
    pattern = re.compile(re.escape(BEGIN) + ".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"markers {BEGIN} / {END} not found")
    return pattern.sub(lambda _: f"{BEGIN}\n{table}{END}", text)


PANELS = (
    ("latency_ms", "Latence ONNX Runtime CPU par image (ms, échelle log)"),
    ("size_mb", "Taille du fichier ONNX FP32 (Mo, échelle log)"),
)


LOG_TICKS = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]


def _crowded(row: Row, points: list[Row], field: str) -> bool:
    """A better point sits just above and close on x: put this label under its point."""
    x = getattr(row, field)
    return any(
        other is not row
        and 0 < (other.test_map or 0) - (row.test_map or 0) < 0.006
        and 0.5 < getattr(other, field) / x < 2.5
        for other in points
    )


def plot(rows: list[Row], path: Path) -> Path:
    """mAP@[.5:.95] (comparison split) against CPU latency and ONNX size, champion in orange.

    Two panels sharing the accuracy axis (never two scales on one axis); every point is
    labelled directly, so the colour only repeats what the label says.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, sharey=True)
    for ax, (field, xlabel) in zip(axes, PANELS, strict=True):
        points = [r for r in rows if r.test_map is not None and getattr(r, field) is not None]
        for r in points:
            champion = r.status == "champion"
            x = getattr(r, field)
            ax.scatter(
                x,
                r.test_map,
                s=70 if champion else 50,
                marker="D" if champion else "o",
                color=CHAMPION if champion else CANDIDATE,
                edgecolors="#fcfcfb",
                linewidths=2,
                zorder=3,
            )
            label = f"{r.arch}\n({r.status})" if r.status else r.arch
            ax.annotate(
                label,
                (x, r.test_map),
                xytext=(7, -22 if _crowded(r, points, field) else 4),
                textcoords="offset points",
                fontsize=7.5,
                color=INK,
            )
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(LOG_TICKS))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(xlabel, color=MUTED)
        ax.grid(True, color=GRID, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED)
        ax.margins(x=0.25)
    axes[0].set_ylabel("mAP@[.5:.95], split de comparaison", color=MUTED)
    fig.suptitle(
        "Détecteurs de comprimés, C3PI v2 (étape 2.2) : précision, latence et taille",
        color=INK,
        x=0.01,
        ha="left",
    )
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
