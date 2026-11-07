"""Comparison table and figure of the classifiers of step 2.3, read from MLflow.

    python -m src.models.classification.compare     # training venv, MLflow server running

Reads the experiment `tujya-vision`: the parent runs of the candidates (tags `stage=compare`,
`task=classify`) and the re-evaluation of the v1 champion (`stage=compare-eval`), smoke runs
excluded. Latency, size and parameters are those logged by
`python -m src.models.detection.compare_latency --task classify` (every model timed together,
D96 / D98). Writes
- docs/module1/assets/classifieurs_comparaison.png: top-1 on the comparison split against
  the ONNX CPU latency and against the size of the ONNX file (champion highlighted);
- the Markdown table between the markers of docs/module1/kaggle_classifieurs.md.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.models.detection.compare import BEGIN, END, replace_between_markers
from src.utils import mlflow_utils as mu
from src.utils.console import console_utf8

__all__ = ["BEGIN", "END", "Row", "collect", "markdown_table", "plot", "replace_between_markers"]

ROOT = mu.ROOT
DOC = ROOT / "docs" / "module1" / "kaggle_classifieurs.md"
FIGURE = ROOT / "docs" / "module1" / "assets" / "classifieurs_comparaison.png"
COMPARED = "tags.stage LIKE 'compare%' and tags.smoke = 'false' and tags.task = 'classify'"
CANDIDATE, CHAMPION = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


@dataclass(frozen=True)
class Row:
    arch: str
    run_id: str
    val_top1: float | None
    top1: float | None  # comparison split (promotion metric)
    top5: float | None
    f1_macro: float | None
    v1c_top1: float | None  # like-for-like with the v1 champion
    all_top1: float | None  # whole v2 test
    latency_ms: float | None
    size_mb: float | None
    parameters: float | None
    status: str
    version: str = ""


def _get(metrics: dict[str, float], key: str) -> float | None:
    value = metrics.get(key)
    return None if value is None else float(value)


def row_from_run(run: Any, status: str = "", version: str = "") -> Row:
    m, tags = run.data.metrics, run.data.tags
    parameters = _get(m, "onnx_parameters")
    return Row(
        arch=tags.get("arch", "?"),
        run_id=run.info.run_id,
        val_top1=_get(m, "val_top1"),
        top1=_get(m, "test_v2_top1"),
        top5=_get(m, "test_v2_top5"),
        f1_macro=_get(m, "test_v2_f1_macro"),
        v1c_top1=_get(m, "test_v2_v1c_top1"),
        all_top1=_get(m, "test_v2_all_top1"),
        latency_ms=_get(m, "onnx_cpu_latency_ms"),
        size_mb=_get(m, "model_size_mb"),
        parameters=parameters if parameters is not None else _get(m, "parameters"),
        status=status,
        version=version,
    )


def collect(
    experiment: str = "tujya-vision", model: str = "tujya-pill-classifier", client: Any = None
) -> list[Row]:
    from mlflow import MlflowClient

    from src.models.detection.compare import final_value

    client = client or MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return []
    status_by_run: dict[str, str] = {}
    version_by_run: dict[str, str] = {}
    for mv in client.search_model_versions(f"name='{model}'"):
        if "precision" in (mv.tags or {}):
            continue  # FP16 / INT8 variants (step 2.4)
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
            row = row_from_run(
                run,
                status_by_run.get(evaluated, "") or "baseline",
                version_by_run.get(evaluated, ""),
            )
            row = replace(row, arch=f"{row.arch} (champion v1, 150 classes)")
        elif stage == "compare":
            run_id = run.info.run_id
            row = row_from_run(run, status_by_run.get(run_id, ""), version_by_run.get(run_id, ""))
            if row.val_top1 is not None:
                row = replace(row, val_top1=final_value(client, run_id, "val_top1", row.val_top1))
        else:
            continue
        rows.setdefault(row.arch, row)  # newest run of each model
    return sorted(rows.values(), key=lambda r: -(r.top1 or 0))


def _fmt(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def markdown_table(rows: list[Row]) -> str:
    lines = [
        "| Modèle | top-1 val | top-1 test | top-5 test | F1 macro test | top-1 classes v1 "
        "| top-1 test v2 complet | Latence ONNX CPU (ms) | Taille ONNX (Mo) | Paramètres (M) "
        "| Version | Statut | Run MLflow |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for r in rows:
        params = "" if r.parameters is None else f"{r.parameters / 1e6:.1f}"
        version = f"v{r.version}" if r.version else ""
        lines.append(
            f"| {r.arch} | {_fmt(r.val_top1)} | {_fmt(r.top1)} | {_fmt(r.top5)} "
            f"| {_fmt(r.f1_macro)} | {_fmt(r.v1c_top1)} | {_fmt(r.all_top1)} "
            f"| {_fmt(r.latency_ms, 1)} | {_fmt(r.size_mb, 1)} | {params} | {version} "
            f"| {r.status} | `{r.run_id[:8]}` |"
        )
    return "\n".join(lines) + "\n"


PANELS = (
    ("latency_ms", "Latence ONNX Runtime CPU par crop (ms, échelle log)"),
    ("size_mb", "Taille du fichier ONNX FP32 (Mo, échelle log)"),
)


def plot(rows: list[Row], path: Path) -> Path:
    """Top-1 (comparison split) against CPU latency and ONNX size, champion in orange."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150, sharey=True)
    for ax, (field, xlabel) in zip(axes, PANELS, strict=True):
        points = [r for r in rows if r.top1 is not None and getattr(r, field) is not None]
        for r in points:
            champion = r.status == "champion"
            x = getattr(r, field)
            ax.scatter(
                x,
                r.top1,
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
                (x, r.top1),
                xytext=(7, 4),
                textcoords="offset points",
                fontsize=7.5,
                color=INK,
            )
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, color=MUTED)
        ax.grid(True, color=GRID, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED)
        ax.margins(x=0.25)
    axes[0].set_ylabel("top-1, split de comparaison", color=MUTED)
    fig.suptitle(
        "Classifieurs de comprimés, C3PI v2 (étape 2.3) : précision, latence et taille",
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
    parser = argparse.ArgumentParser(prog="python -m src.models.classification.compare")
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
