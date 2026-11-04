"""Build the Kaggle notebooks of step 2.2 (notebooks/kaggle/detector_<arch>.ipynb).

    python -m src.models.detection.notebook yolov8n        # writes notebooks/kaggle/...
    python -m src.models.detection.notebook --all

Every notebook is the same short driver: clone the experiment branch, install the pinned
packages (notebooks/kaggle/requirements.txt), find the private dataset, run
src/models/detection/runner.py with config/hpo/detector_<arch>.yaml, zip the export folder.
Only the architecture changes, so they are generated from one template (a test checks that
the committed notebooks are exactly the generated ones). Notebooks are committed without
outputs (nbstripout); the results go to MLflow through import_kaggle.

`TUJYA_SMOKE=1` in the environment runs the notebook in smoke mode (tiny subset, 2 trials,
1 epoch), which is how it is checked locally before going to Kaggle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[3]
FOLDER = ROOT / "notebooks" / "kaggle"
REPO = "https://github.com/melekbouabid/syRX.git"
DATASET = "tujya-c3pi-detect-v2"
# Architecture -> (description, estimated duration on a Kaggle T4, recommended accelerator).
ARCHS = {
    "yolov8n": ("YOLOv8n (Ultralytics, one-stage, 3.2 M parameters)", "about 3 h", "GPU T4 x1"),
    "yolov8s": ("YOLOv8s (Ultralytics, one-stage, 11 M parameters)", "about 5 h", "GPU T4 x1"),
    "yolo11s": ("YOLO11s (Ultralytics, one-stage, 9.4 M parameters)", "about 5 h", "GPU T4 x1"),
    "yolov10n": (
        "YOLOv10n (Ultralytics, one-stage, NMS-free, 2.3 M parameters)",
        "about 3 h",
        "GPU T4 x1",
    ),
    "fasterrcnn": (
        "Faster R-CNN ResNet-50 FPN (torchvision, two-stage, 41 M parameters)",
        "about 7 h",
        "GPU P100 (or T4 x1)",
    ),
}


def branch_of(arch: str) -> str:
    return f"experiment/detector-{arch}"


def path_of(arch: str) -> Path:
    return FOLDER / f"detector_{arch}.ipynb"


def build(arch: str) -> nbformat.NotebookNode:
    description, duration, accelerator = ARCHS[arch]
    branch = branch_of(arch)
    cells = [
        new_markdown_cell(f"""# Tujya, step 2.2: detector `{arch}` on C3PI v2

{description}. Branch `{branch}` of {REPO.removesuffix(".git")}.

What this notebook does (code: `src/models/detection/runner.py`, search space:
`config/hpo/detector_{arch}.yaml`):
1. Optuna search (seeded TPE): each trial trains the pretrained model briefly on part of the
   train split and is scored on the validation split (`val_mAP50_95`);
2. the best trial is retrained on the whole train split;
3. test scores on the comparison split (`test_v2_*`) and the whole v2 test (`test_v2_all_*`),
   ONNX export, CPU latency; everything is written to an export folder, zipped at the end.

Settings before running: **Accelerator** {accelerator}; **Internet** on (clone and `pip`);
**Add data**: your private dataset `{DATASET}` (`python -m src.models.detection.package_kaggle`).
Then **Save Version > Save & Run All (Commit)**: the run continues in the background
({duration}). Download `detector_{arch}_export.zip` from the Output tab and import it:
`python -m src.models.detection.import_kaggle detector_{arch}_export.zip`
(docs/module1/kaggle_detecteurs.md). No secret is needed: never put a token in a notebook."""),
        new_code_cell(f"""import os

ARCH = "{arch}"
BRANCH = "{branch}"
REPO = "{REPO}"
DATASET = "{DATASET}"  # slug of the private Kaggle dataset
# Smoke mode: tiny subset, 2 trials, 1 epoch (a check of the notebook, not an experiment).
SMOKE = os.environ.get("TUJYA_SMOKE", "0").lower() in ("1", "true", "yes")"""),
        new_code_cell("""import subprocess
import sys
from pathlib import Path

ON_KAGGLE = Path("/kaggle/working").is_dir()
if ON_KAGGLE:
    WORK = Path("/kaggle/working")
    REPO_DIR = WORK / "syRX"
    if not REPO_DIR.exists():
        clone = ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, str(REPO_DIR)]
        subprocess.run(clone, check=True)
    requirements = REPO_DIR / "notebooks" / "kaggle" / "requirements.txt"
    pip = [sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)]
    subprocess.run(pip, check=True)
else:  # local smoke run, from the repository
    here = [Path.cwd(), *Path.cwd().parents]
    REPO_DIR = next(p for p in here if (p / "pyproject.toml").is_file())
    WORK = REPO_DIR / "runs" / "notebook"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
commit = subprocess.run(
    ["git", "-C", str(REPO_DIR), "log", "-1", "--format=%H %s"], capture_output=True, text=True
)
print("code:", commit.stdout.strip())"""),
        new_code_cell("""import torch

if ON_KAGGLE:
    found = sorted(Path("/kaggle/input").rglob("test_holdout.txt"))
    assert found, f"Add the private dataset {DATASET} to this notebook (Add data)."
    DATA = found[0].parent
else:
    DATA = REPO_DIR / "data" / "processed" / "c3pi" / "datasets_v2" / "detect"
counts = {s: len(list((DATA / "images" / s).glob("*.jpg"))) for s in ("train", "val", "test")}
print(DATA, counts)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))"""),
        new_code_cell("""from src.models.detection.runner import run_experiment

OUT = WORK / (f"detector-{ARCH}-smoke" if SMOKE else f"detector-{ARCH}")
config = REPO_DIR / "config" / "hpo" / f"detector_{ARCH}.yaml"
summary = run_experiment(config, DATA, OUT, smoke=SMOKE)"""),
        new_code_cell("""import pandas as pd

print(pd.read_csv(OUT / "trials.csv").to_string(index=False))
final = summary["final"]
print("best hyperparameters:", final["params"])
wanted = ("val_mAP", "test_v2_mAP", "test_v2_precision", "test_v2_recall", "test_v2_all_mAP")
print({k: round(v, 4) for k, v in final["metrics"].items() if k.startswith(wanted)})
print(final["latency"])"""),
        new_code_cell("""import shutil

base = str(WORK / f"detector_{ARCH}_export")
archive = shutil.make_archive(base, "zip", OUT.parent, OUT.name)
print(archive, round(Path(archive).stat().st_size / 1e6, 1), "MB")"""),
        new_markdown_cell(f"""## Next steps (on the PC)

1. Download `detector_{arch}_export.zip` (Output tab of the saved version).
2. `docker compose up -d mlflow`, then, in the training venv:
   `python -m src.models.detection.import_kaggle detector_{arch}_export.zip`
   (parent run + one child run per trial in `tujya-vision`, Registry version,
   `promote_if_better` against the champion re-evaluated on the same split).
3. Weights: `dvc add` of the imported export folder (never git), see
   docs/module1/kaggle_detecteurs.md."""),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"{arch}-{index:02d}"
    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    )
    nbformat.validate(notebook)
    return notebook


def write(arch: str) -> Path:
    path = path_of(arch)
    path.parent.mkdir(parents=True, exist_ok=True)
    # nbformat.writes + LF: the same bytes on Windows and Linux (nbstripout's layout).
    path.write_bytes((nbformat.writes(build(arch)) + "\n").encode("utf-8"))
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.detection.notebook")
    parser.add_argument("archs", nargs="*", choices=sorted(ARCHS))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    for arch in sorted(ARCHS) if args.all else args.archs:
        print(write(arch))


if __name__ == "__main__":
    main()
