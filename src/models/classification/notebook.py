"""Build the Kaggle notebooks of step 2.3 (notebooks/kaggle/classifier_<arch>.ipynb).

    python -m src.models.classification.notebook resnet50        # writes notebooks/kaggle/...
    python -m src.models.classification.notebook --all

Same driver as the detectors' notebooks (src/models/detection/notebook.py): clone the
experiment branch, install the pinned packages (notebooks/kaggle/requirements.txt), find the
private crop dataset, run src/models/classification/runner.py with
config/hpo/classifier_<arch>.yaml, zip the export folder. Only the architecture changes, so
they are generated from one template (a test checks that the committed notebooks are exactly
the generated ones). Notebooks are committed without outputs (nbstripout).

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
DATASET = "tujya-c3pi-classify-v2"
DATASET_NAME = "c3pi-v2-classify"  # `name` in the dataset's MANIFEST.json
# Architecture -> (description, estimated duration on a Kaggle T4, recommended accelerator).
# Estimates: docs/module1/kaggle_classifieurs.md (training throughput measured on the PC,
# scaled to a T4 and capped by the data loading of the 4 Kaggle CPU cores, D103).
ARCHS = {
    "baseline-cnn": (
        "Small CNN trained from scratch (4 convolution blocks, 1.3 M parameters), the baseline",
        "about 2.75 h",
        "GPU T4 x1",
    ),
    "resnet50": ("ResNet-50 (torchvision, ImageNet, 24 M parameters)", "about 2.3 h", "GPU T4 x1"),
    "efficientnetv2s": (
        "EfficientNetV2-S (torchvision, ImageNet, 21 M parameters)",
        "about 2.9 h",
        "GPU T4 x1",
    ),
    "convnext-tiny": (
        "ConvNeXt-Tiny (torchvision, ImageNet, 28 M parameters)",
        "about 2.5 h",
        "GPU T4 x1",
    ),
    "vit-b16": (
        "ViT-B/16 (torchvision, ImageNet, 86 M parameters, 224 px)",
        "about 2.8 h",
        "GPU T4 x1",
    ),
    "yolo11s-cls": (
        "YOLO11s-cls (Ultralytics, ImageNet, 5.9 M parameters), the architecture served today",
        "about 2.3 h",
        "GPU T4 x1",
    ),
}


def branch_of(arch: str) -> str:
    return f"experiment/classifier-{arch}"


def path_of(arch: str) -> Path:
    return FOLDER / f"classifier_{arch}.ipynb"


def build(arch: str) -> nbformat.NotebookNode:
    description, duration, accelerator = ARCHS[arch]
    branch = branch_of(arch)
    cells = [
        new_markdown_cell(f"""# Tujya, step 2.3: classifier `{arch}` on C3PI v2

{description}. Branch `{branch}` of {REPO.removesuffix(".git")}.

What this notebook does (code: `src/models/classification/runner.py`, search space:
`config/hpo/classifier_{arch}.yaml`):
1. Optuna search (seeded TPE): each trial trains the model briefly on part of the train
   crops (transfer learning, then selective fine-tuning; class weights or focal loss;
   RandAugment, MixUp and CutMix on the train split only) and is scored on the validation
   split (`val_top1`);
2. the best trial is retrained on the whole train split;
3. test scores on the comparison split (`test_v2_*`: top-1, top-5, macro F1), the whole v2
   test (`test_v2_all_*`) and the 150 classes of v1 (`test_v2_v1c_*`), confusion matrix,
   Grad-CAM pictures, ONNX export (with the class list), CPU latency; everything is written to
   an export folder, zipped at the end.

Settings before running: **Accelerator** {accelerator}; **Internet** on (clone, `pip`,
pretrained weights); **Add data**: your private dataset `{DATASET}`
(`python -m src.models.classification.package_kaggle`). Then **Save Version > Save & Run All
(Commit)**: the run continues in the background ({duration}). Download
`classifier_{arch}_export.zip` from the Output tab and import it:
`python -m src.models.classification.import_kaggle classifier_{arch}_export.zip`
(docs/module1/kaggle_classifieurs.md). No secret is needed: never put a token in a notebook."""),
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
        new_code_cell(f"""import json

import torch

if ON_KAGGLE:
    manifests = sorted(Path("/kaggle/input").rglob("MANIFEST.json"))
    found = [
        p.parent
        for p in manifests
        if json.loads(p.read_text(encoding="utf-8")).get("name") == "{DATASET_NAME}"
    ]
    assert found, f"Add the private dataset {{DATASET}} to this notebook (Add data)."
    DATA = found[0]
else:
    DATA = REPO_DIR / "data" / "processed" / "c3pi" / "datasets_v2" / "classify"
counts = {{s: len(list((DATA / s).glob("*/*.jpg"))) for s in ("train", "val", "test")}}
print(DATA, counts)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))"""),
        new_code_cell("""from src.models.classification.runner import run_experiment

OUT = WORK / (f"classifier-{ARCH}-smoke" if SMOKE else f"classifier-{ARCH}")
config = REPO_DIR / "config" / "hpo" / f"classifier_{ARCH}.yaml"
summary = run_experiment(config, DATA, OUT, smoke=SMOKE)"""),
        new_code_cell("""import pandas as pd

print(pd.read_csv(OUT / "trials.csv").to_string(index=False))
final = summary["final"]
print("best hyperparameters:", final["params"])
wanted = ("val_top", "val_f1", "test_v2_top", "test_v2_f1", "test_v2_all_top", "parity_")
print({k: round(v, 4) for k, v in final["metrics"].items() if k.startswith(wanted)})
print(final["latency"])
print(sorted(p.name for p in (OUT / "gradcam").glob("*.png")))"""),
        new_code_cell("""import shutil

base = str(WORK / f"classifier_{ARCH}_export")
archive = shutil.make_archive(base, "zip", OUT.parent, OUT.name)
print(archive, round(Path(archive).stat().st_size / 1e6, 1), "MB")"""),
        new_markdown_cell(f"""## Next steps (on the PC)

1. Download `classifier_{arch}_export.zip` (Output tab of the saved version).
2. `docker compose up -d mlflow`, then, in the training venv:
   `python -m src.models.classification.import_kaggle classifier_{arch}_export.zip`
   (parent run + one child run per trial in `tujya-vision`, Registry version,
   `promote_if_better` against the champion scored on the same split).
3. Weights: `dvc add` of the imported export folder (never git), see
   docs/module1/kaggle_classifieurs.md."""),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = str(index)  # the ids nbstripout gives (pre-commit hook)
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
    parser = argparse.ArgumentParser(prog="python -m src.models.classification.notebook")
    parser.add_argument("archs", nargs="*", choices=sorted(ARCHS))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    for arch in sorted(ARCHS) if args.all else args.archs:
        print(write(arch))


if __name__ == "__main__":
    main()
