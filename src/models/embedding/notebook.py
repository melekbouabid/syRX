"""Build the Kaggle notebook of step 3.5 (notebooks/kaggle/finetune_bge-m3.ipynb).

    python -m src.models.embedding.notebook        # writes notebooks/kaggle/finetune_bge-m3.ipynb

Same driver as the notebooks of steps 2.2 and 2.3: clone the experiment branch, install the
pinned packages (notebooks/kaggle/requirements-finetune.txt), find the private pairs dataset,
run src/models/embedding/train.py with config/finetune/bge-m3.yaml, zip the export folder.
A test checks that the committed notebook is exactly the generated one. `TUJYA_SMOKE=1` runs
it in smoke mode (64 pairs, one epoch, the small model of the config's `smoke` block), which is
how it is checked locally before Kaggle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[3]
FOLDER = ROOT / "notebooks" / "kaggle"
REPO = "https://github.com/melekbouabid/syRX.git"
BRANCH = "experiment/fine-tuning-embedding"
DATASET = "tujya-rag-finetune-pairs"
MODEL = "bge-m3"
PATH = FOLDER / f"finetune_{MODEL}.ipynb"
DURATION = "about 1 h (2 h at most)"


def build() -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell(f"""# Tujya, step 3.5: fine-tuning the embedding model `BAAI/bge-m3`

The RAG of the app searches openFDA labels, PubMed abstracts and DDInter pairs with bge-m3
embeddings. This notebook fine-tunes bge-m3 on question / passage pairs built from that corpus
(`src/models/embedding/pairs.py`: no question of the benchmark, no passage a benchmark question
expects). Branch `{BRANCH}` of {REPO.removesuffix(".git")}.

What it does (code: `src/models/embedding/train.py`, settings: `config/finetune/{MODEL}.yaml`):
1. retrieval of the validation questions over the whole corpus with the base model
   (Recall@1/5/10, MRR@10, nDCG@10);
2. training with CachedMultipleNegativesRankingLoss (in-batch and hard negatives, FP16);
   full fine-tuning, or the upper layers only if the T4 runs out of memory (recorded);
3. the same retrieval with the fine-tuned model; the model (FP16 safetensors), the history and
   `run.json` go to an export folder, zipped at the end.

Settings before running: **Accelerator** GPU T4 x1; **Internet** on (clone, `pip`, model
download from Hugging Face, no account); **Add data**: your private dataset `{DATASET}`
(`python -m src.models.embedding.package_kaggle`). Then **Save Version > Save & Run All
(Commit)** ({DURATION}). Download `finetune_{MODEL}_export.zip` from the Output tab and import
it: `python -m src.models.embedding.import_kaggle finetune_{MODEL}_export.zip`
(docs/module2/kaggle_finetuning.md). No secret is needed: never put a token in a notebook."""),
        new_code_cell(f"""import os

MODEL = "{MODEL}"
BRANCH = "{BRANCH}"
REPO = "{REPO}"
DATASET = "{DATASET}"  # slug of the private Kaggle dataset
# Smoke mode: 64 pairs, one epoch, a small model (a check of the notebook, not an experiment).
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
    requirements = REPO_DIR / "notebooks" / "kaggle" / "requirements-finetune.txt"
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
        if json.loads(p.read_text(encoding="utf-8")).get("name") == "{DATASET}"
    ]
    assert found, f"Add the private dataset {{DATASET}} to this notebook (Add data)."
    DATA = found[0]
else:
    DATA = REPO_DIR / "runs" / "kaggle" / "{DATASET}"
print(DATA, json.loads((DATA / "MANIFEST.json").read_text(encoding="utf-8"))["summary"])
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    memory = torch.cuda.get_device_properties(0).total_memory // 2**20
    print(torch.cuda.get_device_name(0), memory, "MiB")"""),
        new_code_cell("""from src.models.embedding.train import run

OUT = WORK / (f"finetune-{MODEL}-smoke" if SMOKE else f"finetune-{MODEL}")
config = REPO_DIR / "config" / "finetune" / f"{MODEL}.yaml"
summary = run(config, DATA, OUT, smoke=SMOKE, code_dir=REPO_DIR)"""),
        new_code_cell("""metrics = summary["metrics"]
for key in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10"):
    print(f"{key:<14} base {metrics['base_' + key]:.4f}  tuned {metrics['tuned_' + key]:.4f}")
print("frozen layers:", summary["training"]["frozen_layers"])
print("training minutes:", round(summary["training"]["train_minutes"], 1))"""),
        new_code_cell("""import shutil

base = str(WORK / f"finetune_{MODEL}_export")
archive = shutil.make_archive(base, "zip", OUT.parent, OUT.name)
print(archive, round(Path(archive).stat().st_size / 1e6, 1), "MB")"""),
        new_markdown_cell(f"""## Next steps (on the PC)

1. Download `finetune_{MODEL}_export.zip` (Output tab of the saved version) into `runs/kaggle/`.
2. `docker compose up -d mlflow`, then, in the training venv:
   `python -m src.models.embedding.import_kaggle runs/kaggle/finetune_{MODEL}_export.zip`
   (run in `tujya-rag-finetuning`, model unzipped into `data/models/embeddings/`).
3. Weights: `dvc add data/models/embeddings/finetune-{MODEL}` on `{BRANCH}` (never git).
4. Step 3.3: add the model to the retrieval grid (`--embedder local:...`), see
   docs/module2/kaggle_finetuning.md."""),
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


def write(path: Path = PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((nbformat.writes(build()) + "\n").encode("utf-8"))
    return path


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(prog="python -m src.models.embedding.notebook").parse_args(argv)
    print(write())


if __name__ == "__main__":
    main()
