"""Package the fine-tuning pairs as a private Kaggle dataset (step 3.5).

    # project venv, after `dvc repro finetune_pairs`
    python -m src.models.embedding.package_kaggle      # runs/kaggle/tujya-rag-finetune-pairs.zip

The zip (and the folder next to it, the same files unzipped) holds train.jsonl and val.jsonl
(stage `finetune_pairs`), corpus.jsonl (every passage of the index, `title\\ntext` as
src/rag/index.py embeds it: the validation questions are searched in the whole corpus) and
MANIFEST.json (name, sizes, md5 of each file, commit).
Upload it on Kaggle as a private dataset named `tujya-rag-finetune-pairs`
(docs/module2/kaggle_finetuning.md). The corpus is public data (openFDA labels, PubMed
abstracts, DDInter pairs; their licences: docs/module2/corpus.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from src.models.embedding.pairs import CHUNKS, OUT, passage, read_chunks

ROOT = Path(__file__).resolve().parents[3]
NAME = "tujya-rag-finetune-pairs"
TARGET = ROOT / "runs" / "kaggle" / f"{NAME}.zip"


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def corpus_lines(chunks_path: Path = CHUNKS) -> bytes:
    rows = [
        json.dumps({"chunk_id": c["chunk_id"], "text": passage(c)}, ensure_ascii=False)
        for c in read_chunks(chunks_path)
    ]
    return ("\n".join(rows) + "\n").encode("utf-8")


def package(pairs: Path = OUT, target: Path = TARGET, chunks_path: Path = CHUNKS) -> Path:
    files = {
        "train.jsonl": (pairs / "train.jsonl").read_bytes(),
        "val.jsonl": (pairs / "val.jsonl").read_bytes(),
        "corpus.jsonl": corpus_lines(chunks_path),
    }
    commit = subprocess.run(  # noqa: S603
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "name": NAME,
        "summary": json.loads((pairs / "summary.json").read_text(encoding="utf-8")),
        "files": {name: {"bytes": len(data), "md5": md5(data)} for name, data in files.items()},
        "packed_from_commit": commit,
    }
    files["MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    folder = target.with_suffix("")  # the same files unzipped: the local smoke run reads them
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
            (folder / name).write_bytes(data)
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.embedding.package_kaggle")
    parser.add_argument("--out", type=Path, default=TARGET)
    args = parser.parse_args(argv)
    path = package(target=args.out)
    print(path, round(path.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
