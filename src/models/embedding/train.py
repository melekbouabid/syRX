"""Fine-tune the embedding model of the RAG on the question / passage pairs (step 3.5).

    # Kaggle notebook notebooks/kaggle/finetune_bge-m3.ipynb (GPU T4 16 GB), or locally:
    python -m src.models.embedding.train --config config/finetune/bge-m3.yaml --data <dir>
    python -m src.models.embedding.train --config config/finetune/bge-m3.yaml --data <dir> --smoke

Steps (sentence-transformers, seed fixed):
1. retrieval of the validation questions over the whole corpus with the base model
   (Recall@1/5/10, MRR@10, nDCG@10: src/models/embedding/evaluate.py);
2. training with CachedMultipleNegativesRankingLoss: each question must rank its passage
   above its hard negative and above every other passage of the batch; the cache (GradCache)
   keeps a large batch of in-batch negatives within the memory of a T4;
3. the same retrieval with the fine-tuned model; then the model (safetensors, FP16), the
   training history and run.json (config, environment, metrics before / after, timings) are
   written to the export folder, zipped by the notebook.

Memory: full fine-tuning of bge-m3 (568 M parameters, XLM-RoBERTa large) needs about 9 GB for
the weights, gradients and AdamW states in FP32, plus the activations of one mini-batch
(FP16 autocast). If the first steps run out of GPU memory, the run restarts with the lower
`fallback_frozen_layers` layers and the embeddings frozen (only the upper layers train), after
the first attempt's model is released; run.json records which mode ran (`frozen_layers`).
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.models.embedding.evaluate import retrieval_metrics

SMOKE = {"max_train_pairs": 64, "max_val_pairs": 16, "max_corpus": 200, "epochs": 1}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path, smoke: bool) -> dict[str, Any]:
    config = dict(yaml.safe_load(path.read_text(encoding="utf-8")))
    if smoke:
        config.update(config.get("smoke") or {})
        config.update(SMOKE)
    config.pop("smoke", None)
    return config


def load_data(data: Path, config: dict[str, Any]) -> dict[str, Any]:
    train = read_jsonl(data / "train.jsonl")
    val = read_jsonl(data / "val.jsonl")
    corpus = read_jsonl(data / "corpus.jsonl")
    rng = random.Random(int(config["seed"]))
    if config.get("max_train_pairs"):
        train = rng.sample(train, min(len(train), int(config["max_train_pairs"])))
    if config.get("max_val_pairs"):
        val = rng.sample(val, min(len(val), int(config["max_val_pairs"])))
    if config.get("max_corpus"):
        keep = {c for p in val for c in p.get("relevant_ids") or [p["positive_id"]]}
        rest = [c for c in corpus if c["chunk_id"] not in keep]
        corpus = [c for c in corpus if c["chunk_id"] in keep] + rest[: int(config["max_corpus"])]
    return {"train": train, "val": val, "corpus": corpus}


def freeze_lower_layers(model: Any, n: int) -> int:
    """Freeze the embeddings and the first `n` transformer layers; returns the layers frozen."""
    if n <= 0:
        return 0
    backbone = model[0].auto_model
    for p in backbone.embeddings.parameters():
        p.requires_grad = False
    layers = backbone.encoder.layer
    for layer in layers[: min(n, len(layers))]:
        for p in layer.parameters():
            p.requires_grad = False
    return min(n, len(layers))


def train_once(
    config: dict[str, Any], data: dict[str, Any], out: Path, frozen: int
) -> tuple[Any, dict[str, Any]]:
    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    seed_everything(int(config["seed"]))
    model = SentenceTransformer(
        config["model"], device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.max_seq_length = int(config["max_seq_length"])
    frozen = freeze_lower_layers(model, frozen)
    # A pair without a hard negative is left out: its positive as negative would ask the model
    # to rank a passage above itself (pairs.py gives every pair one; this is a guard).
    rows = [pair for pair in data["train"] if pair["negative"]]
    columns: dict[str, list[str]] = {"anchor": [], "positive": [], "negative": []}
    for pair in rows:
        columns["anchor"].append(pair["query"])
        columns["positive"].append(pair["positive"])
        columns["negative"].append(pair["negative"])
    dataset = Dataset.from_dict(columns)
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=int(config["mini_batch_size"]))
    cuda = torch.cuda.is_available()
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=float(config["epochs"]),
        per_device_train_batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        warmup_ratio=float(config["warmup_ratio"]),
        weight_decay=float(config["weight_decay"]),
        fp16=cuda and bool(config.get("fp16", True)),
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # one question once per batch
        logging_steps=int(config.get("logging_steps", 10)),
        save_strategy="no",
        report_to=[],
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss)
    start = time.perf_counter()
    trainer.train()
    history = [h for h in trainer.state.log_history if "loss" in h or "train_loss" in h]
    return model, {
        "frozen_layers": frozen,
        "pairs_trained": len(rows),
        "train_minutes": (time.perf_counter() - start) / 60,
        "steps": int(trainer.state.global_step),
        "history": history,
    }


def evaluate(model: Any, data: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    corpus = data["corpus"]
    ids = [c["chunk_id"] for c in corpus]
    start = time.perf_counter()
    docs = model.encode(
        [c["text"] for c in corpus],
        batch_size=int(config.get("encode_batch_size", 32)),
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    queries = model.encode(
        [p["query"] for p in data["val"]], normalize_embeddings=True, convert_to_numpy=True
    )
    relevant = [p.get("relevant_ids") or [p["positive_id"]] for p in data["val"]]
    metrics = retrieval_metrics(queries @ docs.T, ids, relevant)
    metrics["encode_seconds"] = time.perf_counter() - start
    return metrics


def environment() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "platform": "kaggle" if Path("/kaggle/working").is_dir() else platform.system().lower(),
    }
    for name in ("sentence_transformers", "transformers", "datasets"):
        try:
            info[name] = __import__(name).__version__
        except ImportError:
            info[name] = ""
    return info


def code_info(repo: Path | None) -> dict[str, Any]:
    """Commit and branch of the code that ran (the clone on Kaggle)."""
    import subprocess

    if repo is None:
        return {}

    def git(*args: str) -> str:
        done = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo), *args],  # noqa: S607
            capture_output=True,
            text=True,
        )
        return done.stdout.strip()

    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def run(
    config_path: Path,
    data_dir: Path,
    out: Path,
    *,
    smoke: bool = False,
    code_dir: Path | None = None,
) -> dict[str, Any]:
    import torch
    from sentence_transformers import SentenceTransformer

    config = load_config(config_path, smoke)
    data = load_data(data_dir, config)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = SentenceTransformer(config["model"], device=device)
    base.max_seq_length = int(config["max_seq_length"])
    before = evaluate(base, data, config)
    print("before:", {k: round(v, 4) for k, v in before.items()}, flush=True)
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    frozen = int(config.get("frozen_layers", 0))
    trained: tuple[Any, dict[str, Any]] | None = None
    # On OOM, retried below, once the traceback (and the first model it holds) is released.
    with contextlib.suppress(torch.cuda.OutOfMemoryError):
        trained = train_once(config, data, out, frozen)
    if trained is None:
        gc.collect()
        torch.cuda.empty_cache()
        frozen = int(config["fallback_frozen_layers"])
        print(f"out of GPU memory: retry with the lower {frozen} layers frozen", flush=True)
        trained = train_once(config, data, out, frozen)
        trained[1]["fallback"] = True
    model, training = trained
    after = evaluate(model, data, config)
    print("after:", {k: round(v, 4) for k, v in after.items()}, flush=True)
    if torch.cuda.is_available():
        model.half()  # the export is FP16: half the size, the same ranking
    model.save(str(out / "model"))
    summary = {
        "schema": 1,
        "task": "embedding",
        "smoke": smoke,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": config,
        "data": {
            "train_pairs": len(data["train"]),
            "val_pairs": len(data["val"]),
            "corpus": len(data["corpus"]),
            "manifest": _manifest(data_dir),
        },
        "environment": environment(),
        "code": code_info(code_dir),
        "training": training,
        "metrics": {
            **{f"base_{k}": v for k, v in before.items()},
            **{f"tuned_{k}": v for k, v in after.items()},
            "gain_recall_at_5": after["recall_at_5"] - before["recall_at_5"],
            "gain_mrr_at_10": after["mrr_at_10"] - before["mrr_at_10"],
        },
        "minutes": (time.perf_counter() - started) / 60,
    }
    (out / "run.json").write_text(
        json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8"
    )
    return summary


def _manifest(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "MANIFEST.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.embedding.train")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs") / "finetune")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    summary = run(args.config, args.data, args.out, smoke=args.smoke, code_dir=Path.cwd())
    print(json.dumps(summary["metrics"], indent=1))


if __name__ == "__main__":
    main()
