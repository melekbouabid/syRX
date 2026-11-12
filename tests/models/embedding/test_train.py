"""Training helpers of the embedding fine-tuning (src/models/embedding/train.py), without
PyTorch: configuration, smoke sampling, layer freezing (step 3.5)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from src.models.embedding import pairs as P
from src.models.embedding import train

CONFIG = Path(__file__).resolve().parents[3] / "config" / "finetune" / "bge-m3.yaml"


def test_config_is_bge_m3_and_smoke_overrides_it() -> None:
    full = train.load_config(CONFIG, smoke=False)
    assert full["model"] == "BAAI/bge-m3" and full["seed"] == 0 and "smoke" not in full
    assert 0 < full["fallback_frozen_layers"] < 24  # bge-m3 has 24 layers
    smoke = train.load_config(CONFIG, smoke=True)
    assert smoke["model"] != full["model"] and smoke["epochs"] == 1
    assert smoke["max_train_pairs"] == train.SMOKE["max_train_pairs"]
    assert smoke["learning_rate"] == full["learning_rate"]  # what smoke does not set is kept


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_smoke_sampling_keeps_the_relevant_passages_of_the_validation_questions(
    tmp_path: Path,
) -> None:
    pair = {"query": "q", "positive": "p", "negative": "n"}
    write_jsonl(tmp_path / "train.jsonl", [{**pair, "positive_id": f"t{i}"} for i in range(50)])
    val = [{**pair, "positive_id": f"c{i}", "relevant_ids": [f"c{i}", f"c{i}b"]} for i in range(20)]
    write_jsonl(tmp_path / "val.jsonl", val)
    write_jsonl(
        tmp_path / "corpus.jsonl",
        [{"chunk_id": f"x{i}", "text": "t"} for i in range(500)]
        + [{"chunk_id": c, "text": "t"} for p in val for c in p["relevant_ids"]],
    )
    config = {"seed": 0, "max_train_pairs": 8, "max_val_pairs": 4, "max_corpus": 10}
    data = train.load_data(tmp_path, config)
    assert len(data["train"]) == 8 and len(data["val"]) == 4
    ids = {c["chunk_id"] for c in data["corpus"]}
    assert all(c in ids for p in data["val"] for c in p["relevant_ids"])
    assert len(ids) == 4 * 2 + 10
    again = train.load_data(tmp_path, config)  # seeded: the same sample
    assert [p["positive_id"] for p in again["val"]] == [p["positive_id"] for p in data["val"]]


def fake_model(layers: int) -> list[SimpleNamespace]:
    def module() -> SimpleNamespace:
        params = [SimpleNamespace(requires_grad=True) for _ in range(2)]
        return SimpleNamespace(parameters=lambda: params, params=params)

    backbone = SimpleNamespace(
        embeddings=module(), encoder=SimpleNamespace(layer=[module() for _ in range(layers)])
    )
    return [SimpleNamespace(auto_model=backbone)]


def test_freeze_lower_layers_keeps_the_upper_ones_trainable() -> None:
    model = fake_model(24)
    assert train.freeze_lower_layers(model, 16) == 16
    backbone = model[0].auto_model
    assert not any(p.requires_grad for p in backbone.embeddings.params)
    trainable = [all(p.requires_grad for p in layer.params) for layer in backbone.encoder.layer]
    assert trainable == [False] * 16 + [True] * 8
    untouched = fake_model(4)
    assert train.freeze_lower_layers(untouched, 0) == 0  # full fine-tuning
    assert all(p.requires_grad for p in untouched[0].auto_model.embeddings.params)
    assert train.freeze_lower_layers(fake_model(4), 10) == 4  # never more than the model has


def test_val_share_and_seed_come_from_params(tmp_path: Path) -> None:
    params = tmp_path / "params.yaml"
    params.write_text(yaml.safe_dump({"finetune": {"seed": 3, "val_share": 0.5}}), encoding="utf-8")
    assert P.read_params(params) == {"seed": 3, "val_share": 0.5}
    params.write_text("other: 1\n", encoding="utf-8")
    assert P.read_params(params) == {"seed": 0, "val_share": P.VAL_SHARE}
    rows = [{"doc_id": f"d{i}"} for i in range(400)]
    half = P.split(rows, seed=3, val_share=0.5)
    assert 150 < len(half["val"]) < 250
    assert P.split(rows, seed=3, val_share=0.0)["val"] == []
