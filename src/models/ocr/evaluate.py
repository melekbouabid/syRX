"""Score an OCR engine on medicine box photos, log it to MLflow, promote the best (step 2.7).

    # training venv, MLflow server running (docker compose up -d mlflow), from the branch
    # experiment/ocr-<engine> that holds config/ocr/<engine>.yaml
    python -m src.models.ocr.evaluate --engine rapidocr
    python -m src.models.ocr.evaluate --engine easyocr --smoke      # 20 photos, no Registry
    python -m src.models.ocr.evaluate --engine rapidocr --set web   # free photos (Commons)
    python -m src.models.ocr.evaluate --engine rapidocr --set user  # owner's photos (later)

Three sets (docs/module1/boites.md, decisions D108-D111):

- `box`: the 718 Italian OTC box photos (data/processed/medicine_boxes/eval.csv, stage
  prepare_boxes). The ground truth comes from the metadata, not from the photo: brand
  (first word of the commercial name) and active ingredients ("Principi attivi");
- `pill`: 200 C3PI photos of the detector's comparison split (test_holdout.txt, D93),
  never seen by the served models;
- `off_topic`: 19 photos that are neither (data/raw/off_topic, CC0 / public domain).

`--set web` replaces the Italian boxes by real box photos under a free licence from
Wikimedia Commons (data/raw/medicine_boxes/web_commons, Panadol among them, D113) and
`--set user` by the owner's photos; both read a labels.csv (image, brand, ingredients).

Every photo goes through the API's own code (app.boxes: OCR, TextMatcher with the brand
table and the INN aliases, decide_kind with the pill detector of models/); RxNav is off so
that the score is reproducible offline. Metrics (all in [0, 1] except latency):

- `ingredient_recall` (promotion metric): box photos with a known ingredient where at least
  one true ingredient is found; `ingredient_full_recall`: all of them;
- `ingredient_precision`: found ingredients that are true (micro), and
  `wrong_ingredient_rate`: photos with at least one ingredient that is not on the box;
- `brand_read_rate`: the brand is among the words read (photos with a brand);
- `box_as_box_rate`, `box_as_pill_rate`, `pill_as_pill_rate`, `pill_as_box_rate`,
  `off_topic_refused_rate` (no pill class and no ingredient): the out-of-distribution check;
- `latency_ms_median`, `latency_ms_p95`: OCR of 30 box photos, 3 rounds with a pause in
  between (the PC throttles under continuous load, D96).

A run of the full sets logs the engine as a pyfunc LoggedModel (src/models/ocr/
pyfunc_model.py), registers it in `tujya-box-ocr` and calls promote_if_better on
`ingredient_recall` (params.yaml `compare.ocr`). `--smoke` never reaches the Registry (D94).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.boxes import (
    BRAND_MIN_SCORE,
    Choices,
    KindThresholds,
    OcrEngine,
    OcrLine,
    TextMatcher,
    decide_kind,
    load_brands,
    load_inn_aliases,
    words,
)
from app.inference import PillScanner, decode_image

ROOT = Path(__file__).resolve().parents[3]
PARAMS = ROOT / "params.yaml"
CONFIG_DIR = ROOT / "config" / "ocr"
EVAL_CSV = ROOT / "data" / "processed" / "medicine_boxes" / "eval.csv"
BOX_IMAGES = ROOT / "data" / "raw" / "medicine_boxes" / "italian_otc" / "out"
USER_DIR = ROOT / "data" / "raw" / "medicine_boxes" / "user_photos"
WEB_DIR = ROOT / "data" / "raw" / "medicine_boxes" / "web_commons"
LABELLED_SETS = {"user": USER_DIR, "web": WEB_DIR}
DATASETS = {
    "box": ("italian-otc-medicines@29ebcd7", "data/processed/medicine_boxes"),
    "web": ("wikimedia-commons-boxes", "data/raw/medicine_boxes/web_commons"),
    "user": ("owner-photos", "data/raw/medicine_boxes/user_photos"),
}
OFF_TOPIC_DIR = ROOT / "data" / "raw" / "off_topic"
PILL_DIR = ROOT / "data" / "processed" / "c3pi" / "datasets_v2" / "detect"
INTERACTIONS = ROOT / "data" / "processed" / "pharma" / "interactions_with_any_drug.csv"
BRANDS = ROOT / "config" / "brand_aliases.csv"
INN_ALIASES = ROOT / "config" / "inn_aliases.csv"
MODELS_DIR = ROOT / "models"
EXPERIMENT = "tujya-boxes"
REGISTERED = "tujya-box-ocr"
PILL_SAMPLE = 200
LATENCY_IMAGES = 30
LATENCY_ROUNDS = 3
SMOKE_IMAGES = 20


@dataclass
class Sample:
    set: str  # box, pill, off_topic, user, web
    image: Path
    brand: str = ""
    truth: list[str] = field(default_factory=list)


@dataclass
class Outcome:
    set: str
    image: str
    brand: str
    truth: list[str]
    found: list[str]
    methods: list[str]
    brand_read: bool | None
    kind: str
    pill_confidence: float
    text_chars: int
    ocr_ms: float
    text: str


# ----------------------------------------------------------------------------- data


def box_samples(path: Path = EVAL_CSV, images: Path = BOX_IMAGES) -> list[Sample]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            Sample("box", images / row["image"], row["brand"], _split(row["ingredients"]))
            for row in csv.DictReader(handle)
        ]


def labelled_samples(folder: Path, set_name: str) -> list[Sample]:
    """Photos described by folder/labels.csv (image, brand, ingredients), empty without it."""
    labels = folder / "labels.csv"
    if not labels.is_file():
        return []
    with labels.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        Sample(set_name, folder / row["image"], row.get("brand", ""), _split(row["ingredients"]))
        for row in rows
    ]


def user_samples(folder: Path = USER_DIR) -> list[Sample]:
    """The owner's photos, empty until they exist."""
    return labelled_samples(folder, "user")


def web_samples(folder: Path = WEB_DIR) -> list[Sample]:
    """Real box photos under a free licence from Wikimedia Commons (D113)."""
    return labelled_samples(folder, "web")


def pill_samples(folder: Path = PILL_DIR, n: int = PILL_SAMPLE, seed: int = 0) -> list[Sample]:
    names = (folder / "test_holdout.txt").read_text(encoding="utf-8").split()
    chosen = sorted(random.Random(seed).sample(sorted(names), min(n, len(names))))
    return [Sample("pill", folder / "images" / "test" / name) for name in chosen]


def off_topic_samples(folder: Path = OFF_TOPIC_DIR) -> list[Sample]:
    with (folder / "licences.csv").open(encoding="utf-8", newline="") as handle:
        return [Sample("off_topic", folder / row["image"]) for row in csv.DictReader(handle)]


def _split(value: str) -> list[str]:
    return [v.strip() for v in value.split(";") if v.strip()]


def interaction_names(path: Path = INTERACTIONS) -> set[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row[f"ingredient_{s}"].strip().lower() for row in csv.DictReader(handle) for s in "ab"
        }


def matcher() -> TextMatcher:
    """The API's matcher without RxNav (offline, reproducible)."""
    return TextMatcher(load_brands(BRANDS), load_inn_aliases(INN_ALIASES), interaction_names())


# ---------------------------------------------------------------------------- scoring


def brand_read(lines: Sequence[OcrLine], brand: str) -> bool:
    """The brand is among the words read (same similarity rule as the brand table)."""
    choices = Choices([brand])
    for line in lines:
        found = words(line.text)
        for size in (1, 2):
            for start in range(len(found) - size + 1):
                gram = " ".join(found[start : start + size])
                if choices.similar(gram, BRAND_MIN_SCORE) or (
                    len(brand) >= 5 and gram.startswith(brand)
                ):
                    return True
    return False


def run_samples(
    engine: OcrEngine,
    samples: Iterable[Sample],
    text_matcher: TextMatcher,
    scanner: PillScanner,
    thresholds: KindThresholds,
) -> list[Outcome]:
    outcomes = []
    for sample in samples:
        image = decode_image(sample.image.read_bytes())
        ocr = engine.read(image)
        match = text_matcher.match(ocr.lines, use_rxnav=False)
        confidences = [c for _, c in scanner.detect(image)]
        outcomes.append(
            Outcome(
                set=sample.set,
                image=sample.image.name,
                brand=sample.brand,
                truth=sample.truth,
                found=[i.name for i in match.ingredients],
                methods=sorted({i.method for i in match.ingredients}),
                brand_read=brand_read(ocr.lines, sample.brand) if sample.brand else None,
                kind=decide_kind(confidences, ocr, match, thresholds),
                pill_confidence=round(max(confidences, default=0.0), 4),
                text_chars=ocr.text_chars(),
                ocr_ms=round(ocr.ocr_ms, 1),
                text=" | ".join(line.text for line in ocr.lines)[:500],
            )
        )
    return outcomes


def _rate(values: Iterable[bool]) -> float | None:
    items = list(values)
    return round(sum(items) / len(items), 4) if items else None


def score(outcomes: Sequence[Outcome], box_set: str = "box") -> dict[str, float]:
    boxes = [o for o in outcomes if o.set == box_set]
    known = [o for o in boxes if o.truth]
    pills = [o for o in outcomes if o.set == "pill"]
    off = [o for o in outcomes if o.set == "off_topic"]
    found_total = sum(len(o.found) for o in known)
    metrics = {
        "ingredient_recall": _rate(bool(set(o.found) & set(o.truth)) for o in known),
        "ingredient_full_recall": _rate(set(o.truth) <= set(o.found) for o in known),
        "ingredient_precision": (
            round(sum(len(set(o.found) & set(o.truth)) for o in known) / found_total, 4)
            if found_total
            else None
        ),
        "wrong_ingredient_rate": _rate(bool(set(o.found) - set(o.truth)) for o in known),
        "brand_read_rate": _rate(bool(o.brand_read) for o in boxes if o.brand_read is not None),
        "box_recognised_rate": _rate(bool(o.found) for o in boxes),
        "box_as_box_rate": _rate(o.kind == "box" for o in boxes),
        "box_as_pill_rate": _rate(o.kind == "pill" for o in boxes),
        "pill_as_pill_rate": _rate(o.kind == "pill" for o in pills),
        "pill_as_box_rate": _rate(o.kind == "box" for o in pills),
        "off_topic_refused_rate": _rate(o.kind != "pill" and not o.found for o in off),
        "ocr_ms_mean": round(statistics.fmean(o.ocr_ms for o in outcomes), 1) if outcomes else None,
        "n_box": len(boxes),
        "n_box_known_ingredient": len(known),
        "n_pill": len(pills),
        "n_off_topic": len(off),
    }
    return {k: float(v) for k, v in metrics.items() if v is not None}


def latency(
    engine: OcrEngine, samples: Sequence[Sample], rounds: int, pause: float
) -> dict[str, float]:
    """OCR time of the same photos in `rounds` rounds separated by `pause` seconds."""
    images = [decode_image(s.image.read_bytes()) for s in samples]
    engine.read(images[0])  # warm-up: model loading and first allocations
    timings: list[float] = []
    for index in range(rounds):
        if index:
            time.sleep(pause)
        for image in images:
            timings.append(engine.read(image).ocr_ms)
    timings.sort()
    return {
        "latency_ms_median": round(statistics.median(timings), 1),
        "latency_ms_p95": round(timings[min(len(timings) - 1, int(0.95 * len(timings)))], 1),
        "latency_ms_mean": round(statistics.fmean(timings), 1),
    }


# ----------------------------------------------------------------------------- MLflow


def load_params() -> dict[str, Any]:
    params = yaml.safe_load(PARAMS.read_text(encoding="utf-8"))
    return {"compare": params["compare"]["ocr"], "boxes": params["boxes"]}


def write_outcomes(outcomes: Sequence[Outcome], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(outcomes[0])))
        writer.writeheader()
        for outcome in outcomes:
            row = asdict(outcome)
            row["truth"], row["found"] = ";".join(outcome.truth), ";".join(outcome.found)
            row["methods"] = ";".join(outcome.methods)
            writer.writerow(row)


def log_run(
    config: dict[str, Any],
    metrics: dict[str, float],
    outcomes: Sequence[Outcome],
    params: dict[str, Any],
    *,
    smoke: bool,
    sample_set: str,
) -> dict[str, Any]:
    import mlflow

    from src.utils import mlflow_utils as mu

    mu.configure()
    engine = config["engine"]
    name = (
        f"{'smoke-' if smoke else ''}ocr-{engine}{'' if sample_set == 'box' else '-' + sample_set}"
    )
    tags = {"task": "ocr", "engine": engine, "set": sample_set, "smoke": str(smoke).lower()}
    flat = {f"engine.{k}": v for k, v in config.items()}
    flat.update({f"boxes.{k}": v for k, v in params["boxes"].items()})
    with mu.start_tracked_run(
        EXPERIMENT,
        name,
        seed=0,
        dataset=DATASETS[sample_set][0],
        dataset_path=DATASETS[sample_set][1],
        params=flat,
        tags=tags,
    ) as run:
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            per_image = Path(tmp) / "per_image.csv"
            write_outcomes(outcomes, per_image)
            mlflow.log_artifact(str(per_image))
            for path in (BRANDS, INN_ALIASES, CONFIG_DIR / f"{engine}.yaml"):
                if path.is_file():
                    mlflow.log_artifact(str(path), "config")
        result: dict[str, Any] = {"run_id": run.info.run_id, "run_name": name, "metrics": metrics}
        if smoke or sample_set != "box":
            return result
        info = mlflow.pyfunc.log_model(
            name="ocr_engine",
            python_model=str(Path(__file__).with_name("pyfunc_model.py")),
            model_config=config,
            code_paths=[str(Path(__file__).parent)],
            pip_requirements=_requirements(engine),
        )
        mlflow.log_metrics(metrics, model_id=info.model_id)
    compare = params["compare"]
    version = mu.register(
        info.model_uri,
        REGISTERED,
        tags={"engine": engine, "dataset": "italian-otc-medicines@29ebcd7"},
        description=f"OCR engine {engine} for medicine boxes (step 2.7)",
    )
    promotion = mu.promote_if_better(
        REGISTERED, version.version, compare["metric"], float(compare["min_gain"])
    )
    result.update(version=version.version, promotion=asdict(promotion))
    return result


def _requirements(engine: str) -> list[str]:
    from importlib import metadata

    packages = {
        "rapidocr": ["rapidocr", "onnxruntime", "rapidfuzz"],
        "easyocr": ["easyocr", "torch", "rapidfuzz"],
    }
    pins = []
    for package in packages.get(engine, []):
        try:
            pins.append(f"{package}=={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            pins.append(package)
    return pins


# ------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    from src.models.ocr.engines import make_engine
    from src.utils.console import console_utf8

    console_utf8()
    parser = argparse.ArgumentParser(prog="python -m src.models.ocr.evaluate")
    parser.add_argument("--engine", required=True, help="config/ocr/<engine>.yaml")
    parser.add_argument("--set", default="box", choices=("box", "user", "web"))
    parser.add_argument("--smoke", action="store_true", help="20 photos per set, no Registry")
    parser.add_argument("--no-mlflow", action="store_true", help="print the metrics only")
    parser.add_argument("--pause", type=float, default=20.0, help="seconds between latency rounds")
    args = parser.parse_args(argv)

    config = yaml.safe_load((CONFIG_DIR / f"{args.engine}.yaml").read_text(encoding="utf-8"))
    params = load_params()
    thresholds = KindThresholds(
        pill_confidence=float(params["boxes"]["pill_confidence"]),
        box_text_chars=int(params["boxes"]["box_text_chars"]),
    )
    if args.set == "box":
        boxes = box_samples()
    else:
        boxes = labelled_samples(LABELLED_SETS[args.set], args.set)
    if not boxes:
        folder = LABELLED_SETS.get(args.set, USER_DIR)
        raise SystemExit(f"no photo in the set {args.set!r} (see {folder / 'README.md'})")
    samples = boxes + pill_samples() + off_topic_samples()
    if args.smoke:
        by_set: dict[str, list[Sample]] = {}
        for sample in samples:
            by_set.setdefault(sample.set, []).append(sample)
        samples = [s for group in by_set.values() for s in group[:SMOKE_IMAGES]]
    engine = make_engine(config)
    scanner = PillScanner.from_dir(MODELS_DIR)
    started = time.perf_counter()
    outcomes = run_samples(engine, samples, matcher(), scanner, thresholds)
    metrics = score(outcomes, args.set)
    known = [s for s in boxes if s.truth][: (5 if args.smoke else LATENCY_IMAGES)]
    metrics.update(latency(engine, known, 1 if args.smoke else LATENCY_ROUNDS, args.pause))
    metrics["eval_minutes"] = round((time.perf_counter() - started) / 60, 2)
    if args.no_mlflow:
        print(json.dumps(metrics, indent=2))
        return
    result = log_run(config, metrics, outcomes, params, smoke=args.smoke, sample_set=args.set)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
