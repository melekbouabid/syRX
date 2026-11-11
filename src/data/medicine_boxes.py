"""Medicine box photos of step 2.7: download, ground truth, out-of-distribution images.

    python -m src.data.medicine_boxes download      # Italian-OTC-medicines (Hugging Face)
    python -m src.data.medicine_boxes off-topic     # training venv: scikit-image samples
    python -m src.data.medicine_boxes prepare       # DVC stage prepare_boxes

Data (docs/module1/boites.md, decision D108; all DVC, never git):

- data/raw/medicine_boxes/italian_otc/   mennox/Italian-OTC-medicines at a fixed revision
  (Apache-2.0, anonymous download): 718 photos of Italian OTC boxes + medicines.json
  (commercial name, EAN, SmPC sections in Italian, "Principi attivi" among them);
- data/raw/medicine_boxes/user_photos/   the owner's photos of boxes sold in Tunisia
  (Panadol, Doliprane...): the final test set, waiting for the photos (README in the folder
  says what to put there);
- data/raw/off_topic/                    out-of-distribution photos: 16 scikit-image samples
  (CC0 or public domain, one licence line each in licences.csv) and 3 generated images.

`prepare` writes data/processed/medicine_boxes/eval.csv: one row per box photo with the
ground truth the OCR engines are scored against, taken from the metadata, never from the
photo: the brand (first word of the commercial name, empty for a generic named after its
ingredient) and the active ingredients (the "Principi attivi" text matched with the
interaction table's names and config/inn_aliases.csv, the brand table left out so that the
brand route is not scored against itself).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from app.boxes import OcrLine, TextMatcher, fold, load_inn_aliases, words

ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "mennox/Italian-OTC-medicines"
REVISION = "29ebcd7385f17a4cdffdc65b5fc7978c543c7b53"  # 2026-01-22, the version evaluated
BOXES_DIR = ROOT / "data" / "raw" / "medicine_boxes"
ITALIAN_DIR = BOXES_DIR / "italian_otc"
USER_DIR = BOXES_DIR / "user_photos"
OFF_TOPIC_DIR = ROOT / "data" / "raw" / "off_topic"
INTERACTIONS = ROOT / "data" / "processed" / "pharma" / "interactions_with_any_drug.csv"
INN_ALIASES = ROOT / "config" / "inn_aliases.csv"
OUT_DIR = ROOT / "data" / "processed" / "medicine_boxes"
EVAL_COLUMNS = ("image", "product", "brand", "ingredients", "principi_attivi", "ean")

# scikit-image samples (skimage.data): file -> licence, from each function's docstring.
OFF_TOPIC = {
    "astronaut.png": "public domain (NASA)",
    "brick.png": "CC0 (CC0Textures Bricks25)",
    "camera.png": "CC0 (Lav Varshney)",
    "cell.png": "CC0",
    "chelsea.png": "CC0 (Stefan van der Walt)",
    "clock_motion.png": "public domain (Stefan van der Walt)",
    "coffee.png": "CC0 (Rachel Michetti)",
    "coins.png": "no known copyright restrictions",
    "grass.png": "CC0 (CC0Textures)",
    "gravel.png": "CC0 (CC0Textures Gravel04)",
    "horse.png": "CC0 (Andreas Preuss)",
    "hubble_deep_field.jpg": "public domain (NASA)",
    "ihc.png": "no known copyright restrictions",
    "retina.jpg": "CC0 1.0",
    "rocket.jpg": "public domain (SpaceX)",
    "text.png": "public domain",
}

USER_README = """\
Photos of medicine boxes taken by the owner (Panadol, Doliprane, Efferalgan, Augmentin...),
the final test set of step 2.7. Put one JPEG per box here and describe it in labels.csv:

    image,brand,ingredients
    panadol_1.jpg,Panadol,acetaminophen
    augmentin_1.jpg,Augmentin,amoxicillin;clavulanic acid

`ingredients` uses the names of the interaction table (English, `;` between two). Then:

    .venv-train/Scripts/python -m dvc add data/raw/medicine_boxes/user_photos
    .venv-train/Scripts/python -m src.models.ocr.evaluate --set user --engine rapidocr

(docs/module1/boites.md, section "Photos de l'utilisateur").
"""


def download(dest: Path = ITALIAN_DIR, revision: str = REVISION) -> Path:
    """Anonymous download of the public dataset at a fixed revision (no token, no account)."""
    from huggingface_hub import snapshot_download

    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"{dest} is not empty: remove it first or use `dvc pull`")
    snapshot_download(
        REPO_ID,
        repo_type="dataset",
        revision=revision,
        local_dir=dest,
        allow_patterns=["README.md", "medicines.json", "out/*.jpg"],
    )
    shutil.rmtree(dest / ".cache", ignore_errors=True)  # huggingface_hub's own bookkeeping
    USER_DIR.mkdir(parents=True, exist_ok=True)
    (USER_DIR / "README.md").write_text(USER_README, encoding="utf-8", newline="\n")
    return dest


def off_topic(dest: Path = OFF_TOPIC_DIR) -> Path:
    """Out-of-distribution photos: scikit-image samples + 3 generated images (seed 0)."""
    import cv2
    import skimage

    source = Path(skimage.__file__).parent / "data"
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, licence in OFF_TOPIC.items():
        shutil.copyfile(source / name, dest / name)
        rows.append({"image": name, "source": "scikit-image (skimage.data)", "licence": licence})
    rng = np.random.default_rng(0)
    generated = {
        "uniform_grey.png": np.full((480, 640, 3), 128, np.uint8),
        "noise.png": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
        "gradient.png": np.repeat(np.linspace(0, 255, 640, dtype=np.uint8)[None, :, None], 480, 0)
        .repeat(3, 2)
        .copy(),
    }
    for name, image in generated.items():
        cv2.imwrite(str(dest / name), image)
        rows.append({"image": name, "source": "generated (numpy, seed 0)", "licence": "CC0"})
    with (dest / "licences.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "source", "licence"])
        writer.writeheader()
        writer.writerows(rows)
    return dest


def interaction_names(path: Path = INTERACTIONS) -> set[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row[f"ingredient_{side}"].strip().lower()
            for row in csv.DictReader(handle)
            for side in ("a", "b")
        }


def ground_truth_matcher(names: set[str], aliases_file: Path = INN_ALIASES) -> TextMatcher:
    """Printed names only: no brand table, no RxNav."""
    return TextMatcher([], load_inn_aliases(aliases_file), names)


def brand_of(product: str, matcher: TextMatcher) -> str:
    """First word of the commercial name ("TACHIPIRINA*20CPR 500MG" -> "tachipirina"), or ""
    when the product is named after its ingredient ("PARACETAMOLO*SAND 20CPR")."""
    head = product.split("*")[0]
    first = words(head)[:1]
    if not first or len(first[0]) < 3:
        return ""
    if matcher.match([OcrLine(head, 1.0, (0, 0, 0, 0))], use_rxnav=False).ingredients:
        return ""
    return first[0]


def active_part(text: str) -> str:
    """The composition after "principio attivo" / "principi attivi" and before the
    excipients: flavours named earlier ("gusto Ginger") are not active ingredients."""
    folded = fold(text)
    for marker in ("principi attivi", "principio attivo"):
        if marker in folded:
            folded = folded.split(marker, 1)[1]
            break
    return folded.split("eccipient", 1)[0]


IMAGE_SIGNATURES = (bytes([0xFF, 0xD8, 0xFF]), bytes([0x89]) + b"PNG", b"RIFF")  # JPEG PNG WebP


def is_image(path: Path) -> bool:
    """By content, not by name: 5 of the dataset's .jpg files are saved HTML pages."""
    with path.open("rb") as handle:
        return handle.read(4).startswith(IMAGE_SIGNATURES)


def eval_rows(source: Path, matcher: TextMatcher) -> list[dict[str, str]]:
    records = json.loads((source / "medicines.json").read_text(encoding="utf-8"))
    rows, seen = [], set()
    for record in records:
        image = record.get("immagine", "")
        path = source / "out" / image
        if not image or image in seen or not path.is_file() or not is_image(path):
            continue
        seen.add(image)
        printed = (record.get("dettagli") or {}).get("Principi attivi", "")
        found = matcher.match([OcrLine(active_part(printed), 1.0, (0, 0, 0, 0))], use_rxnav=False)
        rows.append(
            {
                "image": image,
                "product": record.get("nome", ""),
                "brand": brand_of(record.get("nome", ""), matcher),
                "ingredients": ";".join(sorted(i.name for i in found.ingredients)),
                "principi_attivi": " ".join(printed.split())[:300],
                "ean": str(record.get("Codice EAN", "")),
            }
        )
    return rows


def prepare(source: Path = ITALIAN_DIR, out: Path = OUT_DIR) -> dict[str, object]:
    matcher = ground_truth_matcher(interaction_names())
    rows = eval_rows(source, matcher)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "eval.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVAL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with_ingredients = [r for r in rows if r["ingredients"]]
    counts = Counter(i for r in with_ingredients for i in r["ingredients"].split(";"))
    summary = {
        "dataset": f"{REPO_ID}@{REVISION[:7]}",
        "photos": len(rows),
        "not_an_image": sorted(p.name for p in (source / "out").iterdir() if not is_image(p)),
        "with_brand": sum(1 for r in rows if r["brand"]),
        "with_known_ingredient": len(with_ingredients),
        "without_known_ingredient": len(rows) - len(with_ingredients),
        "distinct_ingredients": len(counts),
        "top_ingredients": dict(counts.most_common(15)),
        "brands_in_brand_table": sorted(
            {fold(r["brand"]) for r in rows if r["brand"]} & _brand_table_keys()
        ),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def _brand_table_keys(path: Path = ROOT / "config" / "brand_aliases.csv") -> set[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {fold(row["brand"]) for row in csv.DictReader(handle)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.data.medicine_boxes")
    parser.add_argument("command", choices=("download", "off-topic", "prepare"))
    args = parser.parse_args(argv)
    if args.command == "download":
        print(download())
    elif args.command == "off-topic":
        print(off_topic())
    else:
        print(json.dumps(prepare(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
