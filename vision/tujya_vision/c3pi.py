"""NLM C3PI pill images: metadata index, class selection, download and splits.

C3PI publishes one XML file per image folder (`ALLXML/PillProjectDiscN.xml`) describing
every image: NDC, part, generic and proprietary names, file name, size, SHA-1, image type
and, for consumer photos, camera, lighting and background. This module turns those files
into one flat table and builds the training subset from it. It only uses the standard
library so that it runs (and is tested) without the vision venv.

Docs: docs/DATASET_C3PI.md. Licence: ODbL, research and development use only.
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from pathlib import Path

BASE_URL = "https://data.lhncbc.nlm.nih.gov/public/Pills/"
USER_AGENT = "tujya-student-project/0.1 (pill identification research; NLM C3PI)"

# Consumer-grade photos: one pill photographed with a phone or consumer camera.
CONSUMER = "C3PI_Test"
# Small reference JPGs served by the former RxImage API (one pill side on a plain background).
REFERENCE_JPG = ("MC_API_RXNAV_V1.3", "MC_API_NLMIMAGE_V1.3", "MC_SPL_SPLIMAGE_V3.0")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ImageRecord:
    disc: str
    file_name: str
    file_type: str
    file_size: int
    sha1: str
    image_type: str  # <Class> for C3PI photos, <Layout> for processed / API images
    ndc11: str
    ndc9: str
    part: int
    parts: int
    generic_name: str
    proprietary_name: str
    labeled_by: str
    imprint: str
    color: str  # several colours joined with ";"
    shape: str
    size_mm: str
    camera: str
    illumination: str
    background: str

    @property
    def path(self) -> str:
        """Location relative to BASE_URL, e.g. PillProjectDisc1/images/X.JPG."""
        return f"{self.disc}/images/{self.file_name}"

    @property
    def class_key(self) -> str:
        """A pill product: labeler + product code (NDC9) and part (kits have several parts).

        NDC9 rather than NDC11 because the last two digits only say which package the pill
        came in; the pill itself is the same.
        """
        return f"{self.ndc9}_{self.part}"


def _text(node: ET.Element, tag: str) -> str:
    child = node.find(tag)
    return (child.text or "").strip() if child is not None else ""


def parse_disc_xml(path: Path) -> Iterator[ImageRecord]:
    """Yield one record per <Image> in one ALLXML file."""
    disc = path.stem
    root = ET.parse(path).getroot()  # noqa: S314 - NLM file; expat limits entity expansion
    for image in root.iter("Image"):
        file_node = image.find("File")
        if file_node is None:
            continue
        ndc11 = _text(image, "NDC11")
        yield ImageRecord(
            disc=disc,
            file_name=_text(file_node, "Name"),
            file_type=_text(file_node, "Type").upper(),
            file_size=int(_text(file_node, "Size") or 0),
            sha1=_text(file_node, "Sha1").lower(),
            image_type=_text(image, "Class") or _text(image, "Layout"),
            ndc11=ndc11,
            ndc9=_text(image, "NDC9") or ndc11[:9],
            part=int(_text(image, "Part") or 1),
            parts=int(_text(image, "Parts") or 1),
            generic_name=_text(image, "GenericName"),
            proprietary_name=_text(image, "ProprietaryName"),
            labeled_by=_text(image, "LabeledBy"),
            imprint=_text(image, "Imprint"),
            color=";".join((c.text or "").strip() for c in image.findall("Color")),
            shape=_text(image, "Shape"),
            # <Size> also appears inside <File>; the direct child is the pill size in mm.
            size_mm=_text(image, "Size"),
            camera=_text(image, "Camera"),
            illumination=_text(image, "Illumination"),
            background=_text(image, "Background"),
        )


def build_index(xml_dir: Path) -> list[ImageRecord]:
    def disc_number(p: Path) -> int:
        digits = re.sub(r"\D", "", p.stem)
        return int(digits) if digits else 0

    records: list[ImageRecord] = []
    for path in sorted(xml_dir.glob("PillProjectDisc*.xml"), key=disc_number):
        records.extend(parse_disc_xml(path))
    return records


def write_index(records: Iterable[ImageRecord], out: Path) -> None:
    names = [f.name for f in fields(ImageRecord)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*names, "path", "class_key"])
        writer.writeheader()
        for record in records:
            writer.writerow({**asdict(record), "path": record.path, "class_key": record.class_key})


def read_index(path: Path) -> list[ImageRecord]:
    names = {f.name for f in fields(ImageRecord)}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        values = {k: v for k, v in row.items() if k in names}
        out.append(
            ImageRecord(
                **{
                    **values,
                    "file_size": int(values["file_size"]),
                    "part": int(values["part"]),
                    "parts": int(values["parts"]),
                }
            )
        )
    return out


def is_consumer_photo(record: ImageRecord) -> bool:
    return record.image_type == CONSUMER and record.file_type == "JPG"


@dataclass(frozen=True)
class PillClass:
    class_id: int
    class_key: str
    ndc9: str
    part: int
    ndc11_list: str  # every package NDC11 seen for this pill, ";"-joined
    name: str
    generic_name: str
    labeled_by: str
    imprint: str
    color: str
    shape: str
    n_consumer_available: int  # every consumer JPG of this pill
    n_consumer_eligible: int  # after the camera filter
    n_consumer_selected: int
    n_reference_selected: int
    consumer_mb: float


def _most_common(values: Iterable[str]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        if value:
            counts[value] += 1
    return max(sorted(counts), key=lambda v: counts[v]) if counts else ""


def select_classes(
    records: list[ImageRecord],
    *,
    max_classes: int,
    min_images: int,
    max_images_per_class: int,
    budget_bytes: int,
    exclude_cameras: Iterable[str] = (),
    seed: int = 0,
) -> tuple[list[PillClass], list[ImageRecord]]:
    """Pick the pills with the most consumer photos, within a download budget.

    Photos taken with a camera in `exclude_cameras` are not eligible. Classes are ranked by
    their number of eligible consumer JPGs (ties: class key). Each class gets at most
    `max_images_per_class` photos (a seeded random sample, so busy classes do not
    dominate) plus one small reference JPG per NDC11 when one exists. Classes are added in
    rank order while the total size stays within `budget_bytes`.
    """
    by_class: dict[str, list[ImageRecord]] = defaultdict(list)
    seen: set[str] = set()
    for record in records:
        # A few hundred files are listed twice in the XML (same pill); count each file once.
        if record.path in seen:
            continue
        seen.add(record.path)
        by_class[record.class_key].append(record)

    excluded = set(exclude_cameras)
    all_photos = {k: [r for r in v if is_consumer_photo(r)] for k, v in by_class.items()}
    consumer = {k: [r for r in v if r.camera not in excluded] for k, v in all_photos.items()}
    ranked = sorted(
        (k for k, v in consumer.items() if len(v) >= min_images),
        key=lambda k: (-len(consumer[k]), k),
    )
    rng = random.Random(seed)
    classes: list[PillClass] = []
    chosen: list[ImageRecord] = []
    used = 0
    for key in ranked:
        if len(classes) >= max_classes:
            break
        photos = sorted(consumer[key], key=lambda r: r.path)
        if len(photos) > max_images_per_class:
            photos = sorted(rng.sample(photos, max_images_per_class), key=lambda r: r.path)
        references = []
        for ndc11 in sorted({r.ndc11 for r in by_class[key]}):
            for layout in REFERENCE_JPG:
                match = [r for r in by_class[key] if r.ndc11 == ndc11 and r.image_type == layout]
                if match:
                    references.extend(sorted(match, key=lambda r: r.path))
                    break
        size = sum(r.file_size for r in photos + references)
        if used + size > budget_bytes:
            continue
        used += size
        everything = by_class[key]
        classes.append(
            PillClass(
                class_id=len(classes),
                class_key=key,
                ndc9=photos[0].ndc9,
                part=photos[0].part,
                ndc11_list=";".join(sorted({r.ndc11 for r in everything})),
                name=_most_common(r.proprietary_name for r in everything),
                generic_name=_most_common(r.generic_name for r in everything),
                labeled_by=_most_common(r.labeled_by for r in everything),
                imprint=_most_common(r.imprint for r in everything),
                color=_most_common(r.color for r in everything),
                shape=_most_common(r.shape for r in everything),
                n_consumer_available=len(all_photos[key]),
                n_consumer_eligible=len(consumer[key]),
                n_consumer_selected=len(photos),
                n_reference_selected=len(references),
                consumer_mb=round(sum(r.file_size for r in photos) / 1e6, 1),
            )
        )
        chosen.extend(photos + references)
    return classes, chosen


def write_classes(classes: list[PillClass], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(PillClass)])
        writer.writeheader()
        for pill in classes:
            writer.writerow(asdict(pill))


def read_classes(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def split_images(
    records: list[ImageRecord], *, val: float, test: float, seed: int = 0
) -> dict[str, str]:
    """Assign every selected image to train, val or test, per class.

    The split is by file (each photo is in exactly one split), stratified by class, and
    files with the same SHA-1 always land together so a duplicated photo cannot leak from
    train into test. Reference images are always in train: they are clean studio shots, not
    photos the app will be asked about. Returns {path: split}.
    """
    rng = random.Random(seed)
    groups: dict[str, dict[str, list[ImageRecord]]] = defaultdict(lambda: defaultdict(list))
    out: dict[str, str] = {}
    for record in records:
        if not is_consumer_photo(record):
            out[record.path] = "train"
            continue
        groups[record.class_key][record.sha1 or record.path].append(record)
    for key in sorted(groups):
        bundles = [groups[key][h] for h in sorted(groups[key])]
        rng.shuffle(bundles)
        n = len(bundles)
        n_test = max(1, round(n * test))
        n_val = max(1, round(n * val))
        for i, bundle in enumerate(bundles):
            split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            for record in bundle:
                out[record.path] = split
    return out


N_DISCS = 110
METADATA_FILES = ("AAREADME", "README", "directory_of_images.txt", "RxImageImageLayouts.docx")


def download_metadata(dest: Path, *, timeout: float = 600) -> list[Path]:
    """Fetch the README files, the image directory and ALLXML/PillProjectDisc{1..110}.xml.

    Files already present and parseable are kept; returns the files written.
    """
    names = [*METADATA_FILES, *(f"ALLXML/PillProjectDisc{i}.xml" for i in range(1, N_DISCS + 1))]
    written = []
    for name in names:
        target = dest / name
        if target.exists() and target.stat().st_size > 0:
            if not name.endswith(".xml"):
                continue
            try:
                ET.parse(target)  # noqa: S314 - NLM file
                continue
            except ET.ParseError:
                pass  # interrupted earlier: fetch again
        target.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(BASE_URL + name, headers={"User-Agent": USER_AGENT})  # noqa: S310
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            target.write_bytes(response.read())
        written.append(target)
    return written


def sha1_of(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(record: ImageRecord, dest: Path) -> Path:
    return dest / record.disc / record.file_name


def _fetch_one(record: ImageRecord, dest: Path, timeout: float) -> str:
    """Download one file; returns "downloaded", "skipped" or "failed: <reason>"."""
    target = local_path(record, dest)
    if target.exists() and (not record.sha1 or sha1_of(target) == record.sha1):
        return "skipped"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    url = BASE_URL + urllib.request.quote(record.path)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            tmp.write_bytes(response.read())
        if record.sha1 and sha1_of(tmp) != record.sha1:
            raise ValueError("SHA-1 mismatch")
        tmp.replace(target)
        return "downloaded"
    except Exception as error:  # reported; a rerun retries missing files
        tmp.unlink(missing_ok=True)
        return f"failed: {error}"


def download(
    records: list[ImageRecord], dest: Path, *, workers: int = 4, timeout: float = 300
) -> tuple[int, int, int]:
    """Download images to dest/<disc>/<file>, skipping files already there with the right SHA-1.

    A few parallel connections (default 4) to one public S3 bucket. Returns
    (downloaded, skipped, failed); failed files are listed on stdout and retried on rerun.
    """
    counts: Counter[str] = Counter()
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, r, dest, timeout): r for r in records}
        for i, future in enumerate(as_completed(futures), 1):
            outcome = future.result()
            if outcome.startswith("failed"):
                counts["failed"] += 1
                print(f"{futures[future].path} {outcome}", flush=True)
            else:
                counts[outcome] += 1
            if i % 100 == 0 or i == len(records):
                minutes = (time.monotonic() - start) / 60
                print(f"{i}/{len(records)} {dict(counts)} {minutes:.1f} min", flush=True)
    return counts["downloaded"], counts["skipped"], counts["failed"]
