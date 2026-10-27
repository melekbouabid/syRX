"""Tests for tujya_vision.c3pi on a tiny hand-written ALLXML file (no network)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from tujya_vision import c3pi


def _image(
    name: str, ndc11: str, kind: str, *, part: int = 1, size: int = 1000, sha1: str = ""
) -> str:
    tag = "Class" if kind.startswith("C3PI") else "Layout"
    file_type = name.rsplit(".", 1)[-1]
    return f"""
    <Image>
      <NDC9>{ndc11[:9]}</NDC9><NDC11>{ndc11}</NDC11><Part>{part}</Part><Parts>1</Parts>
      <LabeledBy>LAB</LabeledBy><GenericName>DRUG {ndc11[:9]}</GenericName>
      <ProprietaryName>BRAND {ndc11[:9]} 10MG</ProprietaryName>
      <File><Name>{name}</Name><Type>{file_type}</Type><Size>{size}</Size>
        <Sha1>{sha1 or name.lower()}</Sha1></File>
      <{tag}>{kind}</{tag}>
      <Imprint>A;1</Imprint><Color>WHITE</Color><Color>BLUE</Color><Shape>ROUND</Shape>
      <Size>9</Size><Camera>Phone</Camera><Background>wood</Background>
    </Image>"""


def _write_disc(tmp_path: Path, images: list[str], disc: int = 1) -> Path:
    xml_dir = tmp_path / "ALLXML"
    xml_dir.mkdir(exist_ok=True)
    body = "".join(images)
    (xml_dir / f"PillProjectDisc{disc}.xml").write_text(
        f'<?xml version="1.0" encoding="ISO-8859-1"?>'
        f"<MedicosConsultants><ImageExport>{body}</ImageExport></MedicosConsultants>",
        encoding="latin-1",
    )
    return xml_dir


@pytest.fixture
def records(tmp_path: Path) -> list[c3pi.ImageRecord]:
    images = []
    # Pill A: 30 consumer photos over two package NDCs, one reference JPG, one raw file.
    for i in range(30):
        images.append(_image(f"A{i}.JPG", "11111111101" if i % 2 else "11111111102", "C3PI_Test"))
    images.append(_image("Aref.JPG", "11111111101", "MC_API_RXNAV_V1.3"))
    images.append(_image("Araw.CR2", "11111111101", "C3PI_Reference", size=20_000_000))
    # Pill B: 25 photos, a duplicated file (same SHA-1) and a video that must be ignored.
    for i in range(25):
        images.append(_image(f"B{i}.JPG", "22222222201", "C3PI_Test"))
    images.append(_image("Bdup.JPG", "22222222201", "C3PI_Test", sha1="b0.jpg"))
    images.append(_image("B.WMV", "22222222201", "C3PI_Test"))
    # Pill C: too few photos.
    for i in range(5):
        images.append(_image(f"C{i}.JPG", "33333333301", "C3PI_Test"))
    return c3pi.build_index(_write_disc(tmp_path, images))


def test_parse_fields(records: list[c3pi.ImageRecord]) -> None:
    first = records[0]
    assert first.disc == "PillProjectDisc1"
    assert first.path == "PillProjectDisc1/images/A0.JPG"
    assert first.ndc9 == "111111111" and first.class_key == "111111111_1"
    assert first.color == "WHITE;BLUE"
    assert first.size_mm == "9"  # the pill size, not the file size
    assert first.file_size == 1000
    assert first.image_type == "C3PI_Test"
    assert records[30].image_type == "MC_API_RXNAV_V1.3"


def test_index_round_trip(records: list[c3pi.ImageRecord], tmp_path: Path) -> None:
    out = tmp_path / "images.csv"
    c3pi.write_index(records, out)
    assert c3pi.read_index(out) == records


def test_select_ranks_and_filters(records: list[c3pi.ImageRecord]) -> None:
    classes, chosen = c3pi.select_classes(
        records, max_classes=10, min_images=20, max_images_per_class=40, budget_bytes=10**9
    )
    assert [c.class_key for c in classes] == ["111111111_1", "222222222_1"]
    a, b = classes
    assert a.n_consumer_available == 30 and a.n_reference_selected == 1
    assert a.ndc11_list == "11111111101;11111111102"
    assert b.n_consumer_available == 26  # 25 + the duplicate, never the video
    assert all(r.file_type == "JPG" for r in chosen)  # no CR2, no WMV
    assert not any(r.class_key.startswith("333") for r in chosen)


def test_select_caps_images_and_respects_budget(records: list[c3pi.ImageRecord]) -> None:
    classes, chosen = c3pi.select_classes(
        records, max_classes=10, min_images=20, max_images_per_class=10, budget_bytes=12_000
    )
    # Pill A needs 11 files (10 photos + 1 reference) = 11,000 bytes; B no longer fits.
    assert [c.n_consumer_selected for c in classes] == [10]
    assert sum(r.file_size for r in chosen) <= 12_000


def test_split_never_shares_a_photo(records: list[c3pi.ImageRecord]) -> None:
    _, chosen = c3pi.select_classes(
        records, max_classes=10, min_images=20, max_images_per_class=40, budget_bytes=10**9
    )
    split = c3pi.split_images(chosen, val=0.15, test=0.15, seed=1)
    assert set(split) == {r.path for r in chosen}
    by_sha: dict[str, set[str]] = {}
    for r in chosen:
        by_sha.setdefault(r.sha1, set()).add(split[r.path])
    assert all(len(s) == 1 for s in by_sha.values())  # duplicates stay together
    assert split["PillProjectDisc1/images/Aref.JPG"] == "train"
    counts = Counter(
        split[r.path]
        for r in chosen
        if r.class_key == "111111111_1" and r.image_type == "C3PI_Test"
    )
    assert counts == {"train": 22, "val": 4, "test": 4}  # round(4.5) == 4
    assert split == c3pi.split_images(chosen, val=0.15, test=0.15, seed=1)  # reproducible


def test_download_skips_good_files(records: list[c3pi.ImageRecord], tmp_path: Path) -> None:
    payload = b"jpeg bytes"
    record = records[0]
    import hashlib

    good = c3pi.ImageRecord(**{**record.__dict__, "sha1": hashlib.sha1(payload).hexdigest()})
    target = c3pi.local_path(good, tmp_path / "img")
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    assert c3pi.download([good], tmp_path / "img", workers=1) == (0, 1, 0)


def test_excluded_cameras_are_not_eligible(records: list[c3pi.ImageRecord]) -> None:
    classes, chosen = c3pi.select_classes(
        records,
        max_classes=10,
        min_images=20,
        max_images_per_class=40,
        budget_bytes=10**9,
        exclude_cameras=["Phone"],
    )
    assert classes == [] and chosen == []


def test_a_file_listed_twice_counts_once(records: list[c3pi.ImageRecord]) -> None:
    doubled = [*records, records[0]]
    classes, chosen = c3pi.select_classes(
        doubled, max_classes=10, min_images=20, max_images_per_class=40, budget_bytes=10**9
    )
    assert classes[0].n_consumer_available == 30
    assert len({r.path for r in chosen}) == len(chosen)
