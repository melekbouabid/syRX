"""Build one training dataset: `python -m src.data.prepare <dataset> [options]`.

    v1           C3PI v1: labels, detection and classification datasets (stage prepare, v1.py)
    detect-v2    detection dataset of C3PI v2 (stage prepare_detect_v2, detect_v2.py)
    classify-v2  classification dataset of C3PI v2 (stage prepare_classify_v2, classify_v2.py)

The options are those of each module (`python -m src.data.prepare v1 --help`).
"""

from __future__ import annotations

import importlib
import sys

DATASETS = {
    "v1": "src.data.prepare.v1",
    "detect-v2": "src.data.prepare.detect_v2",
    "classify-v2": "src.data.prepare.classify_v2",
}


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    if not args or args[0] not in DATASETS:
        sys.exit(f"usage: python -m src.data.prepare {{{'|'.join(DATASETS)}}} [options]")
    importlib.import_module(DATASETS[args[0]]).main(args[1:])


if __name__ == "__main__":
    main()
