"""Build one training dataset: `python -m src.data.prepare <v1|detect-v2|classify-v2>`.

Each dataset has its own module and options (`python -m src.data.prepare v1 --help`).
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
