"""Load a pickled pandas DataFrame without letting the pickle run arbitrary code.

TDC distributes its drug-synergy tables as pickles. A pickle can import and call anything,
so this loader only resolves the handful of pandas / numpy constructors a plain DataFrame
needs and refuses every other global.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.compat import pickle_compat

ALLOWED_GLOBALS = frozenset(
    {
        ("builtins", "slice"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.numeric", "_frombuffer"),
        ("pandas", "DataFrame"),
        ("pandas", "Index"),
        ("pandas", "RangeIndex"),
        ("pandas.core.frame", "DataFrame"),
        ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.range", "RangeIndex"),
        ("pandas.core.internals.managers", "BlockManager"),
        ("pandas.core.internals.blocks", "new_block"),
        ("pandas._libs.internals", "_unpickle_block"),
    }
)

# pandas' own compatibility unpickler knows how to rebuild frames written by old versions.
_Base: Any = getattr(pickle_compat, "Unpickler", pickle.Unpickler)


class _RestrictedUnpickler(_Base):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(
                f"refusing to load {module}.{name}: not a plain DataFrame component"
            )
        return super().find_class(module, name)


def load_dataframe_pickle(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        obj = _RestrictedUnpickler(handle).load()
    if not isinstance(obj, pd.DataFrame):
        raise pickle.UnpicklingError(f"{path.name} does not contain a DataFrame")
    return obj
