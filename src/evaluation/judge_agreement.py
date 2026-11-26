"""Agreement of the LLM judge with human labels: Cohen's kappa, not only the % of agreement.

For two raters on the same items with binary labels (1 = faithful, 0 = not faithful):
- observed agreement p_o = share of items with the same label;
- chance agreement p_e = sum over the labels of P(rater A gives it) * P(rater B gives it);
- kappa = (p_o - p_e) / (1 - p_e): 1 = perfect, 0 = what chance gives, < 0 = worse.
A judge that says "faithful" almost always agrees with mostly-faithful human labels by chance:
a high p_o with a kappa near 0 shows it. Interpretation of Landis & Koch (1977).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

LANDIS_KOCH = (
    (0.0, "poor"),
    (0.20, "slight"),
    (0.40, "fair"),
    (0.60, "moderate"),
    (0.80, "substantial"),
    (1.0, "almost perfect"),
)


def interpret(kappa: float) -> str:
    """Landis & Koch: < 0 poor, 0-0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate,
    0.61-0.80 substantial, 0.81-1 almost perfect."""
    if kappa < 0:
        return "poor"
    for upper, name in LANDIS_KOCH[1:]:
        if kappa <= upper:
            return name
    return "almost perfect"


def confusion(
    a: Sequence[int], b: Sequence[int], labels: Sequence[int] = (1, 0)
) -> list[list[int]]:
    """Rows: labels of rater a; columns: labels of rater b (same order as `labels`)."""
    if len(a) != len(b):
        raise ValueError("both raters must label the same items")
    return [
        [sum(1 for x, y in zip(a, b, strict=True) if x == i and y == j) for j in labels]
        for i in labels
    ]


def cohen_kappa(
    a: Sequence[int], b: Sequence[int], labels: Sequence[int] = (1, 0)
) -> dict[str, Any]:
    if not a or len(a) != len(b):
        raise ValueError("both raters must label the same, non-empty items")
    unknown = {*a, *b} - set(labels)
    if unknown:
        raise ValueError(f"labels outside {list(labels)}: {sorted(unknown)}")
    n = len(a)
    matrix = confusion(a, b, labels)
    observed = sum(matrix[i][i] for i in range(len(labels))) / n
    expected = sum(
        (sum(matrix[i]) / n) * (sum(row[i] for row in matrix) / n) for i in range(len(labels))
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")
    return {
        "n": n,
        "labels": list(labels),
        "confusion": matrix,
        "observed_agreement": observed,
        "expected_agreement": expected,
        "kappa": kappa,
        "interpretation": interpret(kappa) if kappa == kappa else "undefined (one label only)",
    }
