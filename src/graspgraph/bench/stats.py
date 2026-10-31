"""Statistics for benchmark results: confidence intervals and paired comparisons.

Standard library and numpy only.
- Wilson score interval for a success rate (well behaved at 0%, 100% and small n).
- Paired comparisons of two configurations on the same runs (same task, repeat and therefore
  the same fault): the difference in success rate with a paired bootstrap 95% CI, and the
  exact McNemar test on the discordant pairs.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

Z_95 = 1.959963984540054  # two-sided 95% normal quantile
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261101

Record = dict[str, Any]


def wilson(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for `successes` out of `n` (0, 0 when n is 0)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts (A right/B wrong, and back)."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2**n
    return min(1.0, 2 * tail)


@dataclass(frozen=True)
class PairedComparison:
    n: int  # pairs compared
    rate_a: float
    rate_b: float
    diff: float  # rate_a - rate_b
    ci: tuple[float, float]  # paired bootstrap 95% CI of diff
    a_only: int  # pairs where only A succeeded
    b_only: int
    p_value: float  # exact McNemar


def pair_key(record: Record) -> tuple:
    """What makes two runs comparable: the same task and repeat, hence the same seeded fault."""
    return (record["task_id"], record.get("repeat", 0))


def pairs(a: Iterable[Record], b: Iterable[Record]) -> list[tuple[bool, bool]]:
    """Success of A and B on every run key both have."""
    a_by_key = {pair_key(r): bool(r["success"]) for r in a}
    b_by_key = {pair_key(r): bool(r["success"]) for r in b}
    return [(a_by_key[k], b_by_key[k]) for k in sorted(a_by_key.keys() & b_by_key.keys())]


def paired_comparison(
    pairs_ab: list[tuple[bool, bool]], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> PairedComparison | None:
    """Compare A against B on paired outcomes; None if there are no pairs."""
    if not pairs_ab:
        return None
    arr = np.array(pairs_ab, dtype=float)
    diffs = arr[:, 0] - arr[:, 1]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(resamples, len(diffs)))
    boot = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    a_only = int(np.sum((arr[:, 0] == 1) & (arr[:, 1] == 0)))
    b_only = int(np.sum((arr[:, 0] == 0) & (arr[:, 1] == 1)))
    return PairedComparison(
        n=len(diffs),
        rate_a=float(arr[:, 0].mean()),
        rate_b=float(arr[:, 1].mean()),
        diff=float(diffs.mean()),
        ci=(float(lo), float(hi)),
        a_only=a_only,
        b_only=b_only,
        p_value=mcnemar_exact(a_only, b_only),
    )


def per_task_rates(records: Iterable[Record]) -> dict[str, float]:
    """Per-task success, averaged over repeats."""
    by_task: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        by_task[r["task_id"]].append(bool(r["success"]))
    return {task: sum(v) / len(v) for task, v in by_task.items()}
