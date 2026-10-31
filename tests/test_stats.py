from collections import Counter
from pathlib import Path

import pytest

from graspgraph.bench.stats import mcnemar_exact, paired_comparison, pairs, per_task_rates, wilson
from graspgraph.tasks import FAMILY_NAMES, get_task

SUBSET = Path(__file__).resolve().parents[1] / "experiments" / "subsets" / "sonnet50.txt"


@pytest.mark.parametrize(
    "k, n, expected",
    [  # reference values computed by hand from the Wilson formula (z = 1.96)
        (0, 10, (0.0, 0.2775)),
        (10, 10, (0.7225, 1.0)),
        (5, 10, (0.2366, 0.7634)),
        (81, 100, (0.7222, 0.8749)),
    ],
)
def test_wilson_interval_matches_known_values(k, n, expected):
    lo, hi = wilson(k, n)
    assert lo == pytest.approx(expected[0], abs=1e-4) and hi == pytest.approx(expected[1], abs=1e-4)
    assert wilson(0, 0) == (0.0, 0.0)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(6, 0) == pytest.approx(2 / 64)  # 2 * 0.5**6
    assert mcnemar_exact(1, 9) == pytest.approx(2 * (1 + 10) / 1024)  # 0.021484375
    assert mcnemar_exact(0, 10) == mcnemar_exact(10, 0)


def test_paired_comparison_on_a_hand_made_example():
    # 10 pairs: both succeed 4, only A 4, only B 1, neither 1  ->  A 0.8, B 0.5, diff 0.3
    data = [(True, True)] * 4 + [(True, False)] * 4 + [(False, True)] + [(False, False)]
    result = paired_comparison(data)
    assert (result.n, result.rate_a, result.rate_b) == (10, 0.8, 0.5)
    assert result.diff == pytest.approx(0.3)
    assert (result.a_only, result.b_only) == (4, 1)
    assert result.p_value == pytest.approx(mcnemar_exact(4, 1)) == pytest.approx(0.375)
    lo, hi = result.ci
    assert lo < 0.3 < hi and -0.2 < lo and hi <= 0.7
    assert paired_comparison(data).ci == result.ci  # fixed seed: reproducible
    identical = paired_comparison([(True, True), (False, False)] * 5)
    assert identical.diff == 0 and identical.ci == (0.0, 0.0) and identical.p_value == 1.0
    assert paired_comparison([]) is None


def test_pairing_matches_task_and_repeat_and_per_task_means():
    a = [{"task_id": "t1", "repeat": 0, "success": True}, {"task_id": "t1", "repeat": 1, "success": False},
         {"task_id": "t2", "repeat": 0, "success": True}]
    b = [{"task_id": "t1", "repeat": 1, "success": True}, {"task_id": "t2", "repeat": 0, "success": False},
         {"task_id": "t3", "repeat": 0, "success": True}]
    assert pairs(a, b) == [(False, True), (True, False)]  # (t1, 1) and (t2, 0) only
    assert per_task_rates(a) == {"t1": 0.5, "t2": 1.0}


def test_sonnet_subset_is_50_valid_tasks_5_per_family():
    ids = SUBSET.read_text().split()
    assert len(ids) == len(set(ids)) == 50
    families = Counter(get_task(i).family for i in ids)  # get_task raises on an invalid id
    assert families == {f: 5 for f in FAMILY_NAMES}
