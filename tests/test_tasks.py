from collections import Counter

import pytest

from graspgraph.tasks import FAMILY_NAMES, SEEDS_PER_FAMILY, build_suite, get_task, start_episode
from graspgraph.tasks.oracle import run_oracle
from graspgraph.tasks.predicates import positions


@pytest.fixture(scope="module")
def suite():
    return build_suite()


def test_suite_has_110_unique_tasks_11_per_family(suite):
    assert len(suite) == 110
    assert len({t.task_id for t in suite}) == 110
    assert Counter(t.family for t in suite) == {f: SEEDS_PER_FAMILY for f in FAMILY_NAMES}


def test_generation_is_deterministic(suite):
    # Same id -> identical scene config, instruction, predicate, oracle and setup.
    for task in suite[::7]:
        assert get_task(task.task_id) == task


def test_same_task_id_builds_the_same_scene(suite):
    task = next(t for t in suite if t.family == "spatial")
    a, b = start_episode(task), start_episode(task)
    assert a.start_state == b.start_state


def test_expected_outcomes(suite):
    for task in suite:
        expected = {"ambiguous": "ask", "impossible": "refuse"}.get(task.family, "act")
        assert task.expected_outcome == expected, task.task_id
        assert bool(task.setup) == (task.family == "undo"), task.task_id


def test_instructions_vary_within_families(suite):
    for family in FAMILY_NAMES:
        texts = {t.instruction for t in suite if t.family == family}
        assert len(texts) >= 3, family


def test_act_predicates_are_false_at_the_start(suite):
    """No task is solved before the planner starts (undo tasks are checked after their setup)."""
    for task in suite:
        if task.expected_outcome != "act":
            continue  # ask/refuse tasks score the ask/refuse call plus "world unchanged"
        episode = start_episode(task)
        assert not episode.verdict().success, task.task_id


def test_undo_setup_moves_a_block_and_records_history(suite):
    task = next(t for t in suite if t.family == "undo" and len(t.setup) == 4)
    episode = start_episode(task)
    assert [m["move"] for m in episode.history] == [1, 2]
    last = episode.history[-1]
    assert last["block"] == task.predicate.block
    moved = positions(episode.start_state)[last["block"]]
    assert abs(moved[0] - task.predicate.x) + abs(moved[1] - task.predicate.y) > 0.05


def test_unknown_task_id_rejected():
    for bad in ("stack-011", "nope-000", "stack"):
        with pytest.raises(ValueError):
            get_task(bad)


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_oracle_solves_first_seed_of_each_family(family):
    verdict = run_oracle(get_task(f"{family.replace('_', '-')}-000"))
    assert verdict.success, verdict.detail


@pytest.mark.slow
def test_oracle_solves_all_110_tasks(suite):
    failures = [f"{t.task_id}: {v.detail}" for t in suite if not (v := run_oracle(t)).success]
    print(f"\noracle plans: {110 - len(failures)}/110 solved")
    assert not failures, "\n".join(failures)
