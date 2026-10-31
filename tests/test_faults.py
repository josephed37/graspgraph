from collections import Counter

import numpy as np
import pytest

from graspgraph.bench.faults import FAULT_KINDS, FaultInjector, FaultPlan, choose_fault
from graspgraph.planner.backends import make_backend
from graspgraph.planner.graph import run_open_loop
from graspgraph.sim.robot import MAX_REACH
from graspgraph.sim.scene import BLOCK_HALF, BOWL_HALF, BOWL_WALL_T, TABLE_HEIGHT, TABLE_X, TABLE_Y
from graspgraph.tasks import build_suite, get_task, start_episode


@pytest.fixture(scope="module")
def suite():
    return build_suite()


def test_faults_are_deterministic_per_task_repeat_and_seed(suite):
    first = [choose_fault(t, 0, 0.3, seed=0) for t in suite]
    assert first == [choose_fault(t, 0, 0.3, seed=0) for t in suite]
    assert first != [choose_fault(t, 0, 0.3, seed=1) for t in suite]  # the seed matters
    assert first != [choose_fault(t, 1, 0.3, seed=0) for t in suite]  # so does the repeat


def test_roughly_the_requested_share_of_eligible_tasks_gets_a_fault(suite):
    eligible = [t for t in suite if t.expected_outcome == "act"]
    for rate in (0.3, 0.6):
        hits = [choose_fault(t, r, rate) for t in eligible for r in range(3)]
        share = sum(h is not None for h in hits) / len(hits)
        assert abs(share - rate) < 0.08, (rate, share)
    kinds = Counter(p.kind for t in eligible for r in range(10) if (p := choose_fault(t, r, 1.0)))
    assert set(kinds) == set(FAULT_KINDS)  # every type gets used


def test_ask_and_refuse_tasks_are_never_faulted(suite):
    assert all(choose_fault(t, r, 1.0) is None for t in suite if t.expected_outcome != "act" for r in range(5))
    assert choose_fault(suite[0], 0, 0.0) is None


def _assert_world_valid(robot):
    state = robot.get_scene()
    p = {o["name"]: np.array(o["position"]) for o in state["objects"]}
    blocks = [o["name"] for o in state["objects"] if o["kind"] == "block"]
    bowls = [o["name"] for o in state["objects"] if o["kind"] == "bowl"]
    for b in blocks:
        x, y, z = p[b]
        assert TABLE_X[0] < x < TABLE_X[1] and TABLE_Y[0] < y < TABLE_Y[1], f"{b} is off the table"
        assert np.hypot(x, y) < MAX_REACH and z >= TABLE_HEIGHT + BLOCK_HALF - 0.005, f"{b} is out of reach"
        for other in blocks:  # no two blocks interpenetrate (stacked ones sit a block apart in z)
            if other != b and abs(p[other][2] - z) < 2 * BLOCK_HALF - 0.005:
                assert np.linalg.norm(p[other][:2] - p[b][:2]) >= 2 * BLOCK_HALF - 0.005, f"{b} overlaps {other}"
        for bowl in bowls:  # a block is either inside a bowl or clear of its walls
            reach = np.max(np.abs(p[b][:2] - p[bowl][:2]))
            assert reach <= BOWL_HALF - BOWL_WALL_T / 2 - BLOCK_HALF + 0.005 or reach >= BOWL_HALF + BOWL_WALL_T / 2 + BLOCK_HALF - 0.005


@pytest.mark.parametrize("kind", FAULT_KINDS)
def test_each_fault_leaves_a_valid_world(kind):
    for task_id in ("pick-place-000", "tower-000"):
        task = get_task(task_id)
        episode = start_episode(task)
        injector = FaultInjector.for_run(task, FaultPlan(kind, 0))
        run_open_loop(make_backend("fake", "oracle"), episode, hook=injector)
        assert injector.injected and injector.step is not None, injector.detail
        episode.robot._step_for(0.5)
        _assert_world_valid(episode.robot)


def test_fault_fields_in_the_record():
    task = get_task("tower-000")
    injector = FaultInjector.for_run(task, FaultPlan("knock", 1))  # the second placement
    run_open_loop(make_backend("fake", "oracle"), start_episode(task), hook=injector)
    record = injector.record()
    assert (record["fault_injected"], record["fault_type"], record["fault_step"]) == (True, "knock", 4)
    assert "knocked off" in record["fault_detail"]
