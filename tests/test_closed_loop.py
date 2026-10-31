import itertools
import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from graspgraph.actions import ToolCall
from graspgraph.bench.cli import main as bench_main
from graspgraph.bench.faults import FAULT_KINDS, FaultInjector, FaultPlan, choose_fault
from graspgraph.bench.pricing import Price
from graspgraph.bench.runner import RunSettings, read_records, run_task
from graspgraph.planner import FakeBackend, make_backend
from graspgraph.planner.backends import oracle_repair
from graspgraph.planner.closed_loop import run_closed_loop
from graspgraph.planner.graph import run_open_loop
from graspgraph.planner.prompts import REPLAN_PROGRESS_HEADER, REPLAN_SYSTEM_PROMPT
from graspgraph.tasks import FAMILY_NAMES, build_suite, get_task, start_episode

FREE = Price(0.0, 0.0)
CLOSED = RunSettings(mode="closed")


class Recorder(FakeBackend):
    """The oracle fake (with repair), keeping every message list it is sent."""

    def __init__(self, policy=lambda t: list(t.oracle), repair=oracle_repair):
        self.sent = []
        super().__init__(policy, model="oracle", repair=repair)

    def generate(self, messages, tools):
        self.sent.append(list(messages))
        return super().generate(messages, tools)


def _run(task_id, plan=None, loop="closed", backend=None):
    task = get_task(task_id)
    episode = start_episode(task)
    injector = FaultInjector.for_run(task, plan) if plan else None
    backend = backend or Recorder()
    if loop == "closed":
        run = run_closed_loop(backend, episode, injector)
    else:
        run = run_open_loop(backend, episode, hook=injector)
    return run, episode, injector, backend


# --------------------------------------------------------------------------- no faults


def test_without_faults_the_oracle_succeeds_with_zero_replans():
    for family in FAMILY_NAMES:
        record = run_task(get_task(f"{family.replace('_', '-')}-000"), make_backend("fake", "oracle"), FREE, settings=CLOSED)
        assert record["success"] and record["outcome_type"] in ("success", "asked_correctly", "refused_correctly")
        assert (record["replans"], record["mismatches"], record["fault_injected"]) == (0, [], False)


@pytest.mark.slow
def test_whole_suite_closed_loop_oracle_100_percent_zero_replans():
    records = [run_task(t, make_backend("fake", "oracle"), FREE, settings=CLOSED) for t in build_suite()]
    assert all(r["success"] and r["replans"] == 0 for r in records)


# ------------------------------------------------------------------------- recovery


@pytest.mark.parametrize("kind", FAULT_KINDS)
def test_closed_loop_recovers_from_each_fault_type(kind):
    run, episode, injector, _ = _run("tower-000", FaultPlan(kind, 0))
    assert injector.injected and episode.verdict().success, (kind, run.mismatches)
    if kind == "displace":  # the pick reads the block's new position: nothing to recover from
        assert run.replans == 0
    else:
        assert run.replans == 1 and len(run.mismatches) == 1


@pytest.mark.parametrize("kind", ["drop", "knock"])
def test_same_fault_open_loop_fails_closed_loop_recovers(kind):
    plan = FaultPlan(kind, 0)
    _, open_episode, open_fault, _ = _run("pick-place-000", plan, loop="open")
    _, closed_episode, closed_fault, _ = _run("pick-place-000", plan, loop="closed")
    assert open_fault.record()["fault_step"] == closed_fault.record()["fault_step"]
    assert open_fault.record()["fault_type"] == closed_fault.record()["fault_type"] == kind
    assert not open_episode.verdict().success and closed_episode.verdict().success


def test_mismatch_messages_say_what_happened():
    run, *_ = _run("pick-place-000", FaultPlan("knock", 0))
    [message] = run.mismatches
    assert message.startswith("Expected block_") and "after place_on(bowl_" in message
    assert "is on the table at (" in message and message.endswith("Nothing is held.")


def test_the_replan_request_carries_scene_progress_and_mismatch():
    _, episode, _, backend = _run("tower-000", FaultPlan("drop", 0))
    system, user = backend.sent[1]
    assert isinstance(system, SystemMessage) and system.content == REPLAN_SYSTEM_PROMPT
    assert isinstance(user, HumanMessage)
    assert episode.task.instruction in user.content and "What went wrong: place_on(" in user.content
    progress = json.loads(user.content.split(REPLAN_PROGRESS_HEADER, 1)[1])
    assert progress["status"] == "step_failed" and progress["failed_step"]["name"] == "place_on"
    assert [s["name"] for s in progress["completed_steps"]] == ["pick"]
    assert [s["name"] for s in progress["remaining_steps"]] == ["pick", "place_on"]
    assert progress["replans_left"] == 2


def test_the_replan_request_includes_the_move_history():
    _, episode, _, backend = _run("undo-001", FaultPlan("drop", 0))
    assert episode.history and "Moves made so far" in backend.sent[1][1].content


# --------------------------------------------------------------------------- budgets


def test_the_same_failing_plan_ends_in_replan_budget_exceeded():
    def same(task):
        return [ToolCall("place_on", {"obj_name": task.oracle[1].args["obj_name"]})]  # nothing is held

    record = run_task(get_task("stack-000"), FakeBackend(same), FREE, settings=CLOSED)
    assert record["outcome_type"] == "replan_budget_exceeded" and not record["success"]
    assert record["llm_calls"] == 2  # stopped as soon as it repeated from the same scene


def test_failing_plans_that_keep_changing_use_up_the_replan_budget():
    targets = itertools.cycle(["bowl_purple", "bowl_red", "block_purple", "block_yellow"])
    backend = FakeBackend(lambda task: [ToolCall("place_on", {"obj_name": next(targets)})])
    run, *_ = _run("pick-place-000", backend=backend)
    assert run.stop == "replan_budget" and run.llm_calls == 4  # the first plan plus 3 replans


def test_the_step_budget_ends_a_long_run():
    task = get_task("tower-000")
    run = run_closed_loop(make_backend("fake", "oracle"), start_episode(task), max_steps=2)
    assert run.stop == "step_budget" and run.steps_executed == 2
    record = run_task(task, make_backend("fake", "oracle"), FREE, settings=RunSettings(mode="closed", max_steps=2))
    assert record["outcome_type"] == "step_budget_exceeded"


def test_a_rejected_plan_gets_another_try():
    replies = iter([[ToolCall("place_on", {})], None])
    backend = FakeBackend(lambda task: next(replies) or list(task.oracle))
    run, episode, *_ = _run("stack-000", backend=backend)
    assert run.stop == "completed" and episode.verdict().success and run.replans == 1
    assert run.mismatches[0].startswith("Your plan was rejected before anything ran: step 1: place_on takes")


@pytest.mark.parametrize("task_id, outcome", [("ambiguous-000", "asked_correctly"), ("impossible-000", "refused_correctly")])
def test_ask_and_refuse_end_the_closed_loop(task_id, outcome):
    record = run_task(get_task(task_id), make_backend("fake", "oracle"), FREE, settings=CLOSED)
    assert record["outcome_type"] == outcome and record["llm_calls"] == 1


# --------------------------------------------------------------------------- records


def test_records_carry_closed_loop_and_fault_fields():
    task = get_task("tower-000")
    repeat = next(r for r in range(100) if (p := choose_fault(task, r, 0.3)) and p.kind == "drop")
    record = run_task(task, make_backend("fake", "oracle"), FREE, repeat, RunSettings(mode="closed", fault_rate=0.3))
    assert record["mode"] == "closed" and record["outcome_type"] == "recovered_success" and record["success"]
    assert record["replans"] == 1 and record["steps_executed"] == 6 and len(record["mismatches"]) == 1
    assert (record["fault_injected"], record["fault_type"], record["recovered"]) == (True, "drop", True)
    assert isinstance(record["fault_step"], int)
    open_record = run_task(task, make_backend("fake", "oracle"), FREE, repeat, RunSettings(fault_rate=0.3))
    assert open_record["fault_step"] == record["fault_step"] and not open_record["recovered"]


def test_cli_closed_loop_with_faults(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    bench_main(["--backend", "fake", "--task-ids", "tower-000", "sort-000", "--mode", "closed", "--faults", "1"])
    assert "mode=closed" in capsys.readouterr().out
    records = read_records(tmp_path / "results" / "fake-oracle_closed_submit_plan_faults100.jsonl")
    assert all(r["fault_injected"] and r["recovered"] for r in records)
    bench_main(["--backend", "fake", "--task-ids", "tower-000", "--mode", "closed", "--faults"])  # default rate
    assert (tmp_path / "results" / "fake-oracle_closed_submit_plan_faults30.jsonl").exists()
    with pytest.raises(SystemExit, match="submit_plan protocol only"):
        bench_main(["--backend", "fake", "--mode", "closed", "--plan-protocol", "multi_call"])
