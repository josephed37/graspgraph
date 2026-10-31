import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from graspgraph.actions import ToolCall
from graspgraph.bench.cli import main as bench_main
from graspgraph.bench.faults import FaultInjector, choose_fault
from graspgraph.bench.pricing import Price
from graspgraph.bench.runner import RunSettings, read_records, run_task
from graspgraph.planner import FakeBackend, make_backend
from graspgraph.planner.backends import oracle_repair
from graspgraph.planner.closed_loop import run_closed_loop
from graspgraph.planner.graph import run_open_loop
from graspgraph.tasks import FAMILY_NAMES, build_suite, get_task, start_episode

FREE = Price(0.0, 0.0)
CLOSED = RunSettings(mode="closed")


class Recorder(FakeBackend):
    """The oracle fake, keeping every message list it was sent and every report it repaired from."""

    def __init__(self, policy=lambda t: list(t.oracle)):
        self.reports, self.sent = [], []
        super().__init__(policy, model="oracle", repair=self._repair)

    def _repair(self, task, report):
        self.reports.append(report)
        return oracle_repair(task, report)

    def generate(self, messages, tools):
        self.sent.append(list(messages))
        return super().generate(messages, tools)


def _faulted(task_id, kind, loop):
    task = get_task(task_id)
    episode = start_episode(task)
    injector = FaultInjector.for_run(task, 0, kind)
    backend = Recorder()
    run = run_closed_loop(backend, episode, injector) if loop == "closed" else run_open_loop(backend, episode, hook=injector)
    return run, episode, injector, backend


# --------------------------------------------------------------------------- no faults


def test_closed_loop_with_the_oracle_needs_no_replans():
    for family in FAMILY_NAMES:
        record = run_task(get_task(f"{family.replace('_', '-')}-000"), make_backend("fake", "oracle"), FREE, settings=CLOSED)
        assert record["success"], (record["task_id"], record["outcome_type"], record["detail"])
        assert (record["llm_calls"], record["replans"], record["fault"]) == (1, 0, None)
        assert record["mode"] == "closed" and record["prompt_version"] == "closed-v1-submit-plan"


# ------------------------------------------------------------------------------ faults


def test_a_dropped_block_breaks_open_loop_and_closed_loop_recovers():
    run, episode, injector, _ = _faulted("tower-000", "drop", "open")
    assert injector.triggered and not episode.verdict().success
    assert any("dropped on the way" in r["message"] for r in run.results)

    run, episode, injector, backend = _faulted("tower-000", "drop", "closed")
    assert injector.triggered and episode.verdict().success and run.stop == "completed"
    assert run.replans == 1 and len(run.plans) == 2
    [report] = backend.reports
    assert report["status"] == "step_failed" and report["failed_step"]["name"] == "place_on"
    assert "dropped on the way" in report["reason"] and report["block"]


def test_a_displaced_block_is_caught_by_the_end_of_plan_check():
    run, episode, injector, _ = _faulted("sort-000", "displace", "open")
    assert injector.triggered and all(r["success"] for r in run.results) and not episode.verdict().success

    run, episode, injector, backend = _faulted("sort-000", "displace", "closed")
    assert episode.verdict().success and run.replans == 1
    [report] = backend.reports
    assert report["status"] == "effect_undone" and report["reason"].startswith("Expected")
    assert report["unfinished_steps"] == []


def test_the_replan_report_answers_every_tool_call_with_the_current_scene():
    _, episode, _, backend = _faulted("tower-000", "drop", "closed")
    report = backend.reports[0]
    assert set(report) == {"status", "failed_step", "block", "reason", "completed_steps", "unfinished_steps", "scene", "replans_left"}
    assert report["scene"]["held"] is None and report["replans_left"] == 2
    assert [s["name"] for s in report["completed_steps"]] == ["pick"]
    second_call = backend.sent[1]
    assert [type(m).__name__ for m in second_call[-2:]] == ["AIMessage", "ToolMessage"]
    assert json.loads(second_call[-1].content)["status"] == "step_failed"


def test_faults_are_seeded_and_hit_about_30_percent_of_act_tasks():
    suite = build_suite()
    chosen = {t.task_id: choose_fault(t, 0, 0.3) for t in suite}
    assert chosen == {t.task_id: choose_fault(t, 0, 0.3) for t in suite}  # reproducible
    act = [t for t in suite if t.expected_outcome == "act"]
    rate = sum(chosen[t.task_id] is not None for t in act) / len(act)
    assert 0.2 <= rate <= 0.4
    assert all(chosen[t.task_id] is None for t in suite if t.expected_outcome != "act")
    assert all(chosen[t.task_id] != "displace" for t in suite if t.family in ("pick_place", "stack"))  # one place only
    assert choose_fault(act[0], 0, 0.0) is None


def test_records_report_faults_and_recovery():
    task = get_task("tower-000")
    repeat = next(r for r in range(50) if choose_fault(task, r, 0.3) == "drop")
    settings = RunSettings(mode="closed", fault_rate=0.3)
    record = run_task(task, make_backend("fake", "oracle"), FREE, repeat=repeat, settings=settings)
    assert record["fault"]["kind"] == "drop" and record["fault"]["triggered"]
    assert record["recovered"] and record["replans"] == 1 and record["fault_rate"] == 0.3 and record["max_replans"] == 3
    open_record = run_task(task, make_backend("fake", "oracle"), FREE, repeat=repeat, settings=RunSettings(fault_rate=0.3))
    assert open_record["fault"]["kind"] == "drop" and not open_record["success"] and not open_record["recovered"]


# ----------------------------------------------------------------------------- budgets


def test_a_rejected_plan_gets_a_report_and_another_try():
    calls = iter([[ToolCall("place_on", {})], None])

    def policy(task):
        return next(calls) or list(task.oracle)

    backend = FakeBackend(policy)  # no repair: asks the policy again
    episode = start_episode(get_task("stack-000"))
    run = run_closed_loop(backend, episode)
    assert run.stop == "completed" and episode.verdict().success and run.replans == 1


def test_the_step_budget_stops_a_long_run():
    run = run_closed_loop(make_backend("fake", "oracle"), start_episode(get_task("tower-000")), max_steps=2)
    assert run.stop == "budget_exhausted" and len(run.results) == 2


def test_two_tool_calls_both_get_answers_before_the_next_turn():
    first = AIMessage(
        content="",
        tool_calls=[
            {"name": "submit_plan", "args": {"steps": [{"tool": "pick", "args": {}}]}, "id": "a"},
            {"name": "refuse", "args": {"reason": "extra"}, "id": "b"},
        ],
    )
    replies = iter([first, None])
    backend = Recorder(policy=lambda t: next(replies) or list(t.oracle))
    backend.repair = None
    episode = start_episode(get_task("stack-000"))
    assert run_closed_loop(backend, episode).stop == "completed"
    answers = [m for m in backend.sent[1] if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in answers] == ["a", "b"]
    assert json.loads(answers[0].content)["status"] == "plan_rejected" and answers[1].content.startswith("ignored")


def test_ask_still_ends_the_closed_loop():
    episode = start_episode(get_task("ambiguous-000"))
    run = run_closed_loop(make_backend("fake", "oracle"), episode)
    assert run.stop == "asked" and run.llm_calls == 1


# --------------------------------------------------------------------------------- CLI


def test_cli_closed_loop_with_faults(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    bench_main(["--backend", "fake", "--task-ids", "tower-000", "sort-000", "--mode", "closed", "--fault-rate", "1"])
    assert "mode=closed" in capsys.readouterr().out
    records = read_records(tmp_path / "results" / "fake-oracle_closed_submit_plan_faults100.jsonl")
    assert all(r["fault"]["triggered"] and r["recovered"] for r in records)
    with pytest.raises(SystemExit, match="submit_plan protocol only"):
        bench_main(["--backend", "fake", "--mode", "closed", "--plan-protocol", "multi_call"])
