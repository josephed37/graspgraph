import json

import pytest

from graspgraph.actions import ToolCall
from graspgraph.bench.cli import main as bench_main
from graspgraph.bench.pricing import Price, load_prices, price_for
from graspgraph.bench.runner import read_records, run_benchmark, run_task
from graspgraph.bench.summary import latest_records, summarize
from graspgraph.planner import FakeBackend, make_backend
from graspgraph.planner.backends import PlannerBackend
from graspgraph.tasks import FAMILY_NAMES, get_task, start_episode

FREE = Price(0.0, 0.0)


def _scripted(calls):
    return FakeBackend(lambda task: list(calls))


def _first(family):
    return get_task(f"{family.replace('_', '-')}-000")


def test_oracle_fake_scores_100_percent_one_task_per_family():
    backend = make_backend("fake", "oracle")
    records = [run_task(_first(f), backend, FREE) for f in FAMILY_NAMES]
    assert all(r["success"] for r in records), [(r["task_id"], r["outcome_type"], r["detail"]) for r in records]
    assert {r["outcome_type"] for r in records} == {"success", "asked_correctly", "refused_correctly"}
    json.dumps(records)  # every record is plain JSON


def test_wrong_object_fake_is_scored_wrong_object():
    backend = make_backend("fake", "wrong-object")
    for family in ("pick_place", "stack", "spatial"):
        record = run_task(_first(family), backend, FREE)
        assert record["success"] is False
        assert record["outcome_type"] == "wrong_object", (family, record["detail"])


def test_no_tool_calls_fake_is_scored():
    record = run_task(_first("stack"), make_backend("fake", "no-tools"), FREE)
    assert (record["success"], record["outcome_type"], record["tool_calls"]) == (False, "no_tool_calls", [])


@pytest.mark.parametrize(
    "task_id, calls, outcome",
    [
        ("ambiguous-000", [ToolCall("ask_clarification", {"question": "Which?"})], "asked_correctly"),
        ("ambiguous-000", [ToolCall("refuse", {"reason": "no"})], "should_have_asked"),
        ("ambiguous-000", [ToolCall("move_home")], "acted_when_should_not"),
        ("impossible-000", [ToolCall("refuse", {"reason": "no such block"})], "refused_correctly"),
        ("impossible-000", [ToolCall("ask_clarification", {"question": "Which?"})], "should_have_refused"),
        ("stack-000", [ToolCall("ask_clarification", {"question": "Sure?"})], "should_have_acted"),
        ("stack-000", [ToolCall("pick", {"obj_name": "red block"})], "hallucinated_object"),
    ],
)
def test_outcome_types_for_non_motion_plans(task_id, calls, outcome):
    record = run_task(get_task(task_id), _scripted(calls), FREE)
    assert record["outcome_type"] == outcome
    assert record["success"] is (outcome in {"asked_correctly", "refused_correctly"})


def test_acting_on_an_ambiguous_task_is_scored():
    task = get_task("ambiguous-000")
    objects = start_episode(task).start_state["objects"]
    block = next(o["name"] for o in objects if o["kind"] == "block")
    bowl = next(o["name"] for o in objects if o["kind"] == "bowl")
    calls = [ToolCall("pick", {"obj_name": block}), ToolCall("place_on", {"obj_name": bowl})]
    assert run_task(task, _scripted(calls), FREE)["outcome_type"] == "acted_when_should_not"


def test_wrong_object_primitive_failure_and_bad_placement_are_told_apart():
    spatial = _first("spatial")
    pick = spatial.oracle[0]
    untouched = run_task(spatial, _scripted([ToolCall("move_home")]), FREE)
    assert untouched["outcome_type"] == "wrong_object"  # never touched the block it was asked to move
    off_table = run_task(spatial, _scripted([pick, ToolCall("place_at", {"x": 0.30, "y": 1.0})]), FREE)
    assert off_table["outcome_type"] == "primitive_failed"

    tower = _first("tower")
    a, b, c = tower.predicate.bottom_to_top
    wrong_order = [
        ToolCall("pick", {"obj_name": a}),
        ToolCall("place_on", {"obj_name": b}),
        ToolCall("pick", {"obj_name": c}),
        ToolCall("place_on", {"obj_name": a}),
    ]
    record = run_task(tower, _scripted(wrong_order), FREE)
    assert all(r["success"] for r in record["primitive_results"])
    assert record["outcome_type"] == "bad_placement"


def test_cost_comes_from_usage_and_the_price_table():
    prices = load_prices()
    haiku = price_for(prices, "anthropic", "claude-haiku-5-5")
    assert haiku.cost(1_000_000, 1_000_000) == pytest.approx(0.60)
    assert price_for(prices, "anthropic", "claude-sonnet-5-5").cost(1000, 100) == pytest.approx(0.003)
    assert price_for(prices, "ollama", "gemma4:12b").cost(10**6, 10**6) == 0.0
    assert haiku.note(100_001) and haiku.note(99_000) is None
    with pytest.raises(ValueError):
        price_for(prices, "anthropic", "claude-unknown")

    backend = FakeBackend(lambda task: list(task.oracle), input_tokens=2000, output_tokens=100)
    record = run_task(get_task("ambiguous-000"), backend, haiku)
    assert (record["input_tokens"], record["output_tokens"]) == (2000, 100)
    assert record["cost_usd"] == pytest.approx(2000 * 0.10 / 1e6 + 100 * 0.50 / 1e6)


TASKS = [get_task(f"ambiguous-00{i}") for i in range(4)]  # cheap: nothing moves


def test_resume_skips_runs_already_recorded(tmp_path):
    out = tmp_path / "r.jsonl"
    backend = make_backend("fake", "oracle")
    first = run_benchmark(TASKS[:2], backend, out, FREE, log=lambda _: None)
    assert (first.ran, first.skipped) == (2, 0)
    second = run_benchmark(TASKS, backend, out, FREE, repeats=2, log=lambda _: None)
    assert (second.ran, second.skipped) == (6, 2)
    assert len(read_records(out)) == 8


def test_resume_retries_llm_errors_and_survives_a_cut_off_line(tmp_path):
    class Flaky(PlannerBackend):
        def generate(self, messages, tools):
            raise TimeoutError("model timed out")

    out = tmp_path / "r.jsonl"
    run_benchmark(TASKS[:1], Flaky("fake", "oracle"), out, FREE, log=lambda _: None)
    assert read_records(out)[0]["outcome_type"] == "llm_error"
    with out.open("a") as f:
        f.write('{"task_id": "cut-off')  # a crash mid-write
    report = run_benchmark(TASKS[:1], make_backend("fake", "oracle"), out, FREE, log=lambda _: None)
    assert report.ran == 1
    assert [r["outcome_type"] for r in latest_records(out)] == ["asked_correctly"]


def test_spend_cap_stops_the_run_cleanly(tmp_path):
    out = tmp_path / "r.jsonl"
    backend = FakeBackend(lambda task: list(task.oracle), model="oracle", input_tokens=1000, output_tokens=1000)
    price = Price(input=100.0, output=100.0)  # $0.20 per run
    messages = []
    report = run_benchmark(TASKS, backend, out, price, max_usd=0.50, log=messages.append)
    assert report.stopped_by_cap and report.ran == 2
    assert report.spent_usd == pytest.approx(0.40)
    assert "spend cap reached" in messages[-1]
    # Resuming with the same cap does not spend more.
    again = run_benchmark(TASKS, backend, out, price, max_usd=0.50, log=messages.append)
    assert again.ran == 0 and again.stopped_by_cap


def test_cli_runs_fake_backend_and_summary_prints_table(tmp_path, capsys):
    out = tmp_path / "fake.jsonl"
    bench_main(["--backend", "fake", "--model", "oracle", "--task-ids", "ambiguous-001", "impossible-001", "--out", str(out)])
    assert "2 run" in capsys.readouterr().out
    table = summarize(out)
    assert "overall success  2/2 (100.0%)" in table
    assert "ambiguous" in table and "refused_correctly" in table and "cost $0.0000" in table


def test_cli_rejects_unknown_choices(tmp_path):
    with pytest.raises(SystemExit, match="unknown fake policy"):
        bench_main(["--backend", "fake", "--model", "nonsense", "--out", str(tmp_path / "x.jsonl")])
    with pytest.raises(SystemExit, match="unknown families"):
        bench_main(["--backend", "fake", "--families", "juggling", "--out", str(tmp_path / "x.jsonl")])
