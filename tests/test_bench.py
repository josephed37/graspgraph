import json

import pytest
from langchain_core.messages import AIMessage

from graspgraph.actions import ToolCall
from graspgraph.bench.cli import main as bench_main
from graspgraph.bench.pricing import Price, load_prices, price_for
from graspgraph.bench.runner import RunSettings, read_records, run_benchmark, run_task
from graspgraph.bench.summary import latest_records, summarize
from graspgraph.planner import FakeBackend, make_backend
from graspgraph.planner.backends import FAKE_POLICIES, PlannerBackend
from graspgraph.tasks import FAMILY_NAMES, get_task, start_episode

FREE = Price(0.0, 0.0)


def _scripted(calls):
    return FakeBackend(lambda task: list(calls))  # the plan as steps; the fake formats it per protocol


def _first(family):
    return get_task(f"{family.replace('_', '-')}-000")


@pytest.mark.parametrize("protocol", ["submit_plan", "multi_call"])
def test_oracle_fake_scores_100_percent_one_task_per_family(protocol):
    backend = make_backend("fake", "oracle")
    records = [run_task(_first(f), backend, FREE, settings=RunSettings(protocol=protocol)) for f in FAMILY_NAMES]
    assert all(r["success"] for r in records), [(r["task_id"], r["outcome_type"], r["detail"]) for r in records]
    assert {r["outcome_type"] for r in records} == {"success", "asked_correctly", "refused_correctly"}
    json.dumps(records)  # every record is plain JSON


def test_records_hold_the_planned_step_count_and_raw_plan():
    task = _first("tower")
    record = run_task(task, make_backend("fake", "oracle"), FREE)
    assert record["plan_protocol"] == "submit_plan" and record["prompt_version"] == "open-v4-submit-plan"
    assert record["planned_steps"] == 4
    assert record["raw_plan"] == [{"tool": c.name, "args": c.args} for c in task.oracle]
    assert [c["name"] for c in record["tool_calls"]] == ["submit_plan"]
    assert record["plan_error"] is None

    legacy = run_task(task, make_backend("fake", "oracle"), FREE, settings=RunSettings(protocol="multi_call"))
    assert legacy["prompt_version"] == "open-v3-multi-call" and legacy["planned_steps"] == 4
    assert legacy["raw_plan"] == [c.to_dict() for c in task.oracle]

    bad = run_task(task, _scripted([ToolCall("pick", {})]), FREE)
    assert bad["planned_steps"] == 1 and "missing ['obj_name']" in bad["plan_error"]


def _objects_of(task, kind):
    return [o["name"] for o in start_episode(task).start_state["objects"] if o["kind"] == kind]


def _other_bowl(task):
    target = task.oracle[1].args["obj_name"]
    return next(b for b in _objects_of(task, "bowl") if b != target)


def _wrong_tower(task):
    a, b, c = task.predicate.bottom_to_top
    return [ToolCall("pick", {"obj_name": a}), ToolCall("place_on", {"obj_name": b}),
            ToolCall("pick", {"obj_name": c}), ToolCall("place_on", {"obj_name": a})]


def _broken(task):
    raise ConnectionError("model server unreachable")


REPEAT_LOOP = "ResponseError: prediction aborted, token repeat limit reached (status code: 500)"


def _looping(task):
    raise RuntimeError(REPEAT_LOOP)


def _ran_out_of_tokens(task):
    # What Gemma did with nested step schemas: output up to the cap, nothing parseable.
    return AIMessage(content="", response_metadata={"done_reason": "length"})


def _silent(task):
    return AIMessage(content="", response_metadata={"done_reason": "stop"})


# (expected outcome_type, task id, policy: task -> planned steps, protocol)
LABEL_CASES = [
    ("success", "pick-place-000", lambda t: list(t.oracle), "submit_plan"),
    ("wrong_object", "pick-place-000", lambda t: FAKE_POLICIES["wrong-object"](t), "submit_plan"),
    ("wrong_object", "pick-place-000", lambda t: [t.oracle[0], ToolCall("place_on", {"obj_name": _other_bowl(t)})], "submit_plan"),
    ("incomplete_plan", "tower-000", lambda t: FAKE_POLICIES["first-step"](t), "multi_call"),  # the Gemma bug
    ("incomplete_plan", "tower-000", lambda t: FAKE_POLICIES["first-step"](t), "submit_plan"),
    ("incomplete_plan", "tower-000", lambda t: list(t.oracle[:2]), "submit_plan"),  # one of two stacks
    ("incomplete_plan", "count-001", lambda t: list(t.oracle[:-2]), "submit_plan"),  # one block short
    ("incomplete_plan", "spatial-000", lambda t: [ToolCall("move_home")], "submit_plan"),
    ("bad_placement", "tower-000", _wrong_tower, "submit_plan"),
    ("primitive_failed", "spatial-000", lambda t: [t.oracle[0], ToolCall("place_at", {"x": 0.30, "y": 1.0})], "submit_plan"),
    ("invalid_plan", "stack-000", lambda t: [t.oracle[0], ToolCall("place_on", {})], "submit_plan"),
    ("invalid_plan", "stack-000", lambda t: [t.oracle[0], ToolCall("grab", {"obj_name": "block_red"})], "multi_call"),
    ("no_tool_calls", "stack-000", lambda t: [], "submit_plan"),
    ("no_tool_calls", "stack-000", _silent, "submit_plan"),
    ("hallucinated_object", "stack-000", lambda t: [ToolCall("pick", {"obj_name": "red block"})], "submit_plan"),
    ("asked_correctly", "ambiguous-000", lambda t: [ToolCall("ask_clarification", {"question": "Which?"})], "submit_plan"),
    ("should_have_asked", "ambiguous-000", lambda t: [ToolCall("refuse", {"reason": "no"})], "submit_plan"),
    ("acted_when_should_not", "ambiguous-000", lambda t: [ToolCall("move_home")], "submit_plan"),
    (
        "acted_when_should_not",
        "ambiguous-000",
        lambda t: [ToolCall("pick", {"obj_name": _objects_of(t, "block")[0]}),
                   ToolCall("place_on", {"obj_name": _objects_of(t, "bowl")[0]})],
        "submit_plan",
    ),
    ("refused_correctly", "impossible-000", lambda t: [ToolCall("refuse", {"reason": "no such block"})], "submit_plan"),
    ("should_have_refused", "impossible-000", lambda t: [ToolCall("ask_clarification", {"question": "?"})], "submit_plan"),
    ("should_have_acted", "stack-000", lambda t: [ToolCall("ask_clarification", {"question": "Sure?"})], "submit_plan"),
    ("llm_error", "stack-000", _broken, "submit_plan"),
    (  # closed loop: a plan that can never work, retried until the replans run out
        "replan_budget_exhausted",
        "stack-000",
        lambda t: [ToolCall("place_on", {"obj_name": t.oracle[1].args["obj_name"]})],
        RunSettings(mode="closed", max_replans=2),
    ),
    ("degenerate_output", "stack-000", _looping, "submit_plan"),
    ("degenerate_output", "stack-000", _ran_out_of_tokens, "submit_plan"),
]


@pytest.mark.parametrize("outcome, task_id, policy, protocol", LABEL_CASES)
def test_every_outcome_label_with_a_fake_llm(outcome, task_id, policy, protocol):
    settings = protocol if isinstance(protocol, RunSettings) else RunSettings(protocol=protocol)
    record = run_task(get_task(task_id), FakeBackend(policy), FREE, settings=settings)
    assert record["outcome_type"] == outcome, (record["detail"], record["plan_error"], record["primitive_results"])
    assert record["success"] is (outcome in {"success", "asked_correctly", "refused_correctly"})


def test_every_outcome_label_is_covered():
    from graspgraph.bench.scoring import OutcomeType

    assert {case[0] for case in LABEL_CASES} == set(OutcomeType.__args__)


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


def test_resume_is_per_protocol_and_reads_legacy_records_as_multi_call(tmp_path):
    out = tmp_path / "r.jsonl"
    backend = make_backend("fake", "oracle")
    run_benchmark(TASKS[:1], backend, out, FREE, settings=RunSettings(protocol="multi_call"), log=lambda _: None)
    record = read_records(out)[0]
    del record["plan_protocol"]  # what a file written before this field existed looks like
    out.write_text(json.dumps(record) + "\n")
    again = run_benchmark(TASKS[:1], backend, out, FREE, settings=RunSettings(protocol="multi_call"), log=lambda _: None)
    assert (again.ran, again.skipped) == (0, 1)
    other = run_benchmark(TASKS[:1], backend, out, FREE, settings=RunSettings(protocol="submit_plan"), log=lambda _: None)
    assert (other.ran, other.skipped) == (1, 0)


def test_cli_plan_protocol_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    bench_main(["--backend", "fake", "--model", "first-step", "--task-ids", "tower-000", "--plan-protocol", "multi_call"])
    assert "protocol=multi_call" in capsys.readouterr().out
    [record] = read_records(tmp_path / "results" / "fake-first-step_open_multi_call.jsonl")
    assert (record["plan_protocol"], record["outcome_type"], record["planned_steps"]) == ("multi_call", "incomplete_plan", 1)
    bench_main(["--backend", "fake", "--model", "oracle", "--task-ids", "tower-000"])  # default protocol
    [record] = read_records(tmp_path / "results" / "fake-oracle_open_submit_plan.jsonl")
    assert (record["plan_protocol"], record["success"]) == ("submit_plan", True)
    assert "mean planned steps 4.0" in summarize(tmp_path / "results" / "fake-oracle_open_submit_plan.jsonl")


def test_repetition_loops_are_model_failures_and_not_retried(tmp_path):
    out = tmp_path / "r.jsonl"
    looping = FakeBackend(_looping, model="oracle")
    run_benchmark(TASKS[:2], looping, out, FREE, log=lambda _: None)
    assert [r["outcome_type"] for r in read_records(out)] == ["degenerate_output"] * 2
    # Files written before degenerate_output existed logged loops as llm_error; those are not retried either.
    records = read_records(out)
    records[1]["outcome_type"] = "llm_error"
    out.write_text("".join(json.dumps(r) + "\n" for r in records))
    report = run_benchmark(TASKS[:2], make_backend("fake", "oracle"), out, FREE, log=lambda _: None)
    assert (report.ran, report.skipped) == (0, 2)


def test_a_new_prompt_version_is_not_skipped_on_resume(tmp_path):
    out = tmp_path / "r.jsonl"
    backend = make_backend("fake", "oracle")
    run_benchmark(TASKS[:1], backend, out, FREE, log=lambda _: None)
    [record] = read_records(out)
    record["prompt_version"] = "open-v2-submit-plan"  # a run made with the previous prompt
    out.write_text(json.dumps(record) + "\n")
    report = run_benchmark(TASKS[:1], backend, out, FREE, log=lambda _: None)
    assert (report.ran, report.skipped) == (1, 0)
    table = summarize(out)  # one table per configuration, never averaged together
    assert table.count("== ") == 2 and "prompt_version=open-v2-submit-plan" in table and "prompt_version=open-v4-submit-plan" in table


def test_ollama_output_is_capped():
    assert make_backend("ollama").chat_model.num_predict == 2048
    assert make_backend("ollama", think=True).chat_model.num_predict == 8192  # thinking counts against the cap
    assert make_backend("ollama", max_tokens=500).chat_model.num_predict == 500


def test_finish_reason_is_recorded():
    assert run_task(get_task("stack-000"), FakeBackend(_ran_out_of_tokens), FREE)["finish_reason"] == "length"
    assert run_task(get_task("stack-000"), make_backend("fake", "oracle"), FREE)["finish_reason"] is None


def test_temperature_is_recorded_and_keeps_runs_apart(tmp_path):
    out = tmp_path / "r.jsonl"
    default, cold = make_backend("fake", "oracle"), make_backend("fake", "oracle")
    cold.temperature = 0.0
    run_benchmark(TASKS[:1], default, out, FREE, log=lambda _: None)
    report = run_benchmark(TASKS[:1], cold, out, FREE, log=lambda _: None)
    assert (report.ran, report.skipped) == (1, 0)  # a different temperature is a different run
    assert sorted(str(r["temperature"]) for r in read_records(out)) == ["0.0", "None"]
    assert make_backend("ollama", temperature=0.2).temperature == 0.2
    assert summarize(out).count("== ") == 2
