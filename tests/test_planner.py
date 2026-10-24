import json

import pytest
from langchain_core.messages import AIMessage

from graspgraph.actions import ALL_TOOLS, MOTION_TOOLS, ToolCall
from graspgraph.planner import FakeBackend, run_open_loop
from graspgraph.planner.backends import PlannerBackend
from graspgraph.planner.parsing import parse_tool_calls
from graspgraph.planner.plans import extract_plan
from graspgraph.planner.prompts import HISTORY_INTRO, PROMPT_VERSIONS, SYSTEM_PROMPTS, build_user_message
from graspgraph.planner.tools import tool_schemas
from graspgraph.tasks import get_task, start_episode
from graspgraph.tasks.regions import REGIONS

NAMES = ["bowl_blue", "block_red", "block_green"]


def _fake(calls):
    return FakeBackend(lambda task: list(calls))


def _submit(*steps):
    return ToolCall("submit_plan", {"steps": list(steps)})


# ------------------------------------------------------------------ schemas, prompts, parsing


def test_tool_schema_object_enums_match_the_scene():
    for task_id in ("pick-place-003", "count-001", "undo-002"):
        episode = start_episode(get_task(task_id))
        names = [o.name for o in episode.scene.objects]

        multi = {s["function"]["name"]: s["function"] for s in tool_schemas(names, "multi_call")}
        assert set(multi) == set(ALL_TOOLS)
        for tool in ("pick", "place_on"):
            assert multi[tool]["parameters"]["properties"]["obj_name"]["enum"] == names
        for name, fn in multi.items():
            assert fn["parameters"]["required"] == list(ALL_TOOLS[name])

        submit = {s["function"]["name"]: s["function"] for s in tool_schemas(names, "submit_plan")}
        assert set(submit) == {"submit_plan", "ask_clarification", "refuse"}
        step = submit["submit_plan"]["parameters"]["properties"]["steps"]["items"]
        assert "anyOf" not in step  # flat on purpose: nested schemas broke Gemma's tool calls
        assert step["properties"]["tool"]["enum"] == list(MOTION_TOOLS)
        args = step["properties"]["args"]["properties"]
        assert list(args) == ["obj_name", "x", "y"] and args["obj_name"]["enum"] == names


def test_parse_tool_calls_accepts_dicts_strings_and_wrappers():
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "pick", "args": {"obj_name": "block_red"}, "id": "a"},
            {"name": "place_on", "args": {"arguments": {"obj_name": "bowl_blue"}}, "id": "b"},
            {"name": "place_at", "args": {"parameters": '{"x": 0.5, "y": "0.1"}'}, "id": "c"},
        ],
        invalid_tool_calls=[{"name": "pick", "args": "{not json", "id": "d", "error": "bad json"}],
    )
    assert parse_tool_calls(message) == [
        ToolCall("pick", {"obj_name": "block_red"}),
        ToolCall("place_on", {"obj_name": "bowl_blue"}),
        ToolCall("place_at", {"x": 0.5, "y": "0.1"}),
        ToolCall("pick", {}),  # unparseable: kept so it is recorded
    ]


def test_prompts_carry_instruction_scene_history_and_regions():
    episode = start_episode(get_task("undo-001"))
    user = build_user_message(episode.task.instruction, episode.robot.get_scene(), episode.history)
    assert episode.task.instruction in user
    assert json.dumps(episode.robot.get_scene(), indent=1) in user
    assert HISTORY_INTRO in user and episode.history[-1]["block"] in user
    assert '"from"' in HISTORY_INTRO and "undo" in HISTORY_INTRO
    assert HISTORY_INTRO not in build_user_message("Stack them.", episode.robot.get_scene(), [])
    for protocol, prompt in SYSTEM_PROMPTS.items():
        assert all(region.label in prompt for region in REGIONS.values())
        assert PROMPT_VERSIONS[protocol]
    assert "submit_plan" in SYSTEM_PROMPTS["submit_plan"] and "submit_plan" not in SYSTEM_PROMPTS["multi_call"]


# ----------------------------------------------------------------------- plan validation


def test_submit_plan_steps_are_read_defensively():
    plan = extract_plan(
        [
            _submit(
                {"tool": "pick", "args": {"obj_name": "block_red"}},
                '{"tool": "place_on", "args": {"obj_name": "block_green"}}',  # step as a JSON string
                {"tool": "pick", "args": '{"obj_name": "block_red"}'},  # args as a JSON string
                {"name": "place_at", "x": "0.5", "y": -0.1},  # "name" alias, flattened args, numeric string
                {"tool": "move_home", "arguments": {}},
            )
        ],
        "submit_plan",
        NAMES,
    )
    assert plan.error is None and plan.planned_steps == 5
    assert plan.steps == [
        ToolCall("pick", {"obj_name": "block_red"}),
        ToolCall("place_on", {"obj_name": "block_green"}),
        ToolCall("pick", {"obj_name": "block_red"}),
        ToolCall("place_at", {"x": 0.5, "y": -0.1}),
        ToolCall("move_home", {}),
    ]


def test_inline_step_numbers_are_skipped():
    # Gemma's spatial-006 plan: correct steps, numbered inline.
    raw = [1, {"args": {"obj_name": "block_red"}, "tool": "pick"}, 2, {"args": {"x": 0.62, "y": -0.25}, "tool": "place_at"}, 3,
           {"args": {}, "tool": "move_home"}]
    plan = extract_plan([ToolCall("submit_plan", {"steps": raw})], "submit_plan", NAMES)
    assert plan.error is None and plan.planned_steps == 3 and plan.raw == raw
    assert [s.name for s in plan.steps] == ["pick", "place_at", "move_home"]
    assert extract_plan([_submit(1, 2)], "submit_plan", NAMES).error  # nothing but numbers is still empty


def test_steps_as_a_json_string_and_single_ask_are_accepted():
    steps = json.dumps([{"tool": "pick", "args": {"obj_name": "block_red"}}])
    assert extract_plan([_submit()], "submit_plan", NAMES).error  # empty plan
    assert extract_plan([ToolCall("submit_plan", {"steps": steps})], "submit_plan", NAMES).planned_steps == 1
    ask = extract_plan([ToolCall("ask_clarification", {"question": "Which?"}), _submit()], "submit_plan", NAMES)
    assert ask.error is None and [s.name for s in ask.steps] == ["ask_clarification"]


@pytest.mark.parametrize(
    "calls, kind, phrase",
    [
        ([_submit({"tool": "grab", "args": {"obj_name": "block_red"}})], "invalid_plan", "unknown tool 'grab'"),
        ([_submit({"tool": "pick", "args": {}})], "invalid_plan", "missing ['obj_name']"),
        ([_submit({"tool": "move_home", "args": {"speed": 2}})], "invalid_plan", "unexpected ['speed']"),
        ([_submit({"tool": "place_at", "args": {"x": "left", "y": 0}})], "invalid_plan", "x must be a number"),
        ([_submit({"tool": "pick", "args": {"obj_name": 3}})], "invalid_plan", "must be a string"),
        ([_submit("not a step")], "invalid_plan", "step 1: not a {tool, args} object"),
        ([ToolCall("submit_plan", {"steps": "pick the red block"})], "invalid_plan", "non-empty list"),
        ([_submit({"tool": "ask_clarification", "args": {"question": "?"}})], "invalid_plan", "separate tools"),
        ([ToolCall("pick", {"obj_name": "block_red"})], "invalid_plan", "expected submit_plan"),
        ([_submit({"tool": "pick", "args": {"obj_name": "red block"}})], "hallucinated_object", "unknown object 'red block'"),
        (  # a hallucinated name wins over a structural problem elsewhere in the same plan
            [_submit({"tool": "grab", "args": {}}, {"tool": "pick", "args": {"obj_name": "block_pink"}})],
            "hallucinated_object",
            "step 2",
        ),
    ],
)
def test_invalid_submitted_plans_are_rejected(calls, kind, phrase):
    plan = extract_plan(calls, "submit_plan", NAMES)
    assert plan.error_kind == kind and phrase in plan.error and plan.steps == []


def test_multi_call_plans_are_validated_and_stop_at_ask():
    calls = [ToolCall("pick", {"obj_name": "block_red"}), ToolCall("ask_clarification", {"question": "?"}), ToolCall("move_home")]
    plan = extract_plan(calls, "multi_call", NAMES)
    assert [s.name for s in plan.steps] == ["pick", "ask_clarification"] and plan.planned_steps == 3
    bad = extract_plan([ToolCall("pick", {"obj_name": "block_red"}), ToolCall("place_on", {})], "multi_call", NAMES)
    assert bad.error_kind == "invalid_plan" and "step 2" in bad.error


# --------------------------------------------------------------------------- execution


def test_submit_plan_runs_every_step_in_order():
    task = get_task("tower-000")
    run = run_open_loop(FakeBackend(lambda t: list(t.oracle)), start_episode(task), "submit_plan")
    assert [c.name for c in run.tool_calls] == ["submit_plan"]  # one call holds the whole plan
    assert run.plan.planned_steps == 4 and run.stop == "completed"
    assert [(r["name"], r["success"]) for r in run.results] == [(c.name, True) for c in task.oracle]


def test_an_invalid_step_rejects_the_whole_plan_and_nothing_moves():
    episode = start_episode(get_task("pick-place-000"))
    block = episode.task.oracle[0].args["obj_name"]
    plan = [ToolCall("pick", {"obj_name": block}), ToolCall("place_on", {"obj_name": "the bowl"})]
    run = run_open_loop(_fake(plan), episode, "submit_plan")
    assert run.stop == "invalid_plan" and run.results == []
    assert episode.verdict().success is False and episode.robot.get_scene() == episode.start_state


def test_execution_stops_at_ask_in_multi_call():
    episode = start_episode(get_task("ambiguous-000"))
    run = run_open_loop(_fake([ToolCall("ask_clarification", {"question": "Which one?"}), ToolCall("move_home")]), episode, "multi_call")
    assert run.stop == "asked"
    assert [r["name"] for r in run.results] == ["ask_clarification"]
    assert run.results[0]["message"] == "Which one?"


@pytest.mark.parametrize("protocol", ["submit_plan", "multi_call"])
def test_open_loop_keeps_going_after_a_failed_primitive(protocol):
    episode = start_episode(get_task("pick-place-000"))
    bowl = next(o.name for o in episode.scene.objects if o.kind == "bowl")
    run = run_open_loop(_fake([ToolCall("place_on", {"obj_name": bowl}), ToolCall("move_home")]), episode, protocol)
    assert run.stop == "completed"
    assert [r["success"] for r in run.results] == [False, True]
    assert "not holding anything" in run.results[0]["message"]


def test_no_tool_calls_is_recorded():
    run = run_open_loop(_fake([]), start_episode(get_task("stack-000")))
    assert run.stop == "no_tool_calls" and run.results == [] and run.plan is None


def test_model_errors_are_recorded_not_raised():
    class Broken(PlannerBackend):
        def generate(self, messages, tools):
            raise ConnectionError("ollama is not running")

    run = run_open_loop(Broken("ollama", "gemma4:12b"), start_episode(get_task("stack-000")))
    assert run.stop == "llm_error"
    assert "ConnectionError: ollama is not running" in run.error
