import json

from langchain_core.messages import AIMessage

from graspgraph.actions import ALL_TOOLS, ToolCall
from graspgraph.planner import FakeBackend, run_open_loop
from graspgraph.planner.backends import PlannerBackend
from graspgraph.planner.parsing import parse_tool_calls
from graspgraph.planner.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_message
from graspgraph.planner.tools import tool_schemas
from graspgraph.tasks import get_task, start_episode
from graspgraph.tasks.regions import REGIONS


def _fake(calls):
    return FakeBackend(lambda task: list(calls))


def test_tool_schema_object_enums_match_the_scene():
    for task_id in ("pick-place-003", "count-001", "undo-002"):
        episode = start_episode(get_task(task_id))
        names = [o.name for o in episode.scene.objects]
        schemas = {s["function"]["name"]: s["function"] for s in tool_schemas(names)}
        assert set(schemas) == set(ALL_TOOLS)
        for tool in ("pick", "place_on"):
            assert schemas[tool]["parameters"]["properties"]["obj_name"]["enum"] == names
        for name, fn in schemas.items():
            assert fn["parameters"]["required"] == list(ALL_TOOLS[name])


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
    calls = parse_tool_calls(message)
    assert calls == [
        ToolCall("pick", {"obj_name": "block_red"}),
        ToolCall("place_on", {"obj_name": "bowl_blue"}),
        ToolCall("place_at", {"x": 0.5, "y": "0.1"}),
        ToolCall("pick", {}),  # unparseable: kept so it is recorded as a failed call
    ]


def test_prompts_carry_instruction_scene_history_and_regions():
    episode = start_episode(get_task("undo-001"))
    user = build_user_message(episode.task.instruction, episode.robot.get_scene(), episode.history)
    assert episode.task.instruction in user
    assert json.dumps(episode.robot.get_scene(), indent=1) in user
    assert "Moves made so far" in user and episode.history[-1]["block"] in user
    plain = build_user_message("Stack them.", episode.robot.get_scene(), [])
    assert "Moves made so far" not in plain
    for region in REGIONS.values():
        assert region.label in SYSTEM_PROMPT
    assert PROMPT_VERSION


def test_execution_stops_at_ask_and_records_results():
    episode = start_episode(get_task("ambiguous-000"))
    run = run_open_loop(_fake([ToolCall("ask_clarification", {"question": "Which one?"}), ToolCall("move_home")]), episode)
    assert run.stop == "asked"
    assert [r["name"] for r in run.results] == ["ask_clarification"]
    assert run.results[0]["message"] == "Which one?"


def test_open_loop_keeps_going_after_a_failed_primitive():
    episode = start_episode(get_task("pick-place-000"))
    bowl = next(o.name for o in episode.scene.objects if o.kind == "bowl")
    run = run_open_loop(_fake([ToolCall("place_on", {"obj_name": bowl}), ToolCall("move_home")]), episode)
    assert run.stop == "completed"
    assert [r["success"] for r in run.results] == [False, True]
    assert "not holding anything" in run.results[0]["message"]


def test_no_tool_calls_is_recorded():
    run = run_open_loop(_fake([]), start_episode(get_task("stack-000")))
    assert run.stop == "no_tool_calls" and run.results == []


def test_model_errors_are_recorded_not_raised():
    class Broken(PlannerBackend):
        def generate(self, messages, tools):
            raise ConnectionError("ollama is not running")

    run = run_open_loop(Broken("ollama", "gemma4:12b"), start_episode(get_task("stack-000")))
    assert run.stop == "llm_error"
    assert "ConnectionError: ollama is not running" in run.error
