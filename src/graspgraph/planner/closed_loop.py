"""The closed-loop planner: plan, run one step at a time, verify, and replan on trouble.

    build_prompt -> plan -> validate -> execute_step -> verify -> execute_step ... -> end
                     ^         |                          |
                     +------ replan <---------------------+

- **verify** checks each step's expected effect (`verify.py`) right after it runs, and when
  the plan has finished re-checks every placement still in force.
- **replan** reports what happened (the failed step, why, the steps done and not done, and
  the current scene) as the answer to the model's submit_plan call, and asks for a new plan.
  A rejected plan (invalid steps) also gets a report and another try.
- **Budgets** stop loops: at most `max_replans` new plans and `max_steps` executed motion
  steps. Running out ends the run as "budget_exhausted", which scores as a failure.

The closed loop uses the submit_plan protocol only.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from graspgraph.actions import STOP_TOOLS, ToolCall
from graspgraph.planner.backends import PlannerBackend
from graspgraph.planner.graph import PlannerState, PlanRun, StepHook, call_model, check_response, plan_run, run_step, stop_for
from graspgraph.planner.prompts import CLOSED_LOOP_SYSTEM_PROMPT, build_user_message
from graspgraph.planner.tools import tool_schemas
from graspgraph.planner.verify import Expectation, expectation_for
from graspgraph.tasks import Episode

DEFAULT_MAX_REPLANS = 3
DEFAULT_MAX_STEPS = 24  # executed motion steps across all plans (the longest oracle plan has 8)


class ClosedLoopState(PlannerState, total=False):
    queue: list[ToolCall]  # validated steps of the current plan still to run
    done_in_plan: list[dict[str, Any]]  # steps of the current plan that ran and checked out
    in_force: list[Expectation]  # placements made so far that should still hold
    last: tuple[ToolCall, dict[str, Any], Expectation]  # the step just run, its result, its check
    feedback: dict[str, Any]  # the report for the next replan
    plans: list[Any]  # every plan as submitted, raw
    motion_steps: int


def _reply_ids(response: AIMessage) -> list[str]:
    """Ids of every tool call in a reply: each one needs an answer before the next model turn."""
    calls = [*response.tool_calls, *response.invalid_tool_calls]
    return [c.get("id") or f"call_{i}" for i, c in enumerate(calls)]


def build_closed_loop_graph(
    backend: PlannerBackend,
    episode: Episode,
    hook: StepHook | None = None,
    max_replans: int = DEFAULT_MAX_REPLANS,
    max_steps: int = DEFAULT_MAX_STEPS,
):
    """Compile the closed-loop graph for one episode."""
    robot = episode.robot
    object_names = [o.name for o in episode.scene.objects]
    tools = tool_schemas(object_names, "submit_plan")

    def build_prompt(state: ClosedLoopState) -> ClosedLoopState:
        user = build_user_message(episode.task.instruction, robot.get_scene(), episode.history)
        return {
            "messages": [SystemMessage(CLOSED_LOOP_SYSTEM_PROMPT), HumanMessage(user)],
            "results": [],
            "responses": [],
            "plans": [],
            "in_force": [],
            "motion_steps": 0,
            "latency_s": 0.0,
        }

    def plan(state: ClosedLoopState) -> ClosedLoopState:
        reply = call_model(backend, state["messages"], tools)
        return {
            **reply,
            "responses": [*state["responses"], reply["response"]],
            "latency_s": state["latency_s"] + reply["latency_s"],
        }

    def validate(state: ClosedLoopState) -> ClosedLoopState:
        checked = check_response(state, "submit_plan", object_names)
        if "plan" in checked:
            checked["plans"] = [*state["plans"], checked["plan"].raw]
        if checked.get("stop") == "invalid_plan":  # closed loop: report the problems and let it try again
            checked.pop("stop")
            checked["feedback"] = {"status": "plan_rejected", "reason": checked["plan"].error}
            return checked
        if "stop" not in checked:
            checked.update(queue=list(checked["plan"].steps), done_in_plan=[], feedback={})
        return checked

    def execute_step(state: ClosedLoopState) -> ClosedLoopState:
        step, queue = state["queue"][0], state["queue"][1:]
        held_before = robot.held
        result = run_step(robot, step, hook)
        update: ClosedLoopState = {"queue": queue, "results": [*state["results"], result]}
        if step.name in STOP_TOOLS:
            return {**update, "stop": stop_for(step)}
        return {**update, "motion_steps": state["motion_steps"] + 1, "last": (step, result, expectation_for(step, held_before))}

    def verify(state: ClosedLoopState) -> ClosedLoopState:
        step, result, expected = state["last"]
        scene = robot.get_scene()
        if not result["success"] or not expected.holds(scene):
            reason = result["message"] if not result["success"] else f"check failed: expected {expected.description}"
            return {"feedback": _report("step_failed", step, expected.block, reason, state, scene)}
        in_force = [e for e in state["in_force"] if e.block != expected.block]  # moving a block again supersedes
        if step.name != "pick":
            in_force.append(expected)
        done = [*state["done_in_plan"], step.to_dict()]
        if state["queue"]:
            if state["motion_steps"] >= max_steps:
                return {"in_force": in_force, "done_in_plan": done, "stop": "budget_exhausted"}
            return {"in_force": in_force, "done_in_plan": done}
        broken = [e for e in in_force if not e.holds(scene)]
        if broken:  # the plan finished, but something it did has since been undone
            first = broken[0]
            reason = f"{first.description} no longer holds"
            report = _report("effect_undone", first.step, first.block, reason, {**state, "done_in_plan": done}, scene)
            return {"in_force": in_force, "done_in_plan": done, "feedback": report}
        return {"in_force": in_force, "done_in_plan": done, "stop": "completed"}

    def replan(state: ClosedLoopState) -> ClosedLoopState:
        if len(state["responses"]) > max_replans:
            return {"stop": "budget_exhausted"}
        feedback = {**state["feedback"], "replans_left": max_replans - len(state["responses"])}
        if "scene" not in feedback:
            feedback["scene"] = robot.get_scene()
        content = json.dumps(feedback)
        answers = [
            ToolMessage(content if i == 0 else "ignored: only the first tool call is used", tool_call_id=call_id)
            for i, call_id in enumerate(_reply_ids(state["response"]))
        ]
        messages = [*state["messages"], state["response"], *answers]
        reply = call_model(backend, messages, tools)
        return {
            **reply,
            "messages": messages,
            "responses": [*state["responses"], reply["response"]],
            "latency_s": state["latency_s"] + reply["latency_s"],
        }

    def after_validate(state: ClosedLoopState) -> str:
        if "stop" in state:
            return END
        return "replan" if state.get("feedback") else "execute_step"

    def after_execute(state: ClosedLoopState) -> str:
        return END if "stop" in state else "verify"

    def after_verify(state: ClosedLoopState) -> str:
        if "stop" in state:
            return END
        return "replan" if state.get("feedback") else "execute_step"

    graph = StateGraph(ClosedLoopState)
    for name, node in [
        ("build_prompt", build_prompt),
        ("plan", plan),
        ("validate", validate),
        ("execute_step", execute_step),
        ("verify", verify),
        ("replan", replan),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "build_prompt")
    graph.add_edge("build_prompt", "plan")
    graph.add_edge("plan", "validate")
    graph.add_conditional_edges("validate", after_validate, ["execute_step", "replan", END])
    graph.add_conditional_edges("execute_step", after_execute, ["verify", END])
    graph.add_conditional_edges("verify", after_verify, ["execute_step", "replan", END])
    graph.add_conditional_edges("replan", lambda s: END if "stop" in s else "validate", ["validate", END])
    return graph.compile()


def _report(
    status: str, step: ToolCall, block: str | None, reason: str, state: ClosedLoopState, scene: dict[str, Any]
) -> dict[str, Any]:
    """What the model is told when it has to replan."""
    return {
        "status": status,
        "failed_step": step.to_dict(),
        "block": block,
        "reason": reason,
        "completed_steps": state["done_in_plan"],
        "unfinished_steps": [s.to_dict() for s in state["queue"]] if status == "step_failed" else [],
        "scene": scene,
    }


def run_closed_loop(
    backend: PlannerBackend,
    episode: Episode,
    hook: StepHook | None = None,
    max_replans: int = DEFAULT_MAX_REPLANS,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> PlanRun:
    """Plan, execute with verification and replan. Nothing a model or primitive does raises."""
    backend.prepare(episode.task)
    graph = build_closed_loop_graph(backend, episode, hook, max_replans, max_steps)
    # Each loop pass is a few graph steps; leave room for the step and replan budgets.
    final = graph.invoke({}, {"recursion_limit": 10 * (max_steps + max_replans) + 20})
    return plan_run(final, "submit_plan")
