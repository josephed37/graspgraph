"""The closed-loop planner: plan, run one step at a time, verify, and replan on a mismatch.

    build_prompt -> plan -> validate -> execute_step -> verify -> execute_step ... -> end
                              ^                            |
                              +--------- replan <----------+

- **verify** checks each step's expected effect (`verify.py`) right after it runs; a failed
  primitive is a mismatch too. When the plan has finished it re-checks every placement still
  in force, since a block can be moved after its step succeeded.
- **replan** is a fresh model call with its own prompt: the instruction, the scene now, the
  move history, the mismatch, and the progress so far (steps completed, the failed step, the
  steps that did not run). The model answers with a new submit_plan, or ask / refuse. A
  rejected plan (invalid steps) is reported the same way and also gets another try.
- **Budgets** end the run cleanly: at most `max_replans` replans ("replan_budget") and
  `max_steps` executed motion steps ("step_budget").
- **No progress**: the same step failing twice in a row from the same scene (block positions
  to the centimetre, and what is held) stops the run ("no_progress"), since another replan
  would only repeat it.

The closed loop uses the submit_plan protocol only.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from graspgraph.actions import STOP_TOOLS, ToolCall
from graspgraph.planner.backends import PlannerBackend
from graspgraph.planner.graph import PlannerState, PlanRun, StepHook, call_model, check_response, plan_run, run_step, stop_for
from graspgraph.planner.prompts import (
    CLOSED_LOOP_SYSTEM_PROMPT,
    REPLAN_SYSTEM_PROMPT,
    build_replan_message,
    build_user_message,
)
from graspgraph.planner.tools import tool_schemas
from graspgraph.planner.verify import Expectation, expectation_for, failure_text
from graspgraph.tasks import Episode

DEFAULT_MAX_REPLANS = 3
DEFAULT_MAX_STEPS = 20  # executed motion steps across all plans (the longest oracle plan has 8)


class ClosedLoopState(PlannerState, total=False):
    queue: list[ToolCall]  # validated steps of the current plan still to run
    completed: list[dict[str, Any]]  # motion steps that ran and checked out, across all plans
    in_force: list[Expectation]  # placements made so far that should still hold
    last: tuple[ToolCall, dict[str, Any], Expectation]  # the step just run, its result, its check
    problem: dict[str, Any]  # what the next replan is told: mismatch plus progress
    mismatches: list[str]  # every mismatch message, in order
    last_failure: str  # signature of the last failure, for the no-progress rule
    plans: list[Any]  # every plan as submitted, raw
    motion_steps: int


def _signature(step: ToolCall | None, extra: str, scene: dict[str, Any]) -> str:
    """Identity of a failure: the step, plus the scene to the centimetre and what is held."""
    layout = sorted((o["name"], *(round(v, 2) for v in o["position"])) for o in scene["objects"])
    return json.dumps([step.to_dict() if step else extra, layout, scene["held"]])


def build_closed_loop_graph(
    backend: PlannerBackend,
    episode: Episode,
    hook: StepHook | None = None,
    max_replans: int = DEFAULT_MAX_REPLANS,
    max_steps: int = DEFAULT_MAX_STEPS,
):
    """Compile the closed-loop graph for one episode."""
    robot = episode.robot
    task = episode.task
    object_names = [o.name for o in episode.scene.objects]
    tools = tool_schemas(object_names, "submit_plan")

    def fail(state: ClosedLoopState, status: str, step: ToolCall | None, block: str | None, mismatch: str) -> ClosedLoopState:
        """Record a mismatch and set up the replan, or stop if it repeats without progress."""
        scene = robot.get_scene()
        signature = _signature(step, mismatch, scene)
        update: ClosedLoopState = {"mismatches": [*state["mismatches"], mismatch], "last_failure": signature}
        if signature == state.get("last_failure"):
            return {**update, "stop": "no_progress"}
        progress = {
            "status": status,
            "failed_step": step.to_dict() if step else None,
            "block": block,
            "completed_steps": state["completed"],
            "remaining_steps": [s.to_dict() for s in state.get("queue", [])] if status == "step_failed" else [],
            "replans_left": max_replans - (len(state["responses"]) - 1) - 1,
        }
        return {**update, "problem": {"mismatch": mismatch, "progress": progress}}

    def build_prompt(state: ClosedLoopState) -> ClosedLoopState:
        user = build_user_message(task.instruction, robot.get_scene(), episode.history)
        return {
            "messages": [SystemMessage(CLOSED_LOOP_SYSTEM_PROMPT), HumanMessage(user)],
            "results": [],
            "responses": [],
            "plans": [],
            "completed": [],
            "in_force": [],
            "mismatches": [],
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
        if checked.get("stop") == "invalid_plan":  # report the problems and let the model try again
            checked.pop("stop")
            mismatch = f"Your plan was rejected before anything ran: {checked['plan'].error}."
            return {**checked, **fail({**state, "queue": []}, "plan_rejected", None, None, mismatch)}
        if "stop" not in checked:
            checked.update(queue=list(checked["plan"].steps), problem={})
        return checked

    def execute_step(state: ClosedLoopState) -> ClosedLoopState:
        step, queue = state["queue"][0], state["queue"][1:]
        if step.name not in STOP_TOOLS and state["motion_steps"] >= max_steps:
            return {"stop": "step_budget"}
        held_before = robot.held
        result = run_step(robot, step, hook)
        update: ClosedLoopState = {"queue": queue, "results": [*state["results"], result]}
        if step.name in STOP_TOOLS:
            return {**update, "stop": stop_for(step)}
        return {**update, "motion_steps": state["motion_steps"] + 1, "last": (step, result, expectation_for(step, held_before))}

    def verify(state: ClosedLoopState) -> ClosedLoopState:
        step, result, expected = state["last"]
        scene = robot.get_scene()
        if not result["success"]:
            return fail(state, "step_failed", step, expected.block, failure_text(step, result["message"], scene))
        mismatch = expected.mismatch(scene, robot.at_home())
        if mismatch:
            return fail(state, "step_failed", step, expected.block, mismatch)
        in_force = [e for e in state["in_force"] if e.block != expected.block]  # moving a block again supersedes
        if step.name in ("place_on", "place_at"):
            in_force.append(expected)
        update: ClosedLoopState = {"in_force": in_force, "completed": [*state["completed"], step.to_dict()]}
        if state["queue"]:
            return update
        for effect in in_force:  # the plan finished: is everything it placed still in place?
            undone = effect.mismatch(scene, robot.at_home())
            if undone:
                return {**update, **fail({**state, **update}, "effect_undone", effect.step, effect.block, undone)}
        return {**update, "stop": "completed"}

    def replan(state: ClosedLoopState) -> ClosedLoopState:
        if len(state["responses"]) - 1 >= max_replans:
            return {"stop": "replan_budget"}
        problem = state["problem"]
        user = build_replan_message(
            task.instruction, robot.get_scene(), episode.history, problem["mismatch"], problem["progress"]
        )
        messages = [SystemMessage(REPLAN_SYSTEM_PROMPT), HumanMessage(user)]
        reply = call_model(backend, messages, tools)
        return {
            **reply,
            "messages": messages,
            "responses": [*state["responses"], reply["response"]],
            "latency_s": state["latency_s"] + reply["latency_s"],
        }

    def next_after(state: ClosedLoopState) -> str:
        if "stop" in state:
            return END
        return "replan" if state.get("problem") else "execute_step"

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
    graph.add_conditional_edges("validate", next_after, ["execute_step", "replan", END])
    graph.add_conditional_edges("execute_step", lambda s: END if "stop" in s else "verify", ["verify", END])
    graph.add_conditional_edges("verify", next_after, ["execute_step", "replan", END])
    graph.add_conditional_edges("replan", lambda s: END if "stop" in s else "validate", ["validate", END])
    return graph.compile()


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
    run = plan_run(final, "submit_plan")
    run.mismatches = final.get("mismatches", [])
    run.steps_executed = final.get("motion_steps", 0)
    return run
