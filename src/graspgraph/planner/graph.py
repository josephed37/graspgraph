"""The open-loop planner graph: build_prompt -> plan -> validate -> execute -> end.

One model call returns the whole plan (see `plans.py` for the two protocols); validate
checks every step before anything moves; execute runs the steps in order with no checking
or replanning in between (the closed loop in `closed_loop.py` adds that). Every primitive
result is still recorded. Execution stops at ask_clarification or refuse.

The model call, plan validation and step execution are shared with the closed loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from graspgraph.actions import ASK, STOP_TOOLS, ToolCall, execute
from graspgraph.planner.backends import PlannerBackend, is_model_failure
from graspgraph.planner.parsing import parse_tool_calls
from graspgraph.planner.plans import Plan, PlanErrorKind, PlanProtocol, extract_plan
from graspgraph.planner.prompts import SYSTEM_PROMPTS, build_user_message
from graspgraph.planner.tools import tool_schemas
from graspgraph.sim.robot import PrimitiveResult, Robot
from graspgraph.tasks import Episode

StopReason = Literal[
    "completed",
    "asked",
    "refused",
    "no_tool_calls",
    "invalid_plan",
    "degenerate_output",
    "llm_error",
    "replan_budget",  # closed loop: ran out of replans
    "step_budget",  # closed loop: ran out of executed steps
    "no_progress",  # closed loop: the same step failed twice in a row from the same scene
]


class StepHook(Protocol):
    """Called around every executed step; the benchmark uses it to inject faults."""

    def before_step(self, robot: Robot, step: ToolCall) -> None: ...

    def after_step(self, robot: Robot, step: ToolCall, result: PrimitiveResult) -> None: ...


class PlannerState(TypedDict, total=False):
    messages: list[BaseMessage]
    response: AIMessage
    responses: list[AIMessage]  # every model reply, in order
    latency_s: float  # summed over model calls
    tool_calls: list[ToolCall]
    plan: Plan
    results: list[dict[str, Any]]
    stop: StopReason
    error: str
    model_failed: bool  # the error was the model's own (e.g. a repetition loop), not the connection


@dataclass
class PlanRun:
    """Everything one planner run produced, for scoring and logging."""

    protocol: PlanProtocol
    tool_calls: list[ToolCall]  # the raw calls of the last model reply
    plan: Plan | None  # the last plan; None when the model call failed or made no calls
    results: list[dict[str, Any]]  # one per executed step, across every plan: tool, args, success, message, steps
    stop: StopReason
    response_text: str
    input_tokens: int  # summed over model calls
    output_tokens: int
    latency_s: float  # wall time of the model calls only
    finish_reason: str | None = None  # why the model last stopped: e.g. "stop", or "length" at the token cap
    error: str | None = None  # set when the model call itself failed
    llm_calls: int = 1
    replans: int = 0  # model calls after the first (closed loop only)
    plans: list[Any] = field(default_factory=list)  # every plan as submitted, raw
    mismatches: list[str] = field(default_factory=list)  # closed loop: every verify mismatch, in order
    steps_executed: int = 0  # motion steps that actually ran

    @property
    def plan_error_kind(self) -> PlanErrorKind | None:
        return self.plan.error_kind if self.plan else None


def call_model(backend: PlannerBackend, messages: list[BaseMessage], tools: list[dict[str, Any]]) -> PlannerState:
    """One model call. Errors are captured in the returned state, never raised."""
    start = time.perf_counter()
    try:
        response = backend.generate(messages, tools)
    except Exception as exc:  # network, API or parsing errors inside the model client
        failed = {"response": AIMessage(""), "tool_calls": [], "error": f"{type(exc).__name__}: {exc}"}
        return {**failed, "model_failed": is_model_failure(exc), "latency_s": time.perf_counter() - start}
    return {"response": response, "latency_s": time.perf_counter() - start, "tool_calls": parse_tool_calls(response)}


def check_response(state: PlannerState, protocol: PlanProtocol, object_names: list[str]) -> PlannerState:
    """Turn the last model reply into a validated plan, or say why the run has to stop."""
    if state.get("error"):
        return {"stop": "degenerate_output" if state.get("model_failed") else "llm_error"}
    if not state["tool_calls"]:
        return {"stop": "no_tool_calls"}
    checked = extract_plan(state["tool_calls"], protocol, object_names)
    return {"plan": checked, "stop": "invalid_plan"} if checked.error else {"plan": checked}


def run_step(robot: Robot, step: ToolCall, hook: StepHook | None = None) -> dict[str, Any]:
    """Execute one validated step (stop tools only record their text) and return its result record."""
    if step.name in STOP_TOOLS:
        return {**step.to_dict(), "success": True, "message": str(next(iter(step.args.values()), "")), "steps": 0}
    if hook:
        hook.before_step(robot, step)
    result = execute(robot, step)
    if hook:
        hook.after_step(robot, step, result)
    return {**step.to_dict(), "success": result.success, "message": result.message, "steps": result.steps}


def stop_for(step: ToolCall) -> StopReason:
    return "asked" if step.name == ASK else "refused"


def build_open_loop_graph(
    backend: PlannerBackend, episode: Episode, protocol: PlanProtocol = "submit_plan", hook: StepHook | None = None
):
    """Compile the graph for one episode (the robot is bound in, since it is not serialisable)."""
    robot = episode.robot
    object_names = [o.name for o in episode.scene.objects]

    def build_prompt(state: PlannerState) -> PlannerState:
        user = build_user_message(episode.task.instruction, robot.get_scene(), episode.history)
        return {"messages": [SystemMessage(SYSTEM_PROMPTS[protocol]), HumanMessage(user)], "results": []}

    def plan(state: PlannerState) -> PlannerState:
        reply = call_model(backend, state["messages"], tool_schemas(object_names, protocol))
        return {**reply, "responses": [reply["response"]]}

    def validate(state: PlannerState) -> PlannerState:
        return check_response(state, protocol, object_names)

    def execute_plan(state: PlannerState) -> PlannerState:
        results: list[dict[str, Any]] = []
        for step in state["plan"].steps:
            results.append(run_step(robot, step, hook))
            if step.name in STOP_TOOLS:
                return {"results": results, "stop": stop_for(step)}
        return {"results": results, "stop": "completed"}

    graph = StateGraph(PlannerState)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("plan", plan)
    graph.add_node("validate", validate)
    graph.add_node("execute", execute_plan)
    graph.add_edge(START, "build_prompt")
    graph.add_edge("build_prompt", "plan")
    graph.add_edge("plan", "validate")
    # A rejected or empty plan ends the run without moving anything.
    graph.add_conditional_edges("validate", lambda s: END if "stop" in s else "execute", ["execute", END])
    graph.add_edge("execute", END)
    return graph.compile()


def plan_run(final: PlannerState, protocol: PlanProtocol) -> PlanRun:
    """Collect a finished graph state into a PlanRun."""
    responses: list[AIMessage] = final.get("responses") or [final["response"]]
    usage = [r.usage_metadata or {} for r in responses]
    plans = final.get("plans") or ([final["plan"].raw] if final.get("plan") else [])
    return PlanRun(
        protocol=protocol,
        tool_calls=final["tool_calls"],
        plan=final.get("plan"),
        results=final.get("results", []),
        stop=final["stop"],
        response_text=final["response"].text,
        input_tokens=sum(int(u.get("input_tokens", 0)) for u in usage),
        output_tokens=sum(int(u.get("output_tokens", 0)) for u in usage),
        latency_s=final["latency_s"],
        finish_reason=finish_reason(final["response"]),
        error=final.get("error"),
        llm_calls=len(responses),
        replans=len(responses) - 1,
        plans=plans,
    )


def run_open_loop(
    backend: PlannerBackend, episode: Episode, protocol: PlanProtocol = "submit_plan", hook: StepHook | None = None
) -> PlanRun:
    """Plan and execute one episode. Model errors, bad plans and primitive failures are recorded, never raised."""
    backend.prepare(episode.task)
    run = plan_run(build_open_loop_graph(backend, episode, protocol, hook).invoke({}), protocol)
    run.steps_executed = sum(r["name"] not in STOP_TOOLS for r in run.results)
    return run


def finish_reason(response: AIMessage) -> str | None:
    """Why the model stopped generating, as each provider reports it."""
    meta = response.response_metadata or {}
    reason = meta.get("done_reason") or meta.get("stop_reason") or meta.get("finish_reason")  # Ollama, Anthropic, others
    return str(reason) if reason else None
