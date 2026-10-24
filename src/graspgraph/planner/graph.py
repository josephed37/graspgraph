"""The open-loop planner graph: build_prompt -> plan -> execute -> end.

One model call returns the whole plan; execute runs every call in order with no checking
or replanning in between (that is week 3's closed loop). Every primitive result is still
recorded. Execution stops at ask_clarification or refuse.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from graspgraph.actions import ASK, STOP_TOOLS, ToolCall, execute
from graspgraph.planner.backends import PlannerBackend
from graspgraph.planner.parsing import parse_tool_calls
from graspgraph.planner.prompts import SYSTEM_PROMPT, build_user_message
from graspgraph.planner.tools import tool_schemas
from graspgraph.tasks import Episode

StopReason = Literal["completed", "asked", "refused", "no_tool_calls", "llm_error"]


class PlannerState(TypedDict, total=False):
    messages: list[BaseMessage]
    response: AIMessage
    latency_s: float
    tool_calls: list[ToolCall]
    results: list[dict[str, Any]]
    stop: StopReason
    error: str


@dataclass
class PlanRun:
    """Everything one planner run produced, for scoring and logging."""

    tool_calls: list[ToolCall]
    results: list[dict[str, Any]]  # one per executed call: tool, args, success, message, steps
    stop: StopReason
    response_text: str
    input_tokens: int
    output_tokens: int
    latency_s: float  # wall time of the model call only
    error: str | None = None  # set when the model call itself failed


def build_open_loop_graph(backend: PlannerBackend, episode: Episode):
    """Compile the graph for one episode (the robot is bound in, since it is not serialisable)."""
    robot = episode.robot
    object_names = [o.name for o in episode.scene.objects]

    def build_prompt(state: PlannerState) -> PlannerState:
        user = build_user_message(episode.task.instruction, robot.get_scene(), episode.history)
        return {"messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(user)]}

    def plan(state: PlannerState) -> PlannerState:
        start = time.perf_counter()
        try:
            response = backend.generate(state["messages"], tool_schemas(object_names))
        except Exception as exc:  # network, API or parsing errors inside the model client
            failed = {"response": AIMessage(""), "tool_calls": [], "error": f"{type(exc).__name__}: {exc}"}
            return {**failed, "latency_s": time.perf_counter() - start}
        return {"response": response, "latency_s": time.perf_counter() - start, "tool_calls": parse_tool_calls(response)}

    def execute_plan(state: PlannerState) -> PlannerState:
        if state.get("error"):
            return {"results": [], "stop": "llm_error"}
        calls = state["tool_calls"]
        if not calls:
            return {"results": [], "stop": "no_tool_calls"}
        results: list[dict[str, Any]] = []
        for call in calls:
            if call.name in STOP_TOOLS:
                text = str(next(iter(call.args.values()), ""))
                results.append({**call.to_dict(), "success": True, "message": text, "steps": 0})
                return {"results": results, "stop": "asked" if call.name == ASK else "refused"}
            result = execute(robot, call)
            results.append({**call.to_dict(), "success": result.success, "message": result.message, "steps": result.steps})
        return {"results": results, "stop": "completed"}

    graph = StateGraph(PlannerState)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute_plan)
    graph.add_edge(START, "build_prompt")
    graph.add_edge("build_prompt", "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", END)
    return graph.compile()


def run_open_loop(backend: PlannerBackend, episode: Episode) -> PlanRun:
    """Plan and execute one episode. Neither model errors nor primitive failures raise: both are recorded."""
    backend.prepare(episode.task)
    final = build_open_loop_graph(backend, episode).invoke({})
    response: AIMessage = final["response"]
    usage = response.usage_metadata or {}
    return PlanRun(
        tool_calls=final["tool_calls"],
        results=final["results"],
        stop=final["stop"],
        response_text=response.text,
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        latency_s=final["latency_s"],
        error=final.get("error"),
    )
