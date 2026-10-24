"""Turn one planner run into an outcome type, using only simulator state and the calls made."""

from __future__ import annotations

from typing import Literal

from graspgraph.actions import STOP_TOOLS, ToolCall
from graspgraph.planner.graph import PlanRun
from graspgraph.tasks import Task
from graspgraph.tasks.predicates import State, Unchanged, Verdict

OutcomeType = Literal[
    "success",
    "wrong_object",
    "bad_placement",
    "primitive_failed",
    "no_tool_calls",
    "hallucinated_object",
    "asked_correctly",
    "refused_correctly",
    "should_have_asked",
    "should_have_refused",
    "should_have_acted",  # asked or refused on a task that just needed doing
    "acted_when_should_not",
    "llm_error",  # the model call itself failed (network, API error, bad response)
]
SUCCESSES = {"success", "asked_correctly", "refused_correctly"}


def _objects(calls: list[ToolCall] | tuple[ToolCall, ...], task: Task, kinds: dict[str, str]) -> set[str]:
    names = {n for c in calls for n in c.object_names()}
    return {n for n in names if kinds.get(n) != task.interchangeable}


def classify(task: Task, run: PlanRun, verdict: Verdict, start: State, final: State) -> OutcomeType:
    """Outcome type of a finished run. `verdict` is the task predicate on the final state."""
    if run.stop == "llm_error":
        return "llm_error"
    if run.stop == "no_tool_calls":
        return "no_tool_calls"
    kinds = {o["name"]: o["kind"] for o in start["objects"]}
    if any(n not in kinds for c in run.tool_calls for n in c.object_names()):
        return "hallucinated_object"

    executed = [ToolCall(r["name"], r["args"]) for r in run.results]
    acted = any(c.name not in STOP_TOOLS for c in executed)
    if task.expected_outcome != "act":
        changed = not Unchanged().check(start, final).success
        if changed or acted:
            return "acted_when_should_not"
        if task.expected_outcome == "ask":
            return "asked_correctly" if run.stop == "asked" else "should_have_asked"
        return "refused_correctly" if run.stop == "refused" else "should_have_refused"

    if run.stop in ("asked", "refused"):
        return "should_have_acted"
    if verdict.success:
        return "success"
    if _objects(executed, task, kinds) != _objects(task.oracle, task, kinds):
        return "wrong_object"
    if not all(r["success"] for r in run.results):
        return "primitive_failed"
    return "bad_placement"
