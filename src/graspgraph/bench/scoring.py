"""Turn one planner run into an outcome type, using only simulator state and the calls made."""

from __future__ import annotations

from typing import Literal

from graspgraph.actions import STOP_TOOLS, ToolCall
from graspgraph.planner.graph import PlanRun
from graspgraph.tasks import Task
from graspgraph.tasks.predicates import State, Unchanged, Verdict

OutcomeType = Literal[
    "success",
    "wrong_object",  # acted on an object the instruction did not refer to
    "incomplete_plan",  # only the right objects, but the plan stopped before the task was done
    "bad_placement",
    "primitive_failed",
    "invalid_plan",  # a planned step was malformed (unknown tool, missing or bad arguments)
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
PLACING = ("place_on", "place_at")


def _objects(calls: list[ToolCall] | tuple[ToolCall, ...], task: Task, kinds: dict[str, str]) -> set[str]:
    """Objects the calls act on, leaving out the task's interchangeable kind (if any)."""
    names = {n for c in calls for n in c.object_names()}
    return {n for n in names if kinds.get(n) != task.interchangeable}


def _placements(calls: list[ToolCall] | tuple[ToolCall, ...]) -> int:
    return sum(c.name in PLACING for c in calls)


def classify(task: Task, run: PlanRun, verdict: Verdict, start: State, final: State) -> OutcomeType:
    """Outcome type of a finished run. `verdict` is the task predicate on the final state.

    For "act" tasks that failed, in order: acting on an object the instruction did not refer
    to is wrong_object; a primitive refusing or failing is primitive_failed; a plan that used
    only referred objects but left some out, placed fewer blocks than the task needs, or
    ended still holding a block is incomplete_plan; anything else is bad_placement.
    """
    if run.stop == "llm_error":
        return "llm_error"
    if run.stop == "no_tool_calls":
        return "no_tool_calls"
    if run.stop == "invalid_plan":
        return run.plan_error_kind or "invalid_plan"

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
    kinds = {o["name"]: o["kind"] for o in start["objects"]}
    touched, referred = _objects(executed, task, kinds), _objects(task.oracle, task, kinds)
    if touched - referred:
        return "wrong_object"
    if not all(r["success"] for r in run.results):
        return "primitive_failed"
    if referred - touched or _placements(executed) < _placements(task.oracle) or final["held"] is not None:
        return "incomplete_plan"
    return "bad_placement"
