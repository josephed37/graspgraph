"""Turn a model response into a validated plan, for either planning protocol.

- "submit_plan" (default): the model makes one call, submit_plan(steps=[...]), whose steps
  are {"tool": ..., "args": {...}}, or a single ask_clarification / refuse call. Many models
  emit only one tool call per turn, so the whole plan has to fit in one call.
- "multi_call": the original protocol: the plan is the sequence of tool calls itself.

Every step is validated before anything runs. A plan with any bad step is rejected whole,
so nothing moves: structural problems are "invalid_plan", unknown object names are
"hallucinated_object".
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Literal

from graspgraph.actions import MOTION_TOOLS, STOP_TOOLS, ToolCall
from graspgraph.planner.parsing import coerce_args

PlanProtocol = Literal["submit_plan", "multi_call"]
PLAN_PROTOCOLS: tuple[PlanProtocol, ...] = ("submit_plan", "multi_call")
SUBMIT_PLAN = "submit_plan"
PlanErrorKind = Literal["invalid_plan", "hallucinated_object"]


@dataclass
class Plan:
    steps: list[ToolCall]  # validated, in execution order; may end with ask_clarification / refuse
    raw: Any  # what the model submitted, for the log
    planned_steps: int  # number of steps the model planned
    error: str | None = None
    error_kind: PlanErrorKind | None = None


def validate_step(call: ToolCall, object_names: set[str]) -> tuple[ToolCall | None, str | None, PlanErrorKind | None]:
    """Check one motion step; return it normalised (numbers as floats), or why it is invalid."""
    params = MOTION_TOOLS.get(call.name)
    if params is None:
        return None, f"unknown tool '{call.name}' (allowed: {', '.join(MOTION_TOOLS)})", "invalid_plan"
    missing = [p for p in params if p not in call.args]
    extra = [a for a in call.args if a not in params]
    if missing or extra:
        return None, f"{call.name} takes ({', '.join(params)}); missing {missing}, unexpected {extra}", "invalid_plan"
    args: dict[str, Any] = {}
    for name in params:
        value = call.args[name]
        if name == "obj_name":
            if not isinstance(value, str):
                return None, f"{call.name}: obj_name must be a string, got {value!r}", "invalid_plan"
            if value not in object_names:
                return None, f"{call.name}: unknown object '{value}'", "hallucinated_object"
            args[name] = value
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = math.nan
            if not math.isfinite(number):
                return None, f"{call.name}: {name} must be a number, got {value!r}", "invalid_plan"
            args[name] = number
    return ToolCall(call.name, args), None, None


def _reject(plan: Plan, problems: list[tuple[str, PlanErrorKind]]) -> Plan:
    # An unknown object name is the more specific diagnosis, so it wins when both occur.
    plan.error_kind = "hallucinated_object" if any(k == "hallucinated_object" for _, k in problems) else "invalid_plan"
    plan.error = "; ".join(p for p, _ in problems)
    plan.steps = []
    return plan


def _validate(plan: Plan, calls: list[ToolCall], object_names: set[str]) -> Plan:
    problems: list[tuple[str, PlanErrorKind]] = []
    steps: list[ToolCall] = []
    for i, call in enumerate(calls, start=1):
        if call.name in STOP_TOOLS:
            steps.append(call)
            break
        step, problem, kind = validate_step(call, object_names)
        if problem:
            problems.append((f"step {i}: {problem}", kind))
        else:
            steps.append(step)
    if problems:
        return _reject(plan, problems)
    plan.steps = steps
    return plan


def _step_call(raw: Any) -> ToolCall | None:
    """Read one submitted step, accepting {"tool", "args"} plus common variants."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict):
        return None
    name = raw.get("tool", raw.get("name"))
    if not isinstance(name, str):
        return None
    for key in ("args", "arguments", "parameters"):
        if key in raw:
            return ToolCall(name, coerce_args(raw[key]))
    return ToolCall(name, {k: v for k, v in raw.items() if k not in ("tool", "name")})  # args flattened in


def extract_plan(calls: list[ToolCall], protocol: PlanProtocol, object_names: list[str]) -> Plan:
    """The plan in a parsed response. `calls` must be non-empty."""
    names = set(object_names)
    if protocol == "multi_call":
        plan = Plan(steps=[], raw=[c.to_dict() for c in calls], planned_steps=len(calls))
        return _validate(plan, calls, names)

    first = calls[0]  # exactly one call is expected; any extra calls are ignored (but logged raw)
    if first.name in STOP_TOOLS:
        return Plan(steps=[first], raw=first.to_dict(), planned_steps=0)
    if first.name != SUBMIT_PLAN:
        plan = Plan(steps=[], raw=first.to_dict(), planned_steps=0)
        return _reject(plan, [(f"expected {SUBMIT_PLAN}, ask_clarification or refuse; got '{first.name}'", "invalid_plan")])

    raw_steps = first.args.get("steps")
    if isinstance(raw_steps, str):
        try:
            raw_steps = json.loads(raw_steps)
        except json.JSONDecodeError:
            pass
    plan = Plan(steps=[], raw=raw_steps, planned_steps=len(raw_steps) if isinstance(raw_steps, list) else 0)
    if not isinstance(raw_steps, list) or not raw_steps:
        return _reject(plan, [("submit_plan needs a non-empty list of steps", "invalid_plan")])
    parsed = [_step_call(s) for s in raw_steps]
    unreadable = [
        (f"step {i}: not a {{tool, args}} object", "invalid_plan") for i, c in enumerate(parsed, start=1) if c is None
    ]
    if unreadable:
        return _reject(plan, unreadable)
    stops = [i for i, c in enumerate(parsed, start=1) if c.name in STOP_TOOLS]
    if stops:
        return _reject(plan, [(f"step {stops[0]}: ask_clarification and refuse are separate tools, not plan steps", "invalid_plan")])
    return _validate(plan, parsed, names)
