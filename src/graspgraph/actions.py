"""Tool calls: the one action format shared by oracle plans, task setups and the planner.

A `ToolCall` names a planner tool and its arguments. `execute` runs a motion tool through
the robot's primitives and never raises: bad names or arguments come back as a failed
`PrimitiveResult`, exactly as a real primitive failure would.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from graspgraph.sim.robot import PrimitiveResult, Robot

# Motion tools: name -> the argument names the robot primitive takes.
MOTION_TOOLS: dict[str, tuple[str, ...]] = {
    "pick": ("obj_name",),
    "place_on": ("obj_name",),
    "place_at": ("x", "y"),
    "move_home": (),
}
# Non-motion tools: calling one ends the plan without touching the world.
ASK = "ask_clarification"
REFUSE = "refuse"
STOP_TOOLS: dict[str, tuple[str, ...]] = {ASK: ("question",), REFUSE: ("reason",)}
ALL_TOOLS = {**MOTION_TOOLS, **STOP_TOOLS}


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)

    def object_names(self) -> list[str]:
        """Object names this call refers to (empty for coordinate and non-motion tools)."""
        value = self.args.get("obj_name")
        return [value] if isinstance(value, str) else []

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": dict(self.args)}

    def __str__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.name}({args})"


def execute(robot: Robot, call: ToolCall) -> PrimitiveResult:
    """Run one motion tool call through the robot. Stop tools are not motion: callers handle them."""
    params = MOTION_TOOLS.get(call.name)
    if params is None:
        known = ", ".join(MOTION_TOOLS)
        return PrimitiveResult(False, f"unknown tool '{call.name}'; motion tools are: {known}", 0)
    missing = [p for p in params if p not in call.args]
    extra = [a for a in call.args if a not in params]
    if missing or extra:
        return PrimitiveResult(
            False, f"{call.name} takes ({', '.join(params)}); missing {missing or 'none'}, unexpected {extra or 'none'}", 0
        )
    return getattr(robot, call.name)(**{p: call.args[p] for p in params})
