"""What each step should have achieved, and the mismatch message when it did not.

The closed-loop planner checks every step right after it runs, against the scene from
get_scene() (and, for move_home, the arm's joints). At the end of a plan it re-checks every
placement still in force, because a block can be moved after its step succeeded. The checks
reuse the task predicates' tolerances, but they only describe the planner's own steps: the
task's success predicate stays hidden from the planner.

Mismatch messages are short and factual, so a model can act on them, for example:
"Expected block_red to be held after pick(block_red), but block_red is on the table at
(0.52, -0.10). Nothing is held."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graspgraph.actions import ToolCall
from graspgraph.tasks.predicates import ON_TABLE_Z, UNDO_XY_TOL, Z_TOL, State, in_bowl, on_table, on_top, positions


def step_text(step: ToolCall) -> str:
    """A compact rendering of a step for messages: pick(block_red), place_at(0.50, -0.10)."""
    if step.name == "place_at":
        return f"place_at({float(step.args['x']):.2f}, {float(step.args['y']):.2f})"
    return f"{step.name}({', '.join(str(v) for v in step.args.values())})"


def held_text(state: State) -> str:
    return f"{state['held']} is held." if state["held"] else "Nothing is held."


def where(state: State, block: str) -> str:
    """Where a block is, in words: held, in a bowl, on a block, on the table, or just a position."""
    if state["held"] == block:
        return f"{block} is in the gripper"
    p = positions(state)
    xy = f"({p[block][0]:.2f}, {p[block][1]:.2f})"
    for obj in state["objects"]:
        name = obj["name"]
        if name != block and obj["kind"] == "bowl" and in_bowl(p[block], p[name]):
            return f"{block} is in {name} at {xy}"
        if name != block and obj["kind"] == "block" and on_top(p[block], p[name]):
            return f"{block} is on {name} at {xy}"
    if on_table(p[block]):
        return f"{block} is on the table at {xy}"
    return f"{block} is at ({p[block][0]:.2f}, {p[block][1]:.2f}, {p[block][2]:.2f})"


@dataclass(frozen=True)
class Expectation:
    """The effect one executed step should have."""

    step: ToolCall
    block: str | None  # the block the step acts on (the held block, for a place)

    def mismatch(self, state: State, arm_at_home: bool) -> str | None:
        """None if the effect is there; otherwise what was expected and what is actually the case."""
        if self.step.name == "move_home":
            return None if arm_at_home else f"Expected the arm to be at home after move_home(), but it is not. {held_text(state)}"
        if self.block is None:
            return None
        p = positions(state)[self.block]
        name = self.step.name
        if name == "pick":
            ok = state["held"] == self.block and p[2] > ON_TABLE_Z + Z_TOL  # held, and clearly above the table
            expected = f"{self.block} to be held"
        elif name == "place_on":
            target = self.step.args["obj_name"]
            kinds = {o["name"]: o["kind"] for o in state["objects"]}
            into = kinds[target] == "bowl"
            rests = (in_bowl if into else on_top)(p, positions(state)[target])
            ok = state["held"] is None and rests
            expected = f"{self.block} to rest {'in' if into else 'on'} {target}"
        elif name == "place_at":
            x, y = float(self.step.args["x"]), float(self.step.args["y"])
            ok = state["held"] is None and np.hypot(p[0] - x, p[1] - y) <= UNDO_XY_TOL and on_table(p)
            expected = f"{self.block} to rest on the table at ({x:.2f}, {y:.2f})"
        else:
            return None
        if ok:
            return None
        return f"Expected {expected} after {step_text(self.step)}, but {where(state, self.block)}. {held_text(state)}"


def expectation_for(step: ToolCall, held_before: str | None) -> Expectation:
    """The expected effect of `step`, given which block the gripper held before it ran."""
    if step.name == "pick":
        return Expectation(step, step.args["obj_name"])
    if step.name in ("place_on", "place_at"):
        return Expectation(step, held_before)
    return Expectation(step, None)


def failure_text(step: ToolCall, message: str, state: State) -> str:
    """The mismatch for a primitive that reported failure: its own message, plus what is held."""
    return f"{step_text(step)} failed: {message.rstrip('.')}. {held_text(state)}"
