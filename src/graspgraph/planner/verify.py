"""What each step should have achieved, checked against the scene afterwards.

The closed-loop planner verifies every step right after it runs, and at the end of a plan
re-checks every placement still in force (a block can be knocked out of a bowl later). The
checks reuse the task predicates' tolerances, but they only ever describe the planner's own
steps: the task's success predicate stays hidden from the planner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graspgraph.actions import ToolCall
from graspgraph.tasks.predicates import UNDO_XY_TOL, State, in_bowl, on_table, on_top, positions


@dataclass(frozen=True)
class Expectation:
    """The effect one executed step should have on the scene."""

    step: ToolCall
    block: str | None  # the block the step moved (the held block, for a place)
    description: str

    def holds(self, state: State) -> bool:
        """Whether the effect is visible in `state`."""
        if self.block is None:
            return True
        if self.step.name == "pick":
            return state["held"] == self.block
        p = positions(state)
        if state["held"] == self.block:
            return False
        if self.step.name == "place_on":
            target = self.step.args["obj_name"]
            kinds = {o["name"]: o["kind"] for o in state["objects"]}
            check = in_bowl if kinds[target] == "bowl" else on_top
            return check(p[self.block], p[target])
        if self.step.name == "place_at":
            x, y = self.step.args["x"], self.step.args["y"]
            return bool(np.hypot(p[self.block][0] - x, p[self.block][1] - y) <= UNDO_XY_TOL) and on_table(p[self.block])
        return True


def expectation_for(step: ToolCall, held_before: str | None) -> Expectation:
    """The expected effect of `step`, given which block the gripper held before it ran."""
    if step.name == "pick":
        block = step.args["obj_name"]
        return Expectation(step, block, f"holding {block}")
    if step.name == "place_on" and held_before:
        return Expectation(step, held_before, f"{held_before} resting on/in {step.args['obj_name']}")
    if step.name == "place_at" and held_before:
        return Expectation(step, held_before, f"{held_before} on the table at ({step.args['x']:.3f}, {step.args['y']:.3f})")
    return Expectation(step, None, "no change to check")
