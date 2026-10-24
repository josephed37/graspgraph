"""Planner prompts, kept in one place so they can be versioned (see PROMPT_VERSIONS)."""

from __future__ import annotations

import json
from typing import Any

from graspgraph.planner.plans import PlanProtocol
from graspgraph.sim.scene import TABLE_X, TABLE_Y
from graspgraph.tasks.regions import describe_regions

# One version per protocol, written into every benchmark record. Bump when wording changes.
# v3 (both protocols): the move history now says what "from" and "to" mean and how to undo.
PROMPT_VERSIONS: dict[PlanProtocol, str] = {"multi_call": "open-v3-multi-call", "submit_plan": "open-v3-submit-plan"}

HISTORY_INTRO = (
    "Moves made so far, oldest first. Each move names the block that moved, where it was before the move "
    '("from") and where it ended up ("to"), as [x, y, z] in metres. To undo a move, pick that block up and '
    'place_at the x and y of its "from" position.'
)

_WORLD = f"""\
You control a Franka Panda robot arm that moves coloured blocks on a table. You write the plan; the robot runs it.

World
- Positions are in metres in the robot frame. The robot base is at (0, 0). x points away from the robot and y points to the robot's left. "Front" means away from the robot, "back" towards it, "left" is +y and "right" is -y: directions are from the robot's point of view.
- The table top spans x {TABLE_X[0]:.2f} to {TABLE_X[1]:.2f} and y {TABLE_Y[0]:.2f} to {TABLE_Y[1]:.2f}. The arm reaches the table at about x 0.30 to 0.70 and y -0.35 to 0.35.
- Blocks are 4 cm cubes. Bowls are 12 cm square trays fixed to the table: they cannot be picked up or moved. A bowl holds at most two blocks.
- The gripper holds one block at a time. A block with another block on top of it cannot be picked up.

Named table regions
{describe_regions()}
"""

_PRIMITIVES = """\
- pick(obj_name): pick up a block.
- place_on(obj_name): put the held block on top of another block or into a bowl.
- place_at(x, y): put the held block on the table at (x, y). Keep the spot at least 12 cm from any bowl's centre and 9 cm from other blocks.
- move_home(): return the arm to its home pose.
"""

_DECIDE = """\
- Use object names exactly as they appear in the scene (for example "block_red", never "red block").
- Every place_on or place_at must come right after a pick.
- If the instruction could mean more than one object or place, call only ask_clarification.
- If it names something that does not exist or asks for something the robot cannot do, call only refuse.
- Otherwise do not ask or refuse: plan the moves.
"""

# The original protocol: one tool call per step.
MULTI_CALL_SYSTEM_PROMPT = (
    _WORLD
    + "\nTools\n"
    + _PRIMITIVES
    + """\
- ask_clarification(question): ask the user when the instruction is ambiguous. Ends the plan.
- refuse(reason): decline an instruction that is impossible. Ends the plan.

Rules
- Reply with the complete plan in a single response: every tool call needed, in the order they must run. You will not see any results and you get no second turn.
"""
    + _DECIDE
)

SUBMIT_PLAN_SYSTEM_PROMPT = (
    _WORLD
    + """
Tools
You have three tools and must call exactly one of them, exactly once:
- submit_plan(steps): the complete plan, as an ordered list of steps. Each step is {"tool": <name>, "args": {...}} using one of these robot actions:
"""
    + _PRIMITIVES.replace("- ", "  - ")
    + """\
- ask_clarification(question): ask the user when the instruction is ambiguous.
- refuse(reason): decline an instruction that is impossible.

Rules
- Put every step in one submit_plan call, in the order they must run. You will not see any results and you get no second turn, so a plan that stops early leaves the task unfinished.
- Example: to put block_red in bowl_blue and then stack block_green on block_red, call submit_plan with steps
  [{"tool": "pick", "args": {"obj_name": "block_red"}}, {"tool": "place_on", "args": {"obj_name": "bowl_blue"}},
   {"tool": "pick", "args": {"obj_name": "block_green"}}, {"tool": "place_on", "args": {"obj_name": "block_red"}}]
"""
    + _DECIDE
)

SYSTEM_PROMPTS: dict[PlanProtocol, str] = {"multi_call": MULTI_CALL_SYSTEM_PROMPT, "submit_plan": SUBMIT_PLAN_SYSTEM_PROMPT}


def build_user_message(instruction: str, scene: dict[str, Any], history: list[dict[str, Any]]) -> str:
    """The instruction, the current scene from get_scene() and, if any, the moves made so far."""
    parts = [
        f"Instruction: {instruction}",
        "Scene (from get_scene(); positions are object centres in metres):",
        json.dumps(scene, indent=1),
    ]
    if history:
        parts += [HISTORY_INTRO, json.dumps(history, indent=1)]
    return "\n\n".join(parts)
