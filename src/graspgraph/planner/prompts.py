"""Planner prompts, kept in one place so they can be versioned (see PROMPT_VERSIONS)."""

from __future__ import annotations

import json
from typing import Any

from graspgraph.planner.plans import PlanProtocol
from graspgraph.sim.scene import TABLE_X, TABLE_Y
from graspgraph.tasks.regions import describe_regions

# One version per protocol, written into every benchmark record. Bump when wording changes.
# The version covers everything the model sees, tool schemas included.
# v3 (both protocols): the move history says what "from" and "to" mean and how to undo.
# v4 (submit_plan): back to the flat step schema (v3's nested anyOf broke Gemma's tool calls).
PROMPT_VERSIONS: dict[PlanProtocol, str] = {"multi_call": "open-v3-multi-call", "submit_plan": "open-v4-submit-plan"}

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

# Closed loop: the same tools as submit_plan, but steps are checked as they run and the model
# is told what went wrong, with the current scene, so it can submit a new plan.
CLOSED_LOOP_PROMPT_VERSION = "closed-v1-submit-plan"
_OPEN_LOOP_RULE = (
    "- Put every step in one submit_plan call, in the order they must run. You will not see any results and you "
    "get no second turn, so a plan that stops early leaves the task unfinished.\n"
)
_CLOSED_LOOP_RULE = (
    "- Put every step in one submit_plan call, in the order they must run. The robot checks each step as it runs. "
    "If a step fails, or the plan finishes but an earlier result no longer holds (for example a block was moved), "
    "you get a report with the current scene: then submit a new plan for what is left, starting from the scene as "
    "it is now. You have a limited number of new plans.\n"
)
assert _OPEN_LOOP_RULE in SUBMIT_PLAN_SYSTEM_PROMPT
CLOSED_LOOP_SYSTEM_PROMPT = SUBMIT_PLAN_SYSTEM_PROMPT.replace(_OPEN_LOOP_RULE, _CLOSED_LOOP_RULE)


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
