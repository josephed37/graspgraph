"""Planner prompts, kept in one place so they can be versioned.

Bump PROMPT_VERSION whenever the wording changes: it is written into every benchmark record.
"""

from __future__ import annotations

import json
from typing import Any

from graspgraph.sim.scene import TABLE_X, TABLE_Y
from graspgraph.tasks.regions import describe_regions

PROMPT_VERSION = "open-v1"

SYSTEM_PROMPT = f"""\
You control a Franka Panda robot arm that moves coloured blocks on a table. You write the plan; the robot runs it.

World
- Positions are in metres in the robot frame. The robot base is at (0, 0). x points away from the robot and y points to the robot's left. "Front" means away from the robot, "back" towards it, "left" is +y and "right" is -y: directions are from the robot's point of view.
- The table top spans x {TABLE_X[0]:.2f} to {TABLE_X[1]:.2f} and y {TABLE_Y[0]:.2f} to {TABLE_Y[1]:.2f}. The arm reaches the table at about x 0.30 to 0.70 and y -0.35 to 0.35.
- Blocks are 4 cm cubes. Bowls are 12 cm square trays fixed to the table: they cannot be picked up or moved. A bowl holds at most two blocks.
- The gripper holds one block at a time. A block with another block on top of it cannot be picked up.

Named table regions
{describe_regions()}

Tools
- pick(obj_name): pick up a block.
- place_on(obj_name): put the held block on top of another block or into a bowl.
- place_at(x, y): put the held block on the table at (x, y). Keep the spot at least 12 cm from any bowl's centre and 9 cm from other blocks.
- move_home(): return the arm to its home pose.
- ask_clarification(question): ask the user when the instruction is ambiguous. Ends the plan.
- refuse(reason): decline an instruction that is impossible. Ends the plan.

Rules
- Reply with the complete plan in a single response: every tool call needed, in the order they must run. You will not see any results and you get no second turn.
- Use object names exactly as they appear in the scene (for example "block_red", never "red block").
- Every place_on or place_at must come right after a pick.
- If the instruction could mean more than one object or place, call only ask_clarification.
- If it names something that does not exist or asks for something the robot cannot do, call only refuse.
- Otherwise do not ask or refuse: plan the moves.
"""


def build_user_message(instruction: str, scene: dict[str, Any], history: list[dict[str, Any]]) -> str:
    """The instruction, the current scene from get_scene() and, if any, the moves made so far."""
    parts = [
        f"Instruction: {instruction}",
        "Scene (from get_scene(); positions are object centres in metres):",
        json.dumps(scene, indent=1),
    ]
    if history:
        parts += ["Moves made so far, oldest first:", json.dumps(history, indent=1)]
    return "\n\n".join(parts)
