"""Tool schemas the planner LLM sees, in the OpenAI function format LangChain accepts.

Every object-name argument is an enum of the names actually in the current scene, so a
model cannot invent names like "red block" (Gemma did, before enums).
"""

from __future__ import annotations

from typing import Any

from graspgraph.actions import ALL_TOOLS, ASK, REFUSE

DESCRIPTIONS = {
    "pick": "Pick up a block. The gripper must be empty and nothing may be on top of the block.",
    "place_on": "Put the held block on top of another block, or into a bowl.",
    "place_at": "Put the held block on the table with its centre at (x, y), in metres.",
    "move_home": "Move the arm back to its home pose.",
    ASK: "Ask the user a question when the instruction is ambiguous. Ends the plan; nothing else runs.",
    REFUSE: "Decline an instruction that is impossible. Ends the plan; nothing else runs.",
}


def tool_schemas(object_names: list[str]) -> list[dict[str, Any]]:
    """Schemas for every planner tool, with object arguments limited to `object_names`."""
    obj = {"type": "string", "enum": list(object_names), "description": "exact object name from the scene"}
    coord = {"type": "number"}
    params: dict[str, dict[str, Any]] = {
        "pick": {"obj_name": obj},
        "place_on": {"obj_name": obj},
        "place_at": {"x": {**coord, "description": "metres, away from the robot"}, "y": {**coord, "description": "metres, to the robot's left"}},
        "move_home": {},
        ASK: {"question": {"type": "string"}},
        REFUSE: {"reason": {"type": "string"}},
    }
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": DESCRIPTIONS[name],
                "parameters": {"type": "object", "properties": params[name], "required": list(ALL_TOOLS[name])},
            },
        }
        for name in ALL_TOOLS
    ]
