"""Tool schemas the planner LLM sees, in the OpenAI function format LangChain accepts.

Every object-name argument is an enum of the names actually in the current scene, so a
model cannot invent names like "red block" (Gemma did, before enums).
"""

from __future__ import annotations

from typing import Any

from graspgraph.actions import ALL_TOOLS, ASK, MOTION_TOOLS, REFUSE, STOP_TOOLS
from graspgraph.planner.plans import SUBMIT_PLAN, PlanProtocol

DESCRIPTIONS = {
    "pick": "Pick up a block. The gripper must be empty and nothing may be on top of the block.",
    "place_on": "Put the held block on top of another block, or into a bowl.",
    "place_at": "Put the held block on the table with its centre at (x, y), in metres.",
    "move_home": "Move the arm back to its home pose.",
    SUBMIT_PLAN: "Submit the complete plan: every step, in the order the robot must run them. Call this once.",
    ASK: "Ask the user a question when the instruction is ambiguous. Ends the plan; nothing else runs.",
    REFUSE: "Decline an instruction that is impossible. Ends the plan; nothing else runs.",
}


def _function(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _params(object_names: list[str]) -> dict[str, dict[str, Any]]:
    obj = {"type": "string", "enum": list(object_names), "description": "exact object name from the scene"}
    return {
        "pick": {"obj_name": obj},
        "place_on": {"obj_name": obj},
        "place_at": {
            "x": {"type": "number", "description": "metres, away from the robot"},
            "y": {"type": "number", "description": "metres, to the robot's left"},
        },
        "move_home": {},
        ASK: {"question": {"type": "string"}},
        REFUSE: {"reason": {"type": "string"}},
    }


def multi_call_schemas(object_names: list[str]) -> list[dict[str, Any]]:
    """One tool per primitive plus ask/refuse; the plan is the sequence of calls."""
    params = _params(object_names)
    return [_function(name, DESCRIPTIONS[name], params[name], list(ALL_TOOLS[name])) for name in ALL_TOOLS]


def submit_plan_schemas(object_names: list[str]) -> list[dict[str, Any]]:
    """submit_plan (the whole plan as an ordered list of steps) plus ask/refuse."""
    params = _params(object_names)
    # One schema per robot action, so each tool offers exactly its own arguments (a flat schema
    # listing obj_name, x and y together led Gemma to send place_at an obj_name).
    step = {
        "anyOf": [
            {
                "type": "object",
                "description": DESCRIPTIONS[name],
                "properties": {
                    "tool": {"type": "string", "enum": [name]},
                    "args": {
                        "type": "object",
                        "properties": params[name],
                        "required": list(MOTION_TOOLS[name]),
                        "additionalProperties": False,
                    },
                },
                "required": ["tool", "args"],
            }
            for name in MOTION_TOOLS
        ]
    }
    submit = _function(
        SUBMIT_PLAN,
        DESCRIPTIONS[SUBMIT_PLAN],
        {"steps": {"type": "array", "items": step, "minItems": 1, "description": "every step, in execution order"}},
        ["steps"],
    )
    stops = [_function(name, DESCRIPTIONS[name], params[name], list(STOP_TOOLS[name])) for name in STOP_TOOLS]
    return [submit, *stops]


def tool_schemas(object_names: list[str], protocol: PlanProtocol = "submit_plan") -> list[dict[str, Any]]:
    """Schemas for a planning protocol, with object arguments limited to `object_names`."""
    return submit_plan_schemas(object_names) if protocol == "submit_plan" else multi_call_schemas(object_names)
