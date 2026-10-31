"""Read tool calls out of a model response, defensively.

Backends disagree on how arguments arrive: a dict, a JSON string, or a dict wrapped in an
"arguments"/"parameters" key. Calls LangChain could not parse are kept with empty
arguments, so they show up as failed calls instead of vanishing.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage

from graspgraph.actions import ToolCall

WRAPPER_KEYS = ("arguments", "parameters", "args", "input")


def coerce_args(raw: Any) -> dict[str, Any]:
    """Arguments as a dict, whether they arrived as a dict, a JSON string or a wrapped dict."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, dict):
        return {}
    if len(raw) == 1 and next(iter(raw)) in WRAPPER_KEYS:
        return coerce_args(next(iter(raw.values())))
    return raw


def parse_tool_calls(message: AIMessage) -> list[ToolCall]:
    """Tool calls in the order the model gave them; unparseable ones come last with no arguments."""
    calls = [ToolCall(str(c.get("name", "")), coerce_args(c.get("args"))) for c in message.tool_calls]
    calls += [ToolCall(str(c.get("name") or ""), coerce_args(c.get("args"))) for c in message.invalid_tool_calls]
    return calls
