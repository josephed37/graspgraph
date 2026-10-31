"""Send a benchmark task's exact planner request to Ollama and keep the raw reply.

For diagnosing responses that come back with no text and no tool calls: the request is
built by the same ChatOllama the benchmark uses (so prompt, tools and options match), but
the reply is read straight from the Ollama client, before LangChain parses it. Streaming
can be switched off to check whether tool calls get lost when chunks are stitched together.

This reaches into ChatOllama internals (`_chat_params`, `_client`); it is a debugging aid,
not part of the benchmark.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from graspgraph.planner.backends import ollama_backend
from graspgraph.planner.plans import PlanProtocol
from graspgraph.planner.prompts import SYSTEM_PROMPTS, build_user_message
from graspgraph.planner.tools import tool_schemas
from graspgraph.tasks import get_task, start_episode


def build_request(
    task_id: str,
    model: str,
    protocol: PlanProtocol = "submit_plan",
    think: bool = False,
    max_tokens: int | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """The Ollama client and the exact chat() parameters the benchmark would send for a task."""
    episode = start_episode(get_task(task_id))
    messages = [
        SystemMessage(SYSTEM_PROMPTS[protocol]),
        HumanMessage(build_user_message(episode.task.instruction, episode.robot.get_scene(), episode.history)),
    ]
    tools = tool_schemas([o.name for o in episode.scene.objects], protocol)
    chat = ollama_backend(model, think=think, base_url=base_url, temperature=temperature, max_tokens=max_tokens).chat_model
    return chat._client, chat._chat_params(messages, tools=tools)


def _as_dict(part: Any) -> dict[str, Any]:
    return part.model_dump(exclude_none=True) if hasattr(part, "model_dump") else dict(part)


def raw_chat(client: Any, params: dict[str, Any], stream: bool) -> dict[str, Any]:
    """Call Ollama once and summarise what came back, keeping every raw chunk."""
    reply = client.chat(**{**params, "stream": stream})
    chunks = [_as_dict(part) for part in (reply if stream else [reply])]
    messages = [c.get("message", {}) for c in chunks]
    last = chunks[-1] if chunks else {}
    return {
        "stream": stream,
        "content": "".join(m.get("content") or "" for m in messages),
        "thinking": "".join(m.get("thinking") or "" for m in messages),
        "tool_calls": [call for m in messages for call in (m.get("tool_calls") or [])],
        "done_reason": last.get("done_reason"),
        "output_tokens": last.get("eval_count"),
        "chunks": chunks,
    }
