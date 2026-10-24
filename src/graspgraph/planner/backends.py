"""Model backends behind one small interface: messages and tool schemas in, an AIMessage out.

- `ollama_backend`: local models via langchain-ollama, with Ollama's thinking toggle.
- `anthropic_backend`: Claude via langchain-anthropic, low max_tokens, no extended thinking.
- `FakeBackend`: scripted tool calls for tests and dry runs; never touches the network.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from graspgraph.actions import STOP_TOOLS, ToolCall
from graspgraph.tasks import Task

DEFAULT_MODELS = {"ollama": "gemma4:12b", "anthropic": "claude-haiku-5-5", "fake": "oracle"}

# Output-token caps when --max-tokens is not given. A plan needs a few hundred tokens; the cap
# stops a model stuck repeating itself within a minute or two. Ollama counts thinking tokens
# against the cap, so thinking runs get more room.
DEFAULT_MAX_TOKENS = {"anthropic": 1024, "ollama": 2048, "ollama_think": 8192}

# Errors that mean the model itself failed (it was generating), not the connection or server.
# Ollama aborts a response when the model loops on the same tokens.
MODEL_FAILURE_MARKERS = ("token repeat limit",)


def is_model_failure(error: Exception | str) -> bool:
    """Whether an error from a model call is the model's own failure (not worth retrying)."""
    return any(marker in str(error).lower() for marker in MODEL_FAILURE_MARKERS)


class PlannerBackend(ABC):
    """A model that turns a prompt into tool calls."""

    def __init__(self, backend: str, model: str, think: bool | None = None) -> None:
        self.backend = backend
        self.model = model
        self.think = think

    def prepare(self, task: Task) -> None:
        """Called before each task. Real models ignore it; FakeBackend uses it to pick its script."""

    @abstractmethod
    def generate(self, messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage:
        """One model call. The response's tool_calls are the plan."""


class ChatModelBackend(PlannerBackend):
    """Any LangChain chat model that supports tool calling."""

    def __init__(self, chat_model: BaseChatModel, backend: str, model: str, think: bool | None = None) -> None:
        super().__init__(backend, model, think)
        self.chat_model = chat_model

    def generate(self, messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage:
        return self.chat_model.bind_tools(tools).invoke(messages)


def ollama_backend(
    model: str = DEFAULT_MODELS["ollama"],
    think: bool = False,
    base_url: str | None = None,
    num_ctx: int = 8192,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatModelBackend:
    """A local Ollama model. `think` is Ollama's thinking toggle (passed as `reasoning`).

    `temperature` None keeps the model's default sampling, so repeated runs measure variance.
    `max_tokens` caps the output (Ollama's num_predict, thinking included).
    """
    from langchain_ollama import ChatOllama

    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS["ollama_think" if think else "ollama"]
    chat = ChatOllama(
        model=model,
        reasoning=think,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=max_tokens,
        base_url=base_url,
    )
    return ChatModelBackend(chat, "ollama", model, think)


def anthropic_backend(
    model: str = DEFAULT_MODELS["anthropic"],
    max_tokens: int = DEFAULT_MAX_TOKENS["anthropic"],
    temperature: float | None = None,
) -> ChatModelBackend:
    """A Claude model. Reads ANTHROPIC_API_KEY from the environment; extended thinking stays off.

    `temperature` None keeps the API's default sampling.
    """
    from langchain_anthropic import ChatAnthropic

    chat = ChatAnthropic(model=model, max_tokens=max_tokens, temperature=temperature)
    return ChatModelBackend(chat, "anthropic", model, None)


Policy = Callable[[Task], list[ToolCall] | AIMessage]


class FakeBackend(PlannerBackend):
    """Answers with the plan a policy scripts for the current task, with made-up token usage.

    A policy returns the plan as a list of steps, which the fake formats for the protocol the
    offered tools imply (one submit_plan call, or one call per step), or a ready AIMessage
    when a test needs exact control over the raw output.
    """

    def __init__(self, policy: Policy, model: str = "custom", input_tokens: int = 0, output_tokens: int = 0) -> None:
        super().__init__("fake", model, None)
        self.policy = policy
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self._task: Task | None = None

    def prepare(self, task: Task) -> None:
        self._task = task

    def generate(self, messages: list[BaseMessage], tools: list[dict[str, Any]]) -> AIMessage:
        if self._task is None:
            raise RuntimeError("FakeBackend.prepare(task) must be called before generate()")
        planned = self.policy(self._task)
        if isinstance(planned, AIMessage):
            return planned
        offered = {t["function"]["name"] for t in tools}
        calls = planned
        if "submit_plan" in offered and planned and planned[0].name not in STOP_TOOLS:
            calls = [ToolCall("submit_plan", {"steps": [{"tool": c.name, "args": dict(c.args)} for c in planned]})]
        return AIMessage(
            content="",
            tool_calls=[{"name": c.name, "args": dict(c.args), "id": f"call_{i}"} for i, c in enumerate(calls)],
            usage_metadata={
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
            },
        )


def oracle_policy(task: Task) -> list[ToolCall]:
    """The task's own correct plan."""
    return list(task.oracle)


def wrong_object_policy(task: Task) -> list[ToolCall]:
    """The oracle plan with its first pick swapped for a block the instruction does not mention.

    Tasks that involve every block (e.g. tower) have no such block and get the oracle plan.
    """
    from graspgraph.sim import build_scene

    calls = list(task.oracle)
    mentioned = {n for c in calls for n in c.object_names()}
    blocks = [o.name for o in build_scene(task.scene_config).objects if o.kind == "block"]
    others = [b for b in blocks if b not in mentioned]
    for i, call in enumerate(calls):
        if call.name == "pick" and others:
            calls[i] = ToolCall("pick", {"obj_name": others[0]})
            break
    return calls


def first_step_policy(task: Task) -> list[ToolCall]:
    """Only the oracle's first step: how Gemma behaved under the multi_call protocol."""
    return list(task.oracle[:1])


def no_tools_policy(task: Task) -> list[ToolCall]:
    """Answers in prose only: no tool calls at all."""
    return []


FAKE_POLICIES: dict[str, Policy] = {
    "oracle": oracle_policy,
    "wrong-object": wrong_object_policy,
    "no-tools": no_tools_policy,
    "first-step": first_step_policy,
}


def make_backend(
    backend: str,
    model: str | None = None,
    think: bool = False,
    max_tokens: int | None = None,
    ollama_url: str | None = None,
    temperature: float | None = None,
) -> PlannerBackend:
    """Build a backend by name: "ollama", "anthropic" or "fake" (model = a FAKE_POLICIES key).

    `max_tokens` None uses DEFAULT_MAX_TOKENS for the backend.
    """
    model = model or DEFAULT_MODELS[backend]
    if backend == "ollama":
        return ollama_backend(model, think=think, base_url=ollama_url, temperature=temperature, max_tokens=max_tokens)
    if backend == "anthropic":
        return anthropic_backend(model, max_tokens=max_tokens or DEFAULT_MAX_TOKENS["anthropic"], temperature=temperature)
    if backend == "fake":
        if model not in FAKE_POLICIES:
            raise ValueError(f"unknown fake policy '{model}'; choose from {list(FAKE_POLICIES)}")
        return FakeBackend(FAKE_POLICIES[model], model=model)
    raise ValueError(f"unknown backend '{backend}'")
