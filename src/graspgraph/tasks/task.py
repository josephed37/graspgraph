"""The Task record and the runtime that turns one into a live episode."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from graspgraph.actions import ToolCall, execute
from graspgraph.sim import Scene, SceneConfig, build_scene
from graspgraph.sim.robot import Robot
from graspgraph.tasks.predicates import Predicate, State, Verdict

ExpectedOutcome = Literal["act", "ask", "refuse"]


@dataclass(frozen=True)
class Task:
    task_id: str  # e.g. "stack-007"
    family: str
    seed: int
    scene_config: SceneConfig
    instruction: str
    expected_outcome: ExpectedOutcome
    predicate: Predicate  # checked on the final simulator state
    oracle: tuple[ToolCall, ...]  # a correct plan, proving the task is solvable
    setup: tuple[ToolCall, ...] = ()  # run before the planner starts (e.g. the move to undo)
    # Object kind whose members are interchangeable when scoring (e.g. "block" for "put two
    # blocks in each bowl"): using different ones than the oracle is not a wrong object.
    interchangeable: str | None = None


@dataclass
class Episode:
    """A task's scene after setup, ready for a planner."""

    task: Task
    scene: Scene
    robot: Robot
    start_state: State  # get_scene() when the planner starts
    history: list[dict[str, Any]] = field(default_factory=list)  # moves made during setup

    def verdict(self) -> Verdict:
        """Check the task's predicate against the current simulator state."""
        return self.task.predicate.check(self.start_state, self.robot.get_scene())


def start_episode(task: Task, on_step: Callable[[], None] | None = None) -> Episode:
    """Build the task's scene and run its setup, recording each pick-and-place as history.

    Raises RuntimeError if the setup itself fails: that is a bug in the task, not a planner failure.
    """
    scene = build_scene(task.scene_config)
    robot = Robot(scene, on_step=on_step)
    history: list[dict[str, Any]] = []
    picked: tuple[str, list[float]] | None = None
    calls: list[str] = []
    for call in task.setup:
        before = _position(robot, call.args["obj_name"]) if call.name == "pick" else None
        result = execute(robot, call)
        if not result.success:
            raise RuntimeError(f"setup of {task.task_id} failed at {call}: {result.message}")
        calls.append(str(call))
        if call.name == "pick":
            picked = (call.args["obj_name"], before)
        elif picked is not None:
            block, origin = picked
            history.append(
                {"move": len(history) + 1, "calls": calls, "block": block, "from": origin, "to": _position(robot, block)}
            )
            picked, calls = None, []
    return Episode(task, scene, robot, robot.get_scene(), history)


def _position(robot: Robot, name: str) -> list[float]:
    return [round(float(v), 3) for v in np.asarray(robot.data.body(name).xpos)]
