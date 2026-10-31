"""Seeded fault injection: things going wrong in the world while a plan runs.

- "drop": the block slips out of the gripper while it is being carried. The gripper is
  forced open partway through the place, the block falls, and the place reports it.
- "displace": a block the plan has just put in a bowl or on the table is moved somewhere
  else before the robot's next step, as if a person nudged it. The step that placed it
  succeeded, so only a planner that re-checks earlier results notices.

A fault is chosen per (task, repeat) from a seeded generator, so a run is reproducible and
open and closed loop face the same faults. Faults only target tasks that need acting.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Literal

import mujoco
import numpy as np

from graspgraph.actions import ToolCall
from graspgraph.sim.robot import GRIPPER_OPEN, PrimitiveResult, Robot
from graspgraph.sim.scene import BLOCK_HALF, TABLE_HEIGHT, WORKSPACE_X, WORKSPACE_Y, footprint_radius
from graspgraph.tasks import Task

FaultKind = Literal["drop", "displace"]
PLACING = ("place_on", "place_at")
DROP_AFTER_STEPS = 30  # physics steps into the place (0.06 s): early in the carry, far from the target
DISPLACE_MIN_MOVE = 0.12  # a displaced block lands at least 12 cm from where it was
SPOT_MARGIN = 0.035  # clearance around other objects for a displaced block (as for oracle spots)
GRIPPER_ACTUATOR = 7


def _stable_seed(task_id: str) -> int:
    return zlib.crc32(task_id.encode())


def _settles_a_block(step: ToolCall) -> bool:
    """A placement into a bowl or onto the table: the kind a person might nudge afterwards."""
    return step.name == "place_at" or (step.name == "place_on" and step.args.get("obj_name", "").startswith("bowl_"))


def choose_fault(task: Task, repeat: int, rate: float, seed: int = 0) -> FaultKind | None:
    """The fault (if any) for this run: about `rate` of the act tasks get one, reproducibly."""
    if rate <= 0 or task.expected_outcome != "act" or not any(s.name in PLACING for s in task.oracle):
        return None
    rng = np.random.default_rng([_stable_seed(task.task_id), repeat, seed])
    if rng.random() >= rate:
        return None
    # Displacing needs a settled block with more of the plan still to run.
    kinds: list[FaultKind] = ["drop", "displace"] if any(map(_settles_a_block, task.oracle[:-1])) else ["drop"]
    return kinds[int(rng.integers(len(kinds)))]


@dataclass
class FaultInjector:
    """A planner StepHook that injects one fault the first time the plan gives it a chance."""

    kind: FaultKind
    rng: np.random.Generator
    triggered: bool = False
    detail: str = "the plan never reached a point where this fault applies"
    _held_before: str | None = None
    _to_displace: str | None = None  # placed by the last step: moved before the next one runs
    _restore_on_step: Callable[[], None] | None = None
    _drop_armed: bool = False

    @classmethod
    def for_run(cls, task: Task, repeat: int, kind: FaultKind, seed: int = 0) -> FaultInjector:
        return cls(kind, np.random.default_rng([_stable_seed(task.task_id), repeat, seed, 1]))

    def record(self) -> dict[str, Any]:
        return {"kind": self.kind, "triggered": self.triggered, "detail": self.detail}

    def before_step(self, robot: Robot, step: ToolCall) -> None:
        if not self.triggered:
            if self.kind == "displace" and self._to_displace:
                self._displace(robot, self._to_displace)
            elif self.kind == "drop" and step.name in PLACING and robot.held:
                self._arm_drop(robot)
        self._to_displace = None
        self._held_before = robot.held

    def after_step(self, robot: Robot, step: ToolCall, result: PrimitiveResult) -> None:
        if self._drop_armed:
            robot.on_step, self._drop_armed = self._restore_on_step, False
        if self.kind == "displace" and not self.triggered and result.success and _settles_a_block(step):
            self._to_displace = self._held_before

    def _arm_drop(self, robot: Robot) -> None:
        """Open the gripper DROP_AFTER_STEPS physics steps into the place, mid-carry."""
        block, previous, steps = robot.held, robot.on_step, 0

        def count_and_drop() -> None:
            nonlocal steps
            if previous:
                previous()
            steps += 1
            if steps == DROP_AFTER_STEPS:
                robot.data.ctrl[GRIPPER_ACTUATOR] = GRIPPER_OPEN
                self.triggered = True
                height = robot.data.body(block).xpos[2] - TABLE_HEIGHT
                self.detail = f"{block} slipped from the gripper {height * 100:.0f} cm above the table"

        self._restore_on_step, self._drop_armed = previous, True
        robot.on_step = count_and_drop

    def _displace(self, robot: Robot, block: str) -> None:
        here = {o["name"]: np.array(o["position"][:2]) for o in robot.get_scene()["objects"]}
        kinds = {o.name: o.kind for o in robot.scene.objects}
        candidates = [
            np.array([x, y])
            for x in np.arange(WORKSPACE_X[0], WORKSPACE_X[1] + 1e-9, 0.05)
            for y in np.arange(WORKSPACE_Y[0], WORKSPACE_Y[1] + 1e-9, 0.05)
        ]
        free = [
            p
            for p in candidates
            if np.linalg.norm(p - here[block]) >= DISPLACE_MIN_MOVE
            and all(
                np.linalg.norm(p - xy) >= footprint_radius(kinds[n]) + footprint_radius("block") + SPOT_MARGIN
                for n, xy in here.items()
                if n != block
            )
        ]
        if not free:
            return
        spot = free[int(self.rng.integers(len(free)))]
        _teleport(robot, block, spot)
        self.triggered = True
        self.detail = f"{block} was moved from ({here[block][0]:.3f}, {here[block][1]:.3f}) to ({spot[0]:.3f}, {spot[1]:.3f})"


def _teleport(robot: Robot, block: str, xy: np.ndarray) -> None:
    """Put a free-jointed block down at (x, y) on the table, at rest."""
    joint = robot.model.body(block).jntadr[0]
    qpos, dof = robot.model.jnt_qposadr[joint], robot.model.jnt_dofadr[joint]
    robot.data.qpos[qpos : qpos + 7] = [xy[0], xy[1], TABLE_HEIGHT + BLOCK_HALF, 1, 0, 0, 0]
    robot.data.qvel[dof : dof + 6] = 0
    mujoco.mj_forward(robot.model, robot.data)
