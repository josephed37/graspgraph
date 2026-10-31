"""Seeded fault injection: things going wrong in the world while a plan runs.

Three fault types, injected at the execution layer (around primitive calls), so a planner
only finds out through what it observes, as a real robot would:

- "drop": the held block slips out of the gripper early in a carry (between the pick and the
  placement), and falls to the table.
- "displace": just before a pick, the block to be picked is moved to another free spot.
- "knock": just after a successful placement, the block that was placed is moved off its
  target to another free spot.

Relocated blocks go to a free, reachable spot on the table, clear of every other object, so
the world stays valid and recovery is possible. A dropped block falls where it was lifted.

Everything is derived from (task_id, repeat, fault_seed): whether the run gets a fault, its
type, and which occurrence it hits (the n-th pick or placement the plan executes). The same
task and repeat therefore get the same fault for every model and in both modes. Only tasks
with a manipulation step are eligible; ask and refuse tasks never get one.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import mujoco
import numpy as np

from graspgraph.actions import ToolCall
from graspgraph.sim.robot import GRIPPER_OPEN, PrimitiveResult, Robot
from graspgraph.sim.scene import BLOCK_HALF, TABLE_HEIGHT, WORKSPACE_X, WORKSPACE_Y, footprint_radius
from graspgraph.tasks import Task

FaultKind = Literal["drop", "displace", "knock"]
FAULT_KINDS: tuple[FaultKind, ...] = ("drop", "displace", "knock")
PLACING = ("place_on", "place_at")
DEFAULT_FAULT_RATE = 0.3  # --faults with no value
DROP_AFTER_STEPS = 30  # physics steps into the place (0.06 s): early in the carry, far from the target
MOVE_MIN_DISTANCE = 0.12  # a relocated block lands at least 12 cm from where it was
SPOT_MARGIN = 0.035  # clearance around other objects for a relocated block (as for oracle spots)
SPOT_GRID = 0.05  # candidate spots: a grid over the spawn workspace, which the arm reaches
GRIPPER_ACTUATOR = 7


@dataclass(frozen=True)
class FaultPlan:
    """Which fault a run gets, and which occurrence of its trigger step it hits (0 = first)."""

    kind: FaultKind
    occurrence: int


def _rng(task_id: str, repeat: int, seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng([zlib.crc32(task_id.encode()), repeat, seed, stream])


def choose_fault(task: Task, repeat: int, rate: float, seed: int = 0) -> FaultPlan | None:
    """The fault (if any) for this run: about `rate` of the eligible tasks get one, reproducibly."""
    picks = sum(s.name == "pick" for s in task.oracle)
    places = sum(s.name in PLACING for s in task.oracle)
    if rate <= 0 or task.expected_outcome != "act" or picks == 0:
        return None
    rng = _rng(task.task_id, repeat, seed, 0)
    if rng.random() >= rate:
        return None
    kind = FAULT_KINDS[int(rng.integers(len(FAULT_KINDS)))]
    # The occurrence is drawn against the oracle plan's length; a model's plan may be shorter.
    occurrence = int(rng.integers(picks if kind == "displace" else max(places, 1)))
    return FaultPlan(kind, occurrence)


@dataclass
class FaultInjector:
    """A planner StepHook that injects one planned fault, the first time its trigger comes round."""

    plan: FaultPlan
    rng: np.random.Generator
    injected: bool = False
    step: int | None = None  # 1-based index of the executed motion step the fault hit
    detail: str = "the plan never reached the step this fault targets"
    _executed: int = 0
    _picks: int = 0
    _places: int = 0
    _restore_on_step: Callable[[], None] | None = None
    _drop_armed: bool = False
    _held_before: str | None = None

    @classmethod
    def for_run(cls, task: Task, plan: FaultPlan, repeat: int = 0, seed: int = 0) -> FaultInjector:
        return cls(plan, _rng(task.task_id, repeat, seed, 1))

    def record(self) -> dict[str, Any]:
        """The per-run record fields."""
        return {
            "fault_injected": self.injected,
            "fault_type": self.plan.kind,
            "fault_step": self.step,
            "fault_detail": self.detail,
        }

    # ------------------------------------------------------------------ hook

    def before_step(self, robot: Robot, step: ToolCall) -> None:
        self._executed += 1
        if self.injected:
            return
        if self.plan.kind == "displace" and step.name == "pick":
            if self._picks == self.plan.occurrence:
                self._relocate(robot, step.args.get("obj_name"), "was moved before it could be picked")
            self._picks += 1
        elif self.plan.kind == "drop" and step.name in PLACING and robot.held:
            if self._places == self.plan.occurrence:
                self._arm_drop(robot)
            self._places += 1
        self._held_before = robot.held

    def after_step(self, robot: Robot, step: ToolCall, result: PrimitiveResult) -> None:
        if self._drop_armed:
            robot.on_step, self._drop_armed = self._restore_on_step, False
        if self.injected or self.plan.kind != "knock" or step.name not in PLACING or not result.success:
            return
        if self._places == self.plan.occurrence:
            self._relocate(robot, self._held_before, "was knocked off where it had just been placed")
        self._places += 1

    # ----------------------------------------------------------------- faults

    def _arm_drop(self, robot: Robot) -> None:
        """Open the gripper DROP_AFTER_STEPS physics steps into the placement, mid-carry."""
        block, previous, steps = robot.held, robot.on_step, 0
        index = self._executed

        def count_and_drop() -> None:
            nonlocal steps
            if previous:
                previous()
            steps += 1
            if steps == DROP_AFTER_STEPS:
                robot.data.ctrl[GRIPPER_ACTUATOR] = GRIPPER_OPEN
                height = robot.data.body(block).xpos[2] - TABLE_HEIGHT
                self._mark(index, f"{block} slipped from the gripper {height * 100:.0f} cm above the table")

        self._restore_on_step, self._drop_armed = previous, True
        robot.on_step = count_and_drop

    def _relocate(self, robot: Robot, block: str | None, what: str) -> None:
        """Move `block` to a random free spot on the table (skipped if it is held or covered)."""
        state = robot.get_scene()
        here = {o["name"]: np.array(o["position"]) for o in state["objects"]}
        if block not in here or state["held"] == block or _covered(here, block):
            return
        spot = free_spot(robot, block, self.rng)
        if spot is None:
            return
        teleport(robot, block, spot)
        before = f"({here[block][0]:.3f}, {here[block][1]:.3f})"
        self._mark(self._executed, f"{block} {what}: moved from {before} to ({spot[0]:.3f}, {spot[1]:.3f})")

    def _mark(self, step: int, detail: str) -> None:
        self.injected, self.step, self.detail = True, step, detail


def _covered(here: dict[str, np.ndarray], block: str) -> bool:
    """Whether another block rests on `block` (moving it would carry that one along)."""
    return any(
        name != block and np.linalg.norm(p[:2] - here[block][:2]) < 2 * BLOCK_HALF and p[2] > here[block][2] + BLOCK_HALF
        for name, p in here.items()
    )


def free_spot(robot: Robot, block: str, rng: np.random.Generator) -> np.ndarray | None:
    """A random grid spot in the reachable workspace, clear of every other object and far from `block`."""
    here = {o["name"]: np.array(o["position"][:2]) for o in robot.get_scene()["objects"]}
    kinds = {o.name: o.kind for o in robot.scene.objects}
    candidates = [
        np.array([x, y])
        for x in np.arange(WORKSPACE_X[0], WORKSPACE_X[1] + 1e-9, SPOT_GRID)
        for y in np.arange(WORKSPACE_Y[0], WORKSPACE_Y[1] + 1e-9, SPOT_GRID)
    ]
    free = [
        p
        for p in candidates
        if np.linalg.norm(p - here[block]) >= MOVE_MIN_DISTANCE
        and all(
            np.linalg.norm(p - xy) >= footprint_radius(kinds[n]) + footprint_radius("block") + SPOT_MARGIN
            for n, xy in here.items()
            if n != block
        )
    ]
    return free[int(rng.integers(len(free)))] if free else None


def teleport(robot: Robot, block: str, xy: np.ndarray) -> None:
    """Put a free-jointed block down upright at (x, y) on the table, at rest."""
    joint = robot.model.body(block).jntadr[0]
    qpos, dof = robot.model.jnt_qposadr[joint], robot.model.jnt_dofadr[joint]
    robot.data.qpos[qpos : qpos + 7] = [xy[0], xy[1], TABLE_HEIGHT + BLOCK_HALF, 1, 0, 0, 0]
    robot.data.qvel[dof : dof + 6] = 0
    mujoco.mj_forward(robot.model, robot.data)
