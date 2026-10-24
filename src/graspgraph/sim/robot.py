"""Motion primitives for the Franka Panda: the tools a planner calls.

Every primitive returns a `PrimitiveResult` instead of raising, so a planner can read the
message and replan. Exceptions are reserved for programming errors.

All arm motion is smooth: a straight-line TCP path is turned into joint waypoints with IK,
then the position actuators' targets are swept along those waypoints with a smoothstep
time profile while the physics runs. Nothing is ever teleported.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from graspgraph.sim.ik import ARM_DOF, IKSolver, normalise_yaw, top_down_rotation
from graspgraph.sim.scene import (
    BLOCK_HALF,
    BOWL_HALF,
    BOWL_WALL_H,
    BOWL_WALL_T,
    HOME_CTRL,
    HOME_QPOS,
    TABLE_HEIGHT,
    TABLE_X,
    TABLE_Y,
    Scene,
    footprint_radius,
)

GRIPPER_OPEN = 255.0  # actuator8 ctrl: 255 = fingers fully open (8 cm), 0 = closed
GRIPPER_CLOSED = 0.0
RELEASE_WIDTH = 0.05  # open only to 5 cm when releasing, so fingers don't hit neighbours
MAX_FINGER_WIDTH = 0.08

SAFE_Z = 0.25  # TCP height for moving between objects
MIN_TCP_Z = TABLE_HEIGHT + 0.012  # fingertips reach ~9 mm below the TCP
MAX_REACH = 0.80  # horizontal distance from the base beyond which we don't even try IK
PLACE_GAP = 0.004  # release a block this far above its resting height

TRANSIT_SPEED = 0.30  # m/s along a straight TCP path
APPROACH_SPEED = 0.10  # m/s for the last few centimetres down to an object
JOINT_SPEED = 1.0  # rad/s, cap on the fastest joint
WAYPOINT_SPACING = 0.01  # metres between IK waypoints on a Cartesian path

# Bowl drop spots (offsets from the bowl centre), used with yaw 0 so the fingers close along
# world x. The fingers stick out ~2.8 cm from the block centre along that axis, so blocks can
# only be offset along y without the fingers hitting the bowl walls: two blocks per bowl.
BOWL_SPOTS = [(0.0, -0.03), (0.0, 0.03)]


@dataclass
class PrimitiveResult:
    success: bool
    message: str
    steps: int = 0  # physics steps the primitive took


def smoothstep(s: float) -> float:
    """Quintic ease in/out on [0, 1]: zero velocity and acceleration at both ends."""
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def _as_floats(*values: Any) -> list[float] | None:
    """Convert planner-supplied arguments to finite floats, or None if any isn't one."""
    try:
        floats = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    return floats if all(np.isfinite(floats)) else None


class Robot:
    """The Panda in a `Scene`, driven through motion primitives.

    Blocks are held by friction alone: no constraint attaches them to the hand.

    Args:
        scene: the scene to act in. The robot steps `scene.data` forward in time.
        on_step: called after every physics step, e.g. to record video frames.
    """

    def __init__(self, scene: Scene, on_step: Callable[[], None] | None = None) -> None:
        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self.on_step = on_step
        self.ik = IKSolver(self.model, "tcp", HOME_QPOS)
        self.total_steps = 0

        self._tcp = self.model.site("tcp").id
        self._fingers = {self.model.body("left_finger").id, self.model.body("right_finger").id}
        self._finger_qadr = [self.model.joint(j).qposadr[0] for j in ("finger_joint1", "finger_joint2")]
        self._finger_dadr = [self.model.joint(j).dofadr[0] for j in ("finger_joint1", "finger_joint2")]
        self._objects = {o.name: o for o in scene.objects}
        self._block_ids = {self.model.body(o.name).id: o.name for o in scene.objects if o.kind == "block"}

    # ------------------------------------------------------------------ state

    @property
    def held(self) -> str | None:
        """Name of the block currently in the gripper (touched by both fingers), read from the sim."""
        return self._block_between_fingers()

    def tcp_position(self) -> np.ndarray:
        """Current world position of the tool-centre point."""
        return self.data.site_xpos[self._tcp].copy()

    def gripper_width(self) -> float:
        """Current distance between the fingers, in metres."""
        return float(sum(self.data.qpos[a] for a in self._finger_qadr))

    def get_scene(self) -> dict[str, Any]:
        """JSON-serialisable snapshot of the world: every object's pose and the held block."""
        objects = []
        for obj in self.scene.objects:
            pos = self.data.body(obj.name).xpos
            objects.append(
                {
                    "name": obj.name,
                    "kind": obj.kind,
                    "colour": obj.colour,
                    "position": [round(float(v), 3) for v in pos],
                }
            )
        return {"objects": objects, "held": self.held}

    # ------------------------------------------------------------- primitives

    def move_home(self) -> PrimitiveResult:
        """Move the arm back to the Menagerie home pose (gripper state unchanged)."""
        start = self.total_steps
        q_goal = HOME_CTRL[:ARM_DOF].copy()
        self._follow_joint_path([self.data.ctrl[:ARM_DOF].copy(), q_goal], min_duration=0.0)
        self._settle_arm(q_goal)
        err = float(np.max(np.abs(self.data.qpos[:ARM_DOF] - q_goal)))
        if err > 0.02:
            return PrimitiveResult(False, f"arm stopped {err:.3f} rad short of home; something is blocking it", self._since(start))
        return PrimitiveResult(True, "arm is at home", self._since(start))

    def move_to(self, x: float, y: float, z: float, yaw: float = 0.0) -> PrimitiveResult:
        """Move the TCP in a straight line to (x, y, z), gripper pointing down, rotated by `yaw` rad."""
        start = self.total_steps
        values = _as_floats(x, y, z, yaw)
        if values is None:
            return PrimitiveResult(False, f"invalid target ({x}, {y}, {z}) / yaw {yaw}: expected numbers", 0)
        target, yaw = np.array(values[:3]), values[3]
        problem = self._check_reachable(target)
        if problem:
            return PrimitiveResult(False, problem, 0)
        result = self._move_linear(target, yaw, TRANSIT_SPEED)
        result.steps = self._since(start)
        return result

    def open_gripper(self, width: float = MAX_FINGER_WIDTH) -> PrimitiveResult:
        """Open the fingers to `width` metres (default fully open), releasing any held block."""
        start = self.total_steps
        width = float(np.clip(width, 0.0, MAX_FINGER_WIDTH))
        self._drive_gripper(GRIPPER_OPEN * width / MAX_FINGER_WIDTH)
        return PrimitiveResult(True, f"gripper open ({self.gripper_width() * 100:.1f} cm)", self._since(start))

    def close_gripper(self) -> PrimitiveResult:
        """Close the fingers. Succeeds if a block ends up between them."""
        start = self.total_steps
        self._drive_gripper(GRIPPER_CLOSED)
        block = self._block_between_fingers()
        if block is None:
            return PrimitiveResult(False, "gripper closed on nothing", self._since(start))
        return PrimitiveResult(True, f"gripper closed on {block}", self._since(start))

    def pick(self, obj_name: str) -> PrimitiveResult:
        """Pick up a block: approach from above, descend, close, lift, and verify it came up."""
        start = self.total_steps
        problem = self._check_pickable(obj_name)
        if problem:
            return PrimitiveResult(False, problem, 0)

        pos = self._pos(obj_name)
        yaw = self._grasp_yaw(obj_name)
        self.open_gripper()
        reached = self._transit(pos, yaw)
        if not reached.success:
            return self._fail(f"could not reach {obj_name}: {reached.message}", start)

        closed = self.close_gripper()
        if not closed.success or self.held != obj_name:
            self.open_gripper()
            self._move_linear(self._above(self.tcp_position()), yaw, APPROACH_SPEED)
            what = closed.message if not closed.success else f"grasped {self.held} instead"
            return self._fail(f"failed to grasp {obj_name}: {what}", start)

        self._move_linear(self._above(self.tcp_position()), yaw, APPROACH_SPEED)
        lifted = self._pos(obj_name)[2] - pos[2]
        if self.held != obj_name or lifted < 0.05:
            self.open_gripper()
            return self._fail(f"{obj_name} slipped during lift", start)
        return PrimitiveResult(True, f"picked up {obj_name}", self._since(start))

    def place_on(self, obj_name: str) -> PrimitiveResult:
        """Put the held block on top of another block or into a bowl, release and retreat."""
        start = self.total_steps
        held = self.held
        if held is None:
            return PrimitiveResult(False, "not holding anything; pick a block first", 0)
        if obj_name not in self._objects:
            return PrimitiveResult(False, self._unknown(obj_name), 0)
        if obj_name == held:
            return PrimitiveResult(False, f"cannot place {held} on itself", 0)

        target = self._pos(obj_name)
        if self._objects[obj_name].kind == "block":
            on_top = self._block_on_top_of(obj_name, ignore=held)
            if on_top:
                return PrimitiveResult(False, f"{obj_name} already has {on_top} on top of it", 0)
            goal = np.array([target[0], target[1], target[2] + 2 * BLOCK_HALF])
            yaw = self._grasp_yaw(obj_name)
        else:
            spot = self._free_bowl_spot(obj_name, held)
            if spot is None:
                return PrimitiveResult(False, f"{obj_name} is full (it holds {len(BOWL_SPOTS)} blocks)", 0)
            goal = np.array([spot[0], spot[1], TABLE_HEIGHT + BOWL_WALL_T + BLOCK_HALF])
            yaw = 0.0

        result = self._place(held, goal, yaw, start)
        if not result.success:
            return result
        final = self._pos(held)
        if self._objects[obj_name].kind == "block":
            ok = np.linalg.norm(final[:2] - target[:2]) < BLOCK_HALF and abs(final[2] - goal[2]) < 0.01
            where = f"on {obj_name}"
        else:
            offset = np.abs(final[:2] - target[:2])
            ok = bool(np.all(offset < BOWL_HALF)) and final[2] < TABLE_HEIGHT + BOWL_WALL_H + BLOCK_HALF
            where = f"in {obj_name}"
        if not ok:
            return self._fail(f"{held} did not stay {where}; it is now at {self._fmt(final)}", start)
        return PrimitiveResult(True, f"placed {held} {where}", self._since(start))

    def place_at(self, x: float, y: float) -> PrimitiveResult:
        """Put the held block down on the table at (x, y), release and retreat."""
        start = self.total_steps
        held = self.held
        if held is None:
            return PrimitiveResult(False, "not holding anything; pick a block first", 0)
        values = _as_floats(x, y)
        if values is None:
            return PrimitiveResult(False, f"invalid position ({x}, {y}): expected numbers", 0)
        x, y = values
        margin = BLOCK_HALF
        if not (TABLE_X[0] + margin <= x <= TABLE_X[1] - margin and TABLE_Y[0] + margin <= y <= TABLE_Y[1] - margin):
            return PrimitiveResult(False, f"({x:.3f}, {y:.3f}) is off the table", 0)
        goal = np.array([x, y, TABLE_HEIGHT + BLOCK_HALF])
        problem = self._check_reachable(goal)
        if problem:
            return PrimitiveResult(False, problem, 0)
        for obj in self.scene.objects:
            if obj.name == held:
                continue
            gap = np.linalg.norm(self._pos(obj.name)[:2] - goal[:2])
            if gap < footprint_radius(obj.kind) + footprint_radius("block"):
                return PrimitiveResult(False, f"({x:.3f}, {y:.3f}) is too close to {obj.name}", 0)

        result = self._place(held, goal, 0.0, start)
        if not result.success:
            return result
        final = self._pos(held)
        if np.linalg.norm(final[:2] - goal[:2]) > 0.015 or abs(final[2] - goal[2]) > 0.005:
            return self._fail(f"{held} did not land at ({x:.3f}, {y:.3f}); it is at {self._fmt(final)}", start)
        return PrimitiveResult(True, f"placed {held} at ({x:.3f}, {y:.3f})", self._since(start))

    # ------------------------------------------------------- place / transit

    def _place(self, held: str, goal: np.ndarray, yaw: float, start: int) -> PrimitiveResult:
        """Carry `held` so its centre ends at `goal` (+ a small gap), release and retreat."""
        # Aim the TCP so the block (not the TCP) lands on the goal: the block is rarely
        # gripped exactly at its centre.
        grip_offset = self._pos(held) - self.tcp_position()
        tcp_goal = goal - grip_offset + np.array([0.0, 0.0, PLACE_GAP])
        moved = self._transit(tcp_goal, yaw)
        if not moved.success:
            return self._fail(f"could not carry {held} there: {moved.message}", start)
        if self.held != held:
            return self._fail(f"{held} was dropped on the way; it is at {self._fmt(self._pos(held))}", start)
        self.open_gripper(RELEASE_WIDTH)
        self._move_linear(self._above(self.tcp_position()), yaw, APPROACH_SPEED)
        self._step_for(0.2)  # let the block settle before checking where it ended up
        return PrimitiveResult(True, "", self._since(start))

    def _transit(self, goal: np.ndarray, yaw: float) -> PrimitiveResult:
        """Rise to a safe height, travel above `goal`, then descend onto it."""
        safe = self._safe_z()
        tcp = self.tcp_position()
        legs = []
        if tcp[2] < safe - 0.005:
            legs.append((np.array([tcp[0], tcp[1], safe]), self._current_yaw(), APPROACH_SPEED))
        legs.append((np.array([goal[0], goal[1], max(safe, goal[2])]), yaw, TRANSIT_SPEED))
        legs.append((goal, yaw, APPROACH_SPEED))
        for target, leg_yaw, speed in legs:
            problem = self._check_reachable(target)
            if problem:
                return PrimitiveResult(False, problem)
            result = self._move_linear(target, leg_yaw, speed)
            if not result.success:
                return result
        return PrimitiveResult(True, "arrived")

    # ----------------------------------------------------------- arm motion

    def _move_linear(self, target: np.ndarray, yaw: float, speed: float) -> PrimitiveResult:
        """Move the TCP along a straight line to `target`, blending yaw on the way."""
        yaw = normalise_yaw(yaw)
        q_start = self.data.ctrl[:ARM_DOF].copy()  # plan from the commanded pose: no jump
        p_start, _ = self.ik.forward(q_start)
        yaw_start = self._current_yaw()
        d_yaw = normalise_yaw(yaw - yaw_start)
        length = float(np.linalg.norm(target - p_start))
        n = max(1, int(np.ceil(length / WAYPOINT_SPACING)), int(np.ceil(abs(d_yaw) / 0.05)))

        waypoints = [q_start]
        q = q_start
        for i in range(1, n + 1):
            s = i / n
            sol = self.ik.solve(p_start + s * (target - p_start), top_down_rotation(yaw_start + s * d_yaw), q)
            if not sol.success:
                where = "target" if i == n else "path to the target"
                return PrimitiveResult(False, f"{where} {self._fmt(target)} is out of reach")
            q = sol.qpos
            waypoints.append(q)

        self._follow_joint_path(waypoints, length / speed)
        self._settle_arm(waypoints[-1])
        err = float(np.linalg.norm(self.tcp_position() - target))
        if err > 0.01:
            return PrimitiveResult(False, f"arm was blocked {err * 100:.1f} cm short of {self._fmt(target)}")
        return PrimitiveResult(True, f"TCP at {self._fmt(self.tcp_position())}")

    def _follow_joint_path(self, waypoints: list[np.ndarray], min_duration: float) -> None:
        """Sweep actuator targets through joint waypoints with a smoothstep time profile.

        The motion lasts at least `min_duration` seconds, and long enough that no joint
        exceeds JOINT_SPEED on average.
        """
        path = np.array(waypoints)
        joint_travel = float(np.max(np.sum(np.abs(np.diff(path, axis=0)), axis=0))) if len(path) > 1 else 0.0
        duration = max(min_duration, joint_travel / JOINT_SPEED, 0.2)
        n_steps = max(1, int(duration / self.model.opt.timestep))
        segments = len(path) - 1
        for k in range(1, n_steps + 1):
            u = smoothstep(k / n_steps) * segments  # position along the path, in segments
            i = min(int(u), segments - 1)
            frac = u - i
            self.data.ctrl[:ARM_DOF] = (1 - frac) * path[i] + frac * path[i + 1]
            self._step()

    def _settle_arm(self, q_goal: np.ndarray, timeout: float = 1.5) -> None:
        """Hold `q_goal` and step until the joints stop moving (or the timeout runs out)."""
        self.data.ctrl[:ARM_DOF] = q_goal
        for _ in range(int(timeout / self.model.opt.timestep)):
            self._step()
            if np.max(np.abs(self.data.qvel[:ARM_DOF])) < 0.01 and np.max(np.abs(self.data.qpos[:ARM_DOF] - q_goal)) < 0.005:
                return

    # -------------------------------------------------------------- gripper

    def _drive_gripper(self, ctrl: float, ramp: float = 0.25, timeout: float = 1.0) -> None:
        """Ramp the gripper target to `ctrl`, then wait for the fingers to stop."""
        begin = float(self.data.ctrl[ARM_DOF])
        n = int(ramp / self.model.opt.timestep)
        for k in range(1, n + 1):
            self.data.ctrl[ARM_DOF] = begin + (ctrl - begin) * smoothstep(k / n)
            self._step()
        still = 0
        for _ in range(int(timeout / self.model.opt.timestep)):
            self._step()
            moving = max(abs(self.data.qvel[a]) for a in self._finger_dadr) > 1e-3
            still = 0 if moving else still + 1
            if still >= 20:
                return

    def _block_between_fingers(self) -> str | None:
        """A block touched by both fingers at once, if any."""
        touching: dict[str, set[int]] = {}
        for c in self.data.contact[: self.data.ncon]:
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            for finger, other in ((b1, b2), (b2, b1)):
                if finger in self._fingers and other in self._block_ids:
                    touching.setdefault(self._block_ids[other], set()).add(finger)
        for name, fingers in touching.items():
            if fingers == self._fingers:
                return name
        return None

    # -------------------------------------------------------------- helpers

    def _step(self) -> None:
        mujoco.mj_step(self.model, self.data)
        self.total_steps += 1
        if self.on_step is not None:
            self.on_step()

    def _step_for(self, seconds: float) -> None:
        for _ in range(int(seconds / self.model.opt.timestep)):
            self._step()

    def _since(self, start: int) -> int:
        return self.total_steps - start

    def _fail(self, message: str, start: int) -> PrimitiveResult:
        return PrimitiveResult(False, message, self._since(start))

    def _pos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    @staticmethod
    def _fmt(p: np.ndarray) -> str:
        return "(" + ", ".join(f"{v:.3f}" for v in p) + ")"

    def _unknown(self, name: str) -> str:
        return f"unknown object '{name}'; known objects: {', '.join(self._objects)}"

    def _above(self, p: np.ndarray) -> np.ndarray:
        return np.array([p[0], p[1], self._safe_z()])

    def _safe_z(self) -> float:
        """Travel height: SAFE_Z, or higher if a tall stack would be in the way."""
        tops = [self._pos(b)[2] + BLOCK_HALF for b in self._block_ids.values() if b != self.held]
        return max([SAFE_Z] + [t + 0.12 for t in tops])

    def _current_yaw(self) -> float:
        """Yaw of the commanded hand pose (the fingers' closing axis, from world x)."""
        _, rot = self.ik.forward(self.data.ctrl[:ARM_DOF])
        closing_axis = rot[:, 1]
        return normalise_yaw(float(np.arctan2(closing_axis[1], closing_axis[0])))

    def _block_yaw(self, name: str) -> float:
        rot = self.data.body(name).xmat.reshape(3, 3)
        return float(np.arctan2(rot[1, 0], rot[0, 0]))

    def _grasp_yaw(self, name: str) -> float:
        """Gripper yaw lined up with a block's faces; a cube repeats every 90 degrees."""
        return (self._block_yaw(name) + np.pi / 4) % (np.pi / 2) - np.pi / 4

    def _check_reachable(self, target: np.ndarray) -> str | None:
        if not np.all(np.isfinite(target)):
            return f"invalid target {target.tolist()}"
        if target[2] < MIN_TCP_Z:
            return f"target {self._fmt(target)} is below the table surface"
        dist = float(np.hypot(target[0], target[1]))
        out_of_reach = f"target {self._fmt(target)} is out of reach ({dist:.2f} m from the robot base)"
        if dist > MAX_REACH:
            return out_of_reach
        sol = self.ik.solve(target, top_down_rotation(0.0), self.data.ctrl[:ARM_DOF])
        if not sol.success:
            return out_of_reach
        return None

    def _check_pickable(self, name: str) -> str | None:
        if name not in self._objects:
            return self._unknown(name)
        if self._objects[name].kind != "block":
            return f"{name} is a bowl; only blocks can be picked"
        held = self.held
        if held == name:
            return f"already holding {name}"
        if held is not None:
            return f"already holding {held}; place it before picking {name}"
        on_top = self._block_on_top_of(name)
        if on_top:
            return f"{name} is under {on_top}; move {on_top} first"
        return None

    def _block_on_top_of(self, name: str, ignore: str | None = None) -> str | None:
        base = self._pos(name)
        for other in self._block_ids.values():
            if other in (name, ignore):
                continue
            p = self._pos(other)
            if np.linalg.norm(p[:2] - base[:2]) < 2 * BLOCK_HALF and p[2] > base[2] + BLOCK_HALF:
                return other
        return None

    def _free_bowl_spot(self, bowl: str, held: str) -> np.ndarray | None:
        centre = self._pos(bowl)
        inside = [
            self._pos(b)[:2]
            for b in self._block_ids.values()
            if b != held and np.all(np.abs(self._pos(b)[:2] - centre[:2]) < BOWL_HALF)
        ]
        for dx, dy in BOWL_SPOTS:
            spot = centre[:2] + np.array([dx, dy])
            if all(np.linalg.norm(spot - p) > 0.05 for p in inside):
                return spot
        return None
