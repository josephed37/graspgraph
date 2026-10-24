"""Damped least-squares inverse kinematics for the Panda's tool-centre point.

The solver works on a scratch copy of the simulation state, so solving never moves the
real arm. Only the 7 arm joints are solved for; the fingers are left alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

ARM_DOF = 7

# Hand orientation when the gripper points straight down at yaw=0 (columns are the hand's
# x, y, z axes in world coordinates). This is the Menagerie home pose: the hand z-axis
# points at the table and the fingers close along world x.
R_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


def top_down_rotation(yaw: float) -> np.ndarray:
    """Gripper pointing straight down, rotated by `yaw` radians about the world z-axis.

    At yaw=0 the fingers close along world x; at yaw=pi/2 they close along world y.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rz @ R_DOWN


def normalise_yaw(yaw: float) -> float:
    """Map a yaw into [-pi/2, pi/2).

    The two-finger gripper is symmetric, so yaw and yaw + pi give the same grasp. Keeping
    yaw in this range keeps joint 7 well inside its +-2.9 rad limit.
    """
    return (yaw + np.pi / 2) % np.pi - np.pi / 2


def rotation_error(r_target: np.ndarray, r_current: np.ndarray) -> np.ndarray:
    """Axis-angle vector (world frame) that rotates `r_current` onto `r_target`."""
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, (r_target @ r_current.T).ravel())
    err = np.empty(3)
    mujoco.mju_quat2Vel(err, quat, 1.0)  # unit-time angular velocity = axis * angle
    return err


@dataclass
class IKResult:
    qpos: np.ndarray  # 7 arm joint angles
    success: bool
    pos_error: float  # metres
    rot_error: float  # radians
    iterations: int


class IKSolver:
    """Solve for arm joint angles that put a site at a position with a given orientation.

    Each iteration linearises the forward kinematics around the current guess q:

        error e = [p_target - p(q);  w * rot_err(q)]     (6-vector)
        J       = d[p; orientation] / dq                 (6 x 7, from mj_jacSite)

    and takes the damped least-squares step

        dq = J^T (J J^T + lambda^2 I)^-1 e

    The damping lambda keeps the step bounded near singularities, where the plain
    pseudo-inverse would blow up. The arm has 7 joints but the task only constrains 6,
    so the leftover freedom (the "elbow") is used to pull gently towards a rest posture:

        dq += (I - pinv(J) J) * k * (q_rest - q)

    (I - pinv(J) J) projects onto the null space of J, so this posture term never disturbs
    the TCP. It uses the exact (undamped) pseudo-inverse: the damped one only approximately
    spans the null space, and the leak stalls convergence about 1 mm short of the target.
    Joint limits are enforced by clipping after every step.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        site: str,
        q_rest: np.ndarray,
        damping: float = 0.05,
        rot_weight: float = 0.5,
        posture_gain: float = 0.1,
        max_step: float = 0.3,
        pos_tol: float = 5e-4,
        rot_tol: float = 5e-3,
        max_iters: int = 200,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)  # scratch state: solving never touches the sim
        self.site_id = model.site(site).id
        self.q_rest = np.asarray(q_rest, dtype=float)[:ARM_DOF]
        self.lower = model.jnt_range[:ARM_DOF, 0].copy()
        self.upper = model.jnt_range[:ARM_DOF, 1].copy()
        self.damping = damping
        self.rot_weight = rot_weight
        self.posture_gain = posture_gain
        self.max_step = max_step
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol
        self.max_iters = max_iters
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Site position and rotation matrix for arm joint angles `q`."""
        self.data.qpos[:ARM_DOF] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # needed by mj_jacSite
        return (
            self.data.site_xpos[self.site_id].copy(),
            self.data.site_xmat[self.site_id].reshape(3, 3).copy(),
        )

    def solve(self, target_pos: np.ndarray, target_rot: np.ndarray, q_init: np.ndarray) -> IKResult:
        """Run damped least squares from `q_init`; falls back to the rest posture if that fails."""
        result = self._solve_from(target_pos, target_rot, np.asarray(q_init, dtype=float)[:ARM_DOF])
        if not result.success:
            retry = self._solve_from(target_pos, target_rot, self.q_rest.copy())
            if retry.pos_error < result.pos_error:
                result = retry
        return result

    def _solve_from(self, target_pos: np.ndarray, target_rot: np.ndarray, q: np.ndarray) -> IKResult:
        q = np.clip(q, self.lower, self.upper)
        eye6 = np.eye(6)
        pos_err = rot_err = np.inf
        for it in range(1, self.max_iters + 1):
            pos, rot = self.forward(q)
            e_pos = target_pos - pos
            e_rot = rotation_error(target_rot, rot)
            pos_err, rot_err = float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))
            if pos_err < self.pos_tol and rot_err < self.rot_tol:
                return IKResult(q, True, pos_err, rot_err, it)

            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
            jac = np.vstack([self._jacp[:, :ARM_DOF], self.rot_weight * self._jacr[:, :ARM_DOF]])
            err = np.concatenate([e_pos, self.rot_weight * e_rot])

            # Damped pseudo-inverse J^+ = J^T (J J^T + lambda^2 I)^-1.
            jac_pinv = jac.T @ np.linalg.inv(jac @ jac.T + self.damping**2 * eye6)
            dq = jac_pinv @ err
            null = np.eye(ARM_DOF) - np.linalg.pinv(jac) @ jac
            dq += null @ (self.posture_gain * (self.q_rest - q))

            # Cap the step so a bad linearisation far from the target can't fling the arm.
            norm = np.linalg.norm(dq)
            if norm > self.max_step:
                dq *= self.max_step / norm
            q = np.clip(q + dq, self.lower, self.upper)
        return IKResult(q, False, pos_err, rot_err, self.max_iters)
