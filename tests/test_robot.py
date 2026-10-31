import json

import mujoco
import numpy as np
import pytest

from graspgraph.sim import SceneConfig, build_scene
from graspgraph.sim.ik import IKSolver, rotation_error, top_down_rotation
from graspgraph.sim.robot import Robot
from graspgraph.sim.scene import BLOCK_HALF, BOWL_HALF, BOWL_WALL_T, HOME_QPOS, TABLE_HEIGHT

from conftest import SUCCESS_RATE_KEY

N_TASKS = 20
REQUIRED_SUCCESS_RATE = 0.95


def _names(scene, kind):
    return [o.name for o in scene.objects if o.kind == kind]


# --------------------------------------------------------------------- TCP and IK


def test_tcp_site_sits_between_fingertip_pads():
    scene = build_scene(SceneConfig(seed=0))
    m, d = scene.model, scene.data
    fingers = {m.body("left_finger").id, m.body("right_finger").id}
    # The main pad of each finger is its largest box geom (fingertip_pad_collision_1).
    pads = [
        g
        for g in range(m.ngeom)
        if m.geom_bodyid[g] in fingers and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and m.geom_size[g][0] > 0.008
    ]
    assert len(pads) == 2
    for finger_q in (0.04, 0.02, 0.0):  # open, holding a block, closed
        d.qpos[7:9] = finger_q
        mujoco.mj_forward(m, d)
        midpoint = d.geom_xpos[pads].mean(axis=0)
        assert np.allclose(d.site("tcp").xpos, midpoint, atol=1e-4)


def test_ik_reaches_random_targets():
    scene = build_scene(SceneConfig(seed=0))
    ik = IKSolver(scene.model, "tcp", HOME_QPOS)
    rng = np.random.default_rng(42)
    lower, upper = scene.model.jnt_range[:7].T
    for _ in range(20):
        target = np.array([rng.uniform(0.3, 0.7), rng.uniform(-0.35, 0.35), rng.uniform(0.04, 0.4)])
        yaw = rng.uniform(-np.pi / 2, np.pi / 2)
        result = ik.solve(target, top_down_rotation(yaw), HOME_QPOS)
        assert result.success
        # Check independently with forward kinematics, not the solver's own error.
        pos, rot = ik.forward(result.qpos)
        assert np.linalg.norm(pos - target) < 0.005
        assert np.linalg.norm(rotation_error(top_down_rotation(yaw), rot)) < 0.01
        assert np.all(result.qpos >= lower) and np.all(result.qpos <= upper)


def test_move_to_is_accurate_and_smooth():
    scene = build_scene(SceneConfig(seed=0))
    joint_jumps = []
    robot = Robot(scene)
    last = scene.data.qpos[:7].copy()

    def track() -> None:
        nonlocal last
        joint_jumps.append(np.max(np.abs(scene.data.qpos[:7] - last)))
        last = scene.data.qpos[:7].copy()

    robot.on_step = track
    result = robot.move_to(0.45, -0.2, 0.15, yaw=0.5)
    assert result.success, result.message
    assert np.linalg.norm(robot.tcp_position() - [0.45, -0.2, 0.15]) < 0.005
    # 0.01 rad per 2 ms step is 5 rad/s: anything above that would be a teleport.
    assert max(joint_jumps) < 0.01
    assert robot.move_home().success


# ------------------------------------------------------------ scripted pick/place


def _bowl_task(seed: int) -> tuple[bool, str]:
    """Pick every block and put it in a bowl; check final poses in the simulator."""
    scene = build_scene(SceneConfig(seed=seed, n_blocks=3, n_bowls=2))
    robot = Robot(scene)
    bowls = _names(scene, "bowl")
    plan = {block: bowls[i % len(bowls)] for i, block in enumerate(_names(scene, "block"))}
    for block, bowl in plan.items():
        for result in (robot.pick(block), robot.place_on(bowl)):
            if not result.success:
                return False, result.message
    robot.move_home()
    for block, bowl in plan.items():
        offset = np.abs(scene.data.body(block).xpos[:2] - scene.data.body(bowl).xpos[:2])
        z = scene.data.body(block).xpos[2]
        if np.any(offset > BOWL_HALF) or abs(z - (TABLE_HEIGHT + BOWL_WALL_T + BLOCK_HALF)) > 0.005:
            return False, f"{block} is not resting in {bowl}"
    return True, "ok"


def _stack_task(seed: int) -> tuple[bool, str]:
    """Stack one block on another; check the final pose in the simulator."""
    scene = build_scene(SceneConfig(seed=seed, n_blocks=3, n_bowls=2))
    robot = Robot(scene)
    top, base = _names(scene, "block")[:2]
    for result in (robot.pick(top), robot.place_on(base)):
        if not result.success:
            return False, result.message
    robot.move_home()
    top_pos, base_pos = scene.data.body(top).xpos, scene.data.body(base).xpos
    if np.linalg.norm(top_pos[:2] - base_pos[:2]) > BLOCK_HALF or abs(top_pos[2] - base_pos[2] - 2 * BLOCK_HALF) > 0.005:
        return False, f"{top} is not resting on {base}"
    return True, "ok"


TASKS = [("bowls", seed, _bowl_task) for seed in range(10)] + [("stack", seed, _stack_task) for seed in range(10, 20)]


@pytest.fixture(scope="module")
def task_outcomes() -> list[tuple[str, int, bool, str]]:
    return [(name, seed, *task(seed)) for name, seed, task in TASKS]


def test_scripted_tasks_success_rate(task_outcomes, request):
    assert len(task_outcomes) == N_TASKS
    successes = sum(ok for _, _, ok, _ in task_outcomes)
    rate = successes / N_TASKS
    request.config.stash[SUCCESS_RATE_KEY] = (successes, N_TASKS)
    failures = [f"{name} seed {seed}: {msg}" for name, seed, ok, msg in task_outcomes if not ok]
    print(f"\nscripted pick-and-place success rate: {successes}/{N_TASKS} ({rate:.0%})")
    assert rate >= REQUIRED_SUCCESS_RATE, "failures:\n" + "\n".join(failures)


# ------------------------------------------------------------------ error cases


@pytest.fixture
def robot():
    return Robot(build_scene(SceneConfig(seed=3, n_blocks=3, n_bowls=2)))


def _assert_fails(result, *phrases):
    assert result.success is False
    assert result.steps == 0  # rejected before moving
    for phrase in phrases:
        assert phrase in result.message


def test_unknown_object_is_rejected(robot):
    _assert_fails(robot.pick("block_pink"), "unknown object 'block_pink'", "known objects")
    robot.pick(_names(robot.scene, "block")[0])
    _assert_fails(robot.place_on("bowl_pink"), "unknown object 'bowl_pink'")


def test_double_pick_is_rejected(robot):
    first, second = _names(robot.scene, "block")[:2]
    assert robot.pick(first).success
    _assert_fails(robot.pick(second), f"already holding {first}")
    _assert_fails(robot.pick(first), f"already holding {first}")
    assert robot.held == first


def test_place_with_empty_gripper_is_rejected(robot):
    _assert_fails(robot.place_on(_names(robot.scene, "bowl")[0]), "not holding anything")
    _assert_fails(robot.place_at(0.5, 0.0), "not holding anything")


def test_unreachable_targets_are_rejected(robot):
    _assert_fails(robot.move_to(1.5, 0.0, 0.2), "out of reach")
    _assert_fails(robot.move_to(0.5, 0.0, -0.05), "below the table")
    _assert_fails(robot.move_to("left", 0.0, 0.2), "expected numbers")
    assert robot.pick(_names(robot.scene, "block")[0]).success
    _assert_fails(robot.place_at(1.2, 0.0), "off the table")
    _assert_fails(robot.place_at(0.25, 0.0), "out of reach")


def test_picking_a_bowl_or_a_covered_block_is_rejected(robot):
    _assert_fails(robot.pick(_names(robot.scene, "bowl")[0]), "only blocks can be picked")
    top, base = _names(robot.scene, "block")[:2]
    assert robot.pick(top).success and robot.place_on(base).success
    _assert_fails(robot.pick(base), f"{base} is under {top}")


def test_get_scene_is_json_and_tracks_held_block(robot):
    block = _names(robot.scene, "block")[0]
    before = robot.get_scene()
    assert before["held"] is None
    assert {o["name"] for o in before["objects"]} == {o.name for o in robot.scene.objects}
    entry = next(o for o in before["objects"] if o["name"] == block)
    assert set(entry) == {"name", "kind", "colour", "position"}
    assert entry["position"][2] == pytest.approx(TABLE_HEIGHT + BLOCK_HALF, abs=0.002)

    assert robot.pick(block).success
    after = json.loads(json.dumps(robot.get_scene()))  # must survive a JSON round trip
    assert after["held"] == block
    lifted = next(o for o in after["objects"] if o["name"] == block)
    assert lifted["position"][2] > entry["position"][2] + 0.1


# ------------------------------------------------- awkward poses (e.g. after a dropped block)


def _put(robot, name, xyz, quat=(1, 0, 0, 0)):
    """Teleport a block to a pose, at rest."""
    joint = robot.model.body(name).jntadr[0]
    qpos, dof = robot.model.jnt_qposadr[joint], robot.model.jnt_dofadr[joint]
    robot.data.qpos[qpos : qpos + 7] = [*xyz, *quat]
    robot.data.qvel[dof : dof + 6] = 0
    mujoco.mj_forward(robot.model, robot.data)


def test_pick_a_block_lying_on_its_side():
    robot = Robot(build_scene(SceneConfig(seed=0)))
    block = _names(robot.scene, "block")[0]
    # On its side (90 degrees about y), then turned 0.5 rad about the vertical.
    on_side = np.empty(4)
    mujoco.mju_mulQuat(on_side, [np.cos(0.25), 0, 0, np.sin(0.25)], [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])
    _put(robot, block, (*robot.data.body(block).xpos[:2], TABLE_HEIGHT + BLOCK_HALF), on_side)
    assert robot._grasp_yaw(block) == pytest.approx(0.5, abs=1e-6)  # read from a horizontal edge
    result = robot.pick(block)
    assert result.success, result.message


@pytest.mark.parametrize("offset", [(0.017, 0.0), (0.0, -0.017)])
def test_pick_a_block_off_centre_in_a_bowl(offset):
    """The open fingers would hit a wall along one axis: the gripper turns to the other faces."""
    robot = Robot(build_scene(SceneConfig(seed=0)))
    block, bowl = _names(robot.scene, "block")[0], _names(robot.scene, "bowl")[0]
    centre = robot.data.body(bowl).xpos
    _put(robot, block, (centre[0] + offset[0], centre[1] + offset[1], TABLE_HEIGHT + BOWL_WALL_T + BLOCK_HALF))
    robot._step_for(0.3)
    result = robot.pick(block)
    assert result.success, result.message


def test_pick_a_block_pressed_against_the_outside_of_a_bowl():
    robot = Robot(build_scene(SceneConfig(seed=0)))
    block, bowl = _names(robot.scene, "block")[0], _names(robot.scene, "bowl")[0]
    centre = robot.data.body(bowl).xpos
    _put(robot, block, (centre[0] - (BOWL_HALF + 0.005 + BLOCK_HALF + 0.002), centre[1], TABLE_HEIGHT + BLOCK_HALF))
    robot._step_for(0.3)
    result = robot.pick(block)
    assert result.success, result.message
