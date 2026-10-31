from graspgraph.actions import ToolCall
from graspgraph.planner.verify import expectation_for, failure_text, where
from graspgraph.sim.robot import GRIPPER_OPEN
from graspgraph.tasks import get_task, start_episode


def _episode():
    episode = start_episode(get_task("pick-place-000"))
    pick, place = episode.task.oracle
    return episode.robot, pick, place, pick.args["obj_name"], place.args["obj_name"]


def _check(robot, step, held_before):
    return expectation_for(step, held_before).mismatch(robot.get_scene(), robot.at_home())


def test_correct_steps_pass():
    robot, pick, place, block, _ = _episode()
    assert robot.pick(block).success and _check(robot, pick, None) is None
    assert robot.place_on(place.args["obj_name"]).success and _check(robot, place, block) is None
    assert robot.move_home().success and _check(robot, ToolCall("move_home"), None) is None


def test_a_drop_is_detected():
    robot, pick, place, block, _ = _episode()
    robot.pick(block)
    robot.data.ctrl[7] = GRIPPER_OPEN  # let go mid-air
    robot._step_for(1.0)
    message = _check(robot, pick, None)
    assert message.startswith(f"Expected {block} to be held after pick({block}), but {block} is on the table at (")
    assert message.endswith("Nothing is held.")


def test_a_wrong_placement_is_detected():
    robot, pick, place, block, bowl = _episode()
    robot.pick(block)
    other = next(o.name for o in robot.scene.objects if o.kind == "bowl" and o.name != bowl)
    robot.place_on(other)  # put it in the other bowl, then check the step that was asked for
    message = _check(robot, place, block)
    assert message == f"Expected {block} to rest in {bowl} after place_on({bowl}), but {where(robot.get_scene(), block)}. Nothing is held."
    assert f"is in {other}" in message


def test_a_failed_primitive_is_a_mismatch_with_its_message():
    robot, _, place, _, bowl = _episode()
    result = robot.place_on(bowl)  # nothing held
    assert not result.success
    assert failure_text(place, result.message, robot.get_scene()) == (
        f"place_on({bowl}) failed: not holding anything; pick a block first. Nothing is held."
    )


def test_place_at_and_move_home_mismatches():
    robot, pick, _, block, _ = _episode()
    robot.pick(block)
    spot = ToolCall("place_at", {"x": 0.45, "y": 0.0})
    assert "is in the gripper" in _check(robot, spot, block)  # not put down yet
    assert _check(robot, ToolCall("move_home"), None).startswith("Expected the arm to be at home")
