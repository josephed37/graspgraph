import mujoco
import numpy as np
import pytest

from graspgraph.sim import SceneConfig, build_scene
from graspgraph.sim.scene import BLOCK_HALF, TABLE_HEIGHT, WORKSPACE_X, WORKSPACE_Y, footprint_radius


def _settle(scene, seconds: float = 1.0) -> None:
    steps = int(seconds / scene.model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(scene.model, scene.data)


def test_same_seed_gives_same_scene():
    a = build_scene(SceneConfig(seed=7))
    b = build_scene(SceneConfig(seed=7))
    assert [(o.name, o.xy) for o in a.objects] == [(o.name, o.xy) for o in b.objects]


def test_different_seeds_differ():
    a = build_scene(SceneConfig(seed=1))
    b = build_scene(SceneConfig(seed=2))
    assert [(o.name, o.xy) for o in a.objects] != [(o.name, o.xy) for o in b.objects]


@pytest.mark.parametrize("seed", range(5))
def test_objects_inside_workspace_and_spaced(seed):
    config = SceneConfig(seed=seed, n_blocks=4, n_bowls=2)
    scene = build_scene(config)
    assert len(scene.objects) == 6
    xys = np.array([o.xy for o in scene.objects])
    assert np.all((xys[:, 0] >= WORKSPACE_X[0]) & (xys[:, 0] <= WORKSPACE_X[1]))
    assert np.all((xys[:, 1] >= WORKSPACE_Y[0]) & (xys[:, 1] <= WORKSPACE_Y[1]))
    for i, a in enumerate(scene.objects):
        for b in scene.objects[i + 1 :]:
            gap = np.hypot(a.xy[0] - b.xy[0], a.xy[1] - b.xy[1])
            assert gap >= footprint_radius(a.kind) + footprint_radius(b.kind) + config.clearance


def test_unique_colours_per_kind():
    scene = build_scene(SceneConfig(seed=3, n_blocks=5, n_bowls=3))
    for kind in ("block", "bowl"):
        colours = [o.colour for o in scene.objects if o.kind == kind]
        assert len(colours) == len(set(colours))


@pytest.mark.parametrize("seed", range(3))
def test_scene_is_physically_stable(seed):
    """Blocks rest on the table and the arm holds its home pose after settling."""
    scene = build_scene(SceneConfig(seed=seed))
    start = scene.object_poses()
    arm_start = scene.data.qpos[:7].copy()
    _settle(scene, 1.0)
    end = scene.object_poses()
    for obj in scene.objects:
        drift = np.linalg.norm(end[obj.name][:2] - start[obj.name][:2])
        assert drift < 0.005, f"{obj.name} moved {drift:.4f} m while settling"
        if obj.kind == "block":
            assert end[obj.name][2] == pytest.approx(TABLE_HEIGHT + BLOCK_HALF, abs=0.003)
    assert np.allclose(scene.data.qpos[:7], arm_start, atol=0.02)


def test_too_many_objects_rejected():
    with pytest.raises(ValueError):
        build_scene(SceneConfig(n_blocks=7))
