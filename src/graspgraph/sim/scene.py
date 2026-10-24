"""Seeded tabletop scene: Franka Panda, a table, coloured blocks and bowls.

Every scene is built from a seed, so the same seed always gives the same
objects, colours and positions. That is what makes benchmark runs reproducible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

# Colour name -> RGBA. Names are what the planner and instructions refer to.
COLOURS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.85, 0.15, 0.15, 1.0),
    "green": (0.15, 0.70, 0.25, 1.0),
    "blue": (0.15, 0.35, 0.85, 1.0),
    "yellow": (0.95, 0.85, 0.15, 1.0),
    "orange": (0.95, 0.50, 0.10, 1.0),
    "purple": (0.55, 0.25, 0.75, 1.0),
}

# Panda "home" pose from Menagerie: 7 arm joints + 2 finger joints.
HOME_QPOS = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
HOME_CTRL = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255])

TABLE_HEIGHT = 0.02  # thin table slab sitting on the floor
BLOCK_HALF = 0.02  # 4 cm cubes; the gripper opens to 8 cm
BOWL_HALF = 0.06  # 12 cm square bowl (a shallow tray)
BOWL_WALL_H = 0.03
BOWL_WALL_T = 0.005

# Region in front of the robot where objects are placed (metres, robot frame).
WORKSPACE_X = (0.35, 0.65)
WORKSPACE_Y = (-0.30, 0.30)


def menagerie_dir() -> Path:
    """Where mujoco_menagerie is cloned. Override with GRASPGRAPH_MENAGERIE."""
    env = os.environ.get("GRASPGRAPH_MENAGERIE")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "third_party" / "mujoco_menagerie"


@dataclass
class SceneConfig:
    seed: int = 0
    n_blocks: int = 3
    n_bowls: int = 2
    clearance: float = 0.03  # minimum free gap between object footprints, so the gripper fits


@dataclass
class SceneObject:
    name: str  # body name in MuJoCo, e.g. "block_red"
    kind: str  # "block" or "bowl"
    colour: str
    xy: tuple[float, float]


@dataclass
class Scene:
    model: mujoco.MjModel
    data: mujoco.MjData
    config: SceneConfig
    objects: list[SceneObject] = field(default_factory=list)

    def object_poses(self) -> dict[str, np.ndarray]:
        """Current world position (x, y, z) of every object, read from the sim."""
        return {o.name: self.data.body(o.name).xpos.copy() for o in self.objects}


def footprint_radius(kind: str) -> float:
    """Radius of a circle that covers the object's square footprint."""
    half = BOWL_HALF + BOWL_WALL_T if kind == "bowl" else BLOCK_HALF
    return half * np.sqrt(2)


def _sample_positions(
    rng: np.random.Generator, kinds: list[str], clearance: float
) -> list[tuple[float, float]]:
    """Place objects (largest first is the caller's job) so footprints never overlap."""
    positions: list[tuple[float, float]] = []
    radii = [footprint_radius(k) for k in kinds]
    for _ in range(20_000):
        i = len(positions)
        if i == len(kinds):
            return positions
        p = (rng.uniform(*WORKSPACE_X), rng.uniform(*WORKSPACE_Y))
        if all(
            np.hypot(p[0] - q[0], p[1] - q[1]) >= radii[i] + radii[j] + clearance
            for j, q in enumerate(positions)
        ):
            positions.append(p)
    raise RuntimeError(f"Could not place {len(kinds)} objects without overlap; reduce counts or clearance")


def _add_table(spec: mujoco.MjSpec) -> None:
    spec.worldbody.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.35, 0.45, TABLE_HEIGHT / 2],
        pos=[0.5, 0.0, TABLE_HEIGHT / 2],
        rgba=[0.55, 0.42, 0.30, 1.0],
    )


def _add_block(spec: mujoco.MjSpec, name: str, colour: str, xy: tuple[float, float]) -> None:
    z = TABLE_HEIGHT + BLOCK_HALF
    body = spec.worldbody.add_body(name=name, pos=[xy[0], xy[1], z])
    body.add_freejoint()
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[BLOCK_HALF] * 3,
        rgba=list(COLOURS[colour]),
        mass=0.05,
        friction=[1.5, 0.01, 0.001],  # grippy enough to pick up
    )


def _add_bowl(spec: mujoco.MjSpec, name: str, colour: str, xy: tuple[float, float]) -> None:
    # A bowl is a fixed tray: a base plus four walls. Fixed so the arm can't knock it away.
    body = spec.worldbody.add_body(name=name, pos=[xy[0], xy[1], TABLE_HEIGHT])
    rgba = list(COLOURS[colour])
    t, h, r = BOWL_WALL_T, BOWL_WALL_H, BOWL_HALF
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[r, r, t / 2], pos=[0, 0, t / 2], rgba=rgba)
    walls = [  # (x, y, half-size x, half-size y)
        (r, 0, t / 2, r),
        (-r, 0, t / 2, r),
        (0, r, r, t / 2),
        (0, -r, r, t / 2),
    ]
    for dx, dy, hx, hy in walls:
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, h / 2], pos=[dx, dy, h / 2], rgba=rgba)


def build_scene(config: SceneConfig | None = None) -> Scene:
    """Build a seeded scene and return it with the arm at its home pose."""
    config = config or SceneConfig()
    if config.n_blocks > len(COLOURS) or config.n_bowls > len(COLOURS):
        raise ValueError(f"At most {len(COLOURS)} blocks and {len(COLOURS)} bowls (one per colour)")

    scene_xml = menagerie_dir() / "franka_emika_panda" / "scene.xml"
    if not scene_xml.exists():
        raise FileNotFoundError(
            f"Menagerie not found at {scene_xml.parent}. Clone it into third_party/ "
            "or set GRASPGRAPH_MENAGERIE (see README)."
        )
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    # The Menagerie keyframe only covers the arm; it would break once blocks add free joints.
    for key in list(spec.keys):
        spec.delete(key)

    rng = np.random.default_rng(config.seed)
    colour_names = list(COLOURS)
    block_colours = rng.choice(colour_names, size=config.n_blocks, replace=False)
    bowl_colours = rng.choice(colour_names, size=config.n_bowls, replace=False)
    kinds = ["bowl"] * config.n_bowls + ["block"] * config.n_blocks  # bowls first: hardest to fit
    positions = _sample_positions(rng, kinds, config.clearance)

    _add_table(spec)
    objects: list[SceneObject] = []
    for colour, xy in zip(bowl_colours, positions[: config.n_bowls]):
        name = f"bowl_{colour}"
        _add_bowl(spec, name, str(colour), xy)
        objects.append(SceneObject(name, "bowl", str(colour), xy))
    for colour, xy in zip(block_colours, positions[config.n_bowls :]):
        name = f"block_{colour}"
        _add_block(spec, name, str(colour), xy)
        objects.append(SceneObject(name, "block", str(colour), xy))

    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[: len(HOME_QPOS)] = HOME_QPOS
    data.ctrl[:] = HOME_CTRL
    mujoco.mj_forward(model, data)
    return Scene(model=model, data=data, config=config, objects=objects)
