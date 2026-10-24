# graspgraph

LLM-planned robot arm in MuJoCo: a LangGraph planner turns plain-language instructions into motion primitives, recovers from failures, and is benchmarked across 110 tasks (Gemma 4 vs Claude).

> Work in progress. Progress is tracked in [issue #1](https://github.com/josephed37/graspgraph/issues/1).

## Setup

Requires [uv](https://docs.astral.sh/uv/), Python 3.11+ and Linux, macOS or Windows with OpenGL for the viewer.

```bash
git clone https://github.com/josephed37/graspgraph.git
cd graspgraph
uv sync          # creates .venv, installs graspgraph + dev tools from uv.lock

# Robot models (Franka Panda) from MuJoCo Menagerie
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
```

Already have Menagerie elsewhere? Point to it instead: `export GRASPGRAPH_MENAGERIE=/path/to/mujoco_menagerie`.

## Try it

```bash
uv run python scripts/view_scene.py --seed 3 --blocks 4 --bowls 2   # opens the 3D viewer
uv run pytest                                                        # scene, IK and primitive tests (~20 s)
```

Every scene is built from a seed: the same seed always gives the same objects, colours and positions.

## Motion primitives

`graspgraph.sim.robot.Robot` wraps a scene and exposes the tools the planner will call. Every primitive returns a `PrimitiveResult(success, message, steps)` and never raises on an ordinary failure: the message says what went wrong ("block_red slipped during lift", "bowl_red is full", "target ... is out of reach") so a planner can replan.

```python
from graspgraph.sim import SceneConfig, build_scene
from graspgraph.sim.robot import Robot

robot = Robot(build_scene(SceneConfig(seed=3)))
robot.pick("block_red")
robot.place_on("bowl_orange")
robot.get_scene()   # {"objects": [{"name", "kind", "colour", "position"}, ...], "held": None}
```

| Primitive | What it does |
|---|---|
| `move_home()` | Joint-space move back to the home pose. |
| `move_to(x, y, z, yaw=0.0)` | Straight-line TCP move with the gripper pointing down, rotated by `yaw` radians. |
| `open_gripper()` / `close_gripper()` | Open fully / close; closing succeeds only if a block ends up between the fingers. |
| `pick(obj)` | Rise to a safe height, travel above the block, descend, close, lift, then check in the simulator that the block came up with the gripper. |
| `place_on(obj)` | Put the held block on top of another block or into a bowl (two spots per bowl), release, retreat, and check where the block ended up. |
| `place_at(x, y)` | Same, onto a free spot on the table. |
| `get_scene()` | JSON-ready dict of every object's position (rounded to mm) and the held block (or `None`). |

Requests that can't work are refused without moving: unknown objects, picking a bowl or a block with another on top of it, picking while holding, placing with an empty gripper, and targets off the table or out of reach.

How it works:
- **TCP**: a `tcp` site on the hand, midway between the fingertip pads (offset taken from Menagerie's `panda.xml`; a test checks it).
- **IK**: damped least squares on `mj_jacSite` for position plus a fixed top-down orientation, with the spare 7th joint pulled towards the home posture and joint limits enforced (`graspgraph/sim/ik.py`).
- **Motion**: Cartesian paths are split into 1 cm IK waypoints, and actuator targets are swept along them with a smooth (quintic) time profile while the physics runs, then held until the arm settles. Nothing is teleported.

**Grasp mode: real friction.** Blocks are held only by finger contact; there is no weld or "magic" attachment. Out of the box the Menagerie gripper squeezes a 4 cm block with only ~2 N and blocks crept out of the fingers while carried (14/30 in a quick trial). Two changes to the scene fixed that: a stiffer gripper servo (1000 N/m, so ~20 N of squeeze, under the real Panda's 70 N) and gravity compensation on the arm, as the real Panda controller does. With those, 500 of 500 tasks succeeded over seeds 0–99 (all blocks into bowls plus a three-block tower). Because friction works, no weld mode was added.

Record a demo (pick red into a bowl, stack two blocks, go home) without a display:

```bash
uv run python scripts/demo_primitives.py            # outputs/demo_primitives.mp4
```

## License

MIT
