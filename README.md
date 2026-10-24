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
uv run pytest                                                        # scene tests
```

Every scene is built from a seed: the same seed always gives the same objects, colours and positions.

## License

MIT
