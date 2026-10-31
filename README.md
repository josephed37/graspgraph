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
uv run pytest                                                        # all fast tests (~70 s)
uv run pytest -m slow                                                # full oracle check of all 110 tasks (~70 s)
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

## Task suite

`graspgraph.tasks` generates the benchmark: 10 families x 11 seeds = 110 tasks. A task id such as `stack-007` always rebuilds the identical scene, instruction, success predicate and oracle plan. Phrasing varies between a few templates per family, chosen by seed. List them with `uv run python scripts/list_tasks.py`.

| family | example instruction | success when |
|---|---|---|
| `pick_place` | "Put the red block in the blue bowl." | the block rests inside that bowl's footprint, low in the bowl |
| `stack` | "Stack the green block on the yellow one." | the top block is within 2 cm (xy) of the lower one, at the right height |
| `tower` | "Stack the blocks into a tower, bottom to top: red, blue, green." | all three are stacked in that order, the bottom one on the table |
| `sort` | "Put each block in the bowl of the same colour." | every block with a same-coloured bowl is in it (scenes always have at least two matches) |
| `spatial` | "Move the left-most block to the front right corner of the table." | the referred block rests on the table inside the named region |
| `arrange` | "Line the blocks up from left to right: red, green, blue." | the blocks are ordered along y as stated, at least 5 cm apart, with x spread at most 4 cm |
| `count` | "Put two blocks in each bowl." | each bowl holds exactly the stated number of blocks |
| `ambiguous` | "Move the block into the bowl." (several blocks) | the planner calls `ask_clarification` and nothing moved |
| `impossible` | "Put the purple block in the bowl." (no purple block) | the planner calls `refuse` and nothing moved |
| `undo` | "Undo the last move." (a setup move happened first; its history is in the prompt) | the moved block is back within 2 cm of where it was and nothing else moved |

How success is checked:
- Predicates (`tasks/predicates.py`) read only the final simulator state (`get_scene()`), never anything the model says. All tolerances are named constants in that file.
- Directions are from the robot's point of view: front is away from the robot (+x) and left is +y. The named regions are defined once in `tasks/regions.py` and are given to the planner in its prompt.
- No "act" task is already solved when the planner starts.
- Every task ships an oracle plan. `uv run python scripts/validate_oracles.py` runs all 110 through the real primitives, which proves the benchmark is solvable: **110/110**.

## Running the benchmark

The open-loop planner (`graspgraph.planner`) is a LangGraph graph: build_prompt -> plan -> validate -> execute.
- **Plan**: one model call returns the whole plan.
- **Validate**: checks every step before anything moves. A plan with any bad step is rejected whole.
- **Execute**: runs the steps in order without checking or replanning, stopping at `ask_clarification` or `refuse`.

Object arguments are enums of the scene's object names. Prompts live in `planner/prompts.py`, and each protocol's prompt version is written into every record.

Plan protocols (`--plan-protocol`):
- **`submit_plan`** (default): the model makes exactly one call, `submit_plan(steps=[{"tool": "pick", "args": {"obj_name": "block_red"}}, ...])`, or a single `ask_clarification` / `refuse` call. Many models, Gemma 4 included, make only one tool call per turn, so the whole plan has to fit in one call.
- **`multi_call`**: the original protocol, where each step is its own tool call, kept for comparison. Gemma 4 returned about one call per task under it, so multi-step tasks stopped after the first primitive.

```bash
# Local Gemma via Ollama (free). Pull the model first: ollama pull gemma4:12b
uv run graspgraph-bench --backend ollama --model gemma4:12b --mode open --families all \
    --no-think --out results/gemma4-12b_open_submit_plan.jsonl
uv run graspgraph-bench --backend ollama --model gemma4:12b --think --out results/gemma4-12b_open_submit_plan_think.jsonl
uv run graspgraph-bench --backend ollama --model gemma4:12b --plan-protocol multi_call   # the original protocol

# Claude (needs ANTHROPIC_API_KEY); --max-usd is the spend cap, default 1.00
export ANTHROPIC_API_KEY=...
uv run graspgraph-bench --backend anthropic --model claude-haiku-5-5 --max-usd 1.00 --out results/haiku-5-5_open_submit_plan.jsonl
uv run graspgraph-bench --backend anthropic --model claude-sonnet-5-5 --limit 50 --max-usd 2.00

# No model at all: scripted plans (oracle, wrong-object, no-tools, first-step) for dry runs
uv run graspgraph-bench --backend fake --model oracle

uv run python scripts/summarize.py results/*.jsonl
```

Other options: `--plan-protocol submit_plan|multi_call`, `--families stack tower`, `--task-ids stack-007 undo-003`, `--limit N`, `--repeats N`, `--max-tokens` (output cap; defaults: Anthropic 1024, Ollama 2048, or 8192 with `--think` because thinking counts against it), `--temperature` (default: the provider's own, so repeats measure variance), `--ollama-url`.

- **Output**: one JSON line per task run, with success, `outcome_type`, `plan_protocol`, `planned_steps`, `raw_plan` (exactly what the model submitted), `plan_error`, `finish_reason` (why the model stopped, e.g. `length` at the token cap), the tool calls, every primitive result, tokens, `cost_usd` and model latency. The default file name includes the protocol.
- **Outcome types** (closed loop adds `replan_budget_exhausted`):
  - **Success**: `success`, `asked_correctly`, `refused_correctly`.
  - **Failures on "act" tasks**:
    - `wrong_object`: acted on an object the instruction did not refer to.
    - `primitive_failed`: a primitive refused or failed.
    - `incomplete_plan`: used only referred objects, but stopped before the task was done (left one out, placed too few blocks, or ended holding a block).
    - `bad_placement`: anything else.
  - **Plan problems**: `invalid_plan` (a malformed step: unknown tool, missing or bad arguments; nothing runs), `hallucinated_object` (an object name not in the scene), `no_tool_calls`.
  - **Ask and refuse mistakes**: `should_have_asked`, `should_have_refused`, `should_have_acted`, `acted_when_should_not`.
  - **Model failures**: `degenerate_output` (the model got stuck repeating itself, or hit the output cap without producing a usable plan).
  - **Model call failed**: `llm_error` (network or API trouble).
- **Resume**: rerunning with the same `--out` skips runs already recorded with the same model, protocol and prompt version, so a crashed run continues where it stopped.
  - Only `llm_error` runs are retried. Model failures such as `degenerate_output` are not, so no task gets extra attempts.
  - `summarize.py` prints one table per configuration (model, protocol, prompt version) found in a file.
- **Spend cap**: before each run, the money already spent in the file plus the most expensive run so far is compared with `--max-usd`, and the run stops cleanly before crossing it. Prices are in `src/graspgraph/bench/prices.toml`; Ollama costs $0.
- **Debugging empty responses**: `uv run python scripts/debug_ollama.py --task-ids tower-001 --repeats 3` sends a task's exact request to Ollama and prints the raw reply (text, thinking, tool calls, `done_reason`), with and without streaming. Raw chunks are saved to `outputs/ollama_debug/`.
- **LangSmith tracing** is on only when both `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are set (optionally `LANGSMITH_PROJECT`). Otherwise nothing is sent anywhere.

## Closed loop and fault injection

`--mode closed` runs the planner as build_prompt -> plan -> validate -> execute_step -> verify, one step at a time, with a replan branch:
- **Verify:** after every step, its expected effect is checked against the scene (holding the block; the block resting in the bowl, on the block or at the spot). When the plan finishes, every placement still in force is checked again, which catches a block moved after it was placed. The checks only describe the planner's own steps; the task's success predicate stays hidden.
- **Replan:** a failed step, an undone effect or a rejected plan goes back to the model as the answer to its `submit_plan` call. The report carries the failed step, the reason, the steps done and not done, and the current scene, and the model submits a new plan from there.
- **Budgets:** `--max-replans` (default 3) and `--max-steps` (24 executed steps) stop loops. Running out scores as `replan_budget_exhausted`.

`--fault-rate 0.3` injects a seeded fault into about 30% of the act tasks, the same ones in open and closed mode:
- **`drop`:** the block slips out of the gripper early in the carry.
- **`displace`:** a block just put in a bowl or on the table is moved elsewhere before the next step.

Records gain `fault`, `recovered`, `llm_calls`, `replans` and every submitted plan, and the summary shows the recovery rate.

```bash
uv run graspgraph-bench --backend ollama --model gemma4:12b --no-think --mode closed --fault-rate 0.3
uv run graspgraph-bench --backend ollama --model gemma4:12b --no-think --fault-rate 0.3   # open loop, same faults
```

With the scripted oracle (`--backend fake`), the 27 triggered faults are recovered 0/27 in open loop and 27/27 in closed loop.

## License

MIT
