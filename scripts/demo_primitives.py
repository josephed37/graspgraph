"""Run a short primitive sequence and record it offscreen to a video.

    uv run python scripts/demo_primitives.py                      # outputs/demo_primitives.mp4
    uv run python scripts/demo_primitives.py --out outputs/demo.gif --width 320 --height 240

Sequence: pick the red block, put it in a bowl, stack the green block on the yellow one,
go home. Uses MUJOCO_GL if set, otherwise EGL with an OSMesa fallback (no display needed).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def choose_gl_backend() -> str:
    """First offscreen OpenGL backend that can create a context (probed in a subprocess,
    because MuJoCo binds the backend once at import time)."""
    for backend in ("egl", "osmesa"):
        probe = subprocess.run(
            [sys.executable, "-c", "import mujoco; mujoco.GLContext(16, 16).make_current()"],
            env={**os.environ, "MUJOCO_GL": backend},
            capture_output=True,
        )
        if probe.returncode == 0:
            return backend
    raise RuntimeError("No offscreen OpenGL backend works; install libEGL or libOSMesa")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=3, help="scene seed (must have red, green and yellow blocks)")
    parser.add_argument("--out", type=Path, default=Path("outputs/demo_primitives.mp4"), help=".mp4 or .gif")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", choose_gl_backend())
    import imageio.v2 as imageio  # after MUJOCO_GL is set
    import mujoco

    from graspgraph.sim import SceneConfig, build_scene
    from graspgraph.sim.robot import Robot

    scene = build_scene(SceneConfig(seed=args.seed, n_blocks=3, n_bowls=2))
    renderer = mujoco.Renderer(scene.model, args.height, args.width)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.42, 0.0, 0.22)
    camera.distance, camera.azimuth, camera.elevation = 1.65, 150.0, -28.0

    frames = []
    steps_per_frame = max(1, round(1.0 / (args.fps * scene.model.opt.timestep)))
    robot = Robot(scene)

    def record() -> None:
        if robot.total_steps % steps_per_frame == 0:
            renderer.update_scene(scene.data, camera)
            frames.append(renderer.render())

    robot.on_step = record
    bowl = next(o.name for o in scene.objects if o.kind == "bowl")
    plan = [
        ("pick", robot.pick, "block_red"),
        ("place_on", robot.place_on, bowl),
        ("pick", robot.pick, "block_green"),
        ("place_on", robot.place_on, "block_yellow"),
        ("move_home", lambda _: robot.move_home(), None),
    ]
    all_ok = True
    for name, primitive, arg in plan:
        result = primitive(arg)
        all_ok &= result.success
        print(f"{name}({arg or ''}): {'ok' if result.success else 'FAILED'} - {result.message} [{result.steps} steps]")
    renderer.close()  # free the GL context now, not during interpreter shutdown

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".gif":
        imageio.mimsave(args.out, frames[::3], duration=3000 / args.fps, loop=0)  # a third of the frames keeps GIFs smaller
    else:
        imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=8)
    print(f"saved {len(frames)} frames to {args.out} (MUJOCO_GL={os.environ['MUJOCO_GL']})")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
