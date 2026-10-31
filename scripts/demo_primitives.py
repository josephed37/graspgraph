"""Run a short primitive sequence and record it offscreen to a video.

    uv run python scripts/demo_primitives.py                      # outputs/demo_primitives.mp4
    uv run python scripts/demo_primitives.py --out outputs/demo.gif --width 320 --height 240

Sequence: pick the red block, put it in a bowl, stack the green block on the yellow one,
go home. Uses MUJOCO_GL if set, otherwise EGL with an OSMesa fallback (no display needed).
"""

import argparse
import os
import sys
from pathlib import Path

from offscreen import VideoRecorder, setup_gl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=3, help="scene seed (must have red, green and yellow blocks)")
    parser.add_argument("--out", type=Path, default=Path("outputs/demo_primitives.mp4"), help=".mp4 or .gif")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    setup_gl()
    from graspgraph.sim import SceneConfig, build_scene  # after the GL backend is chosen
    from graspgraph.sim.robot import Robot

    scene = build_scene(SceneConfig(seed=args.seed, n_blocks=3, n_bowls=2))
    robot = Robot(scene)
    video = VideoRecorder(robot, args.fps, args.width, args.height)
    robot.on_step = video.record
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
    video.save(args.out)
    print(f"saved {len(video.frames)} frames to {args.out} (MUJOCO_GL={os.environ['MUJOCO_GL']})")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
