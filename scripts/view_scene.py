"""Open a seeded scene in the MuJoCo viewer.

    python scripts/view_scene.py --seed 3 --blocks 4 --bowls 2
"""

import argparse

import mujoco.viewer

from graspgraph.sim import SceneConfig, build_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--bowls", type=int, default=2)
    args = parser.parse_args()

    scene = build_scene(SceneConfig(seed=args.seed, n_blocks=args.blocks, n_bowls=args.bowls))
    for obj in scene.objects:
        print(f"{obj.name:14s} {obj.kind:5s} at x={obj.xy[0]:.3f} y={obj.xy[1]:.3f}")

    # The viewer runs the physics live; close the window to exit.
    mujoco.viewer.launch(scene.model, scene.data)


if __name__ == "__main__":
    main()
