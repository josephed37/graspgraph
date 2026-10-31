"""Record the closed loop recovering from an injected fault, offscreen, to a video.

    uv run python scripts/demo_recovery.py                         # outputs/demo_recovery.mp4
    uv run python scripts/demo_recovery.py --task-id tower-000 --fault knock

A scripted fake model plans with the task's oracle plan; the fault hits; the verify node
reports the mismatch; the fake replans (re-picks the block and carries on); the task ends
solved. The step log, mismatch and replan are printed. No LLM is called.
"""

import argparse
import os
import sys
from pathlib import Path

from offscreen import VideoRecorder, setup_gl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-id", default="stack-000", help="a stack or pick_place task works well")
    parser.add_argument("--fault", choices=["drop", "displace", "knock"], default="drop")
    parser.add_argument("--out", type=Path, default=Path("outputs/demo_recovery.mp4"), help=".mp4 or .gif")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    setup_gl()
    from graspgraph.bench.faults import FaultInjector, FaultPlan  # after the GL backend is chosen
    from graspgraph.planner.backends import FakeBackend, oracle_policy, oracle_repair
    from graspgraph.planner.closed_loop import run_closed_loop
    from graspgraph.tasks import get_task, start_episode

    task = get_task(args.task_id)
    episode = start_episode(task)
    video = VideoRecorder(episode.robot, args.fps, args.width, args.height)
    episode.robot.on_step = video.record
    injector = FaultInjector.for_run(task, FaultPlan(args.fault, 0))
    backend = FakeBackend(oracle_policy, model="oracle", repair=oracle_repair)

    print(f"{task.task_id}: {task.instruction}  (fault: {args.fault})")
    run = run_closed_loop(backend, episode, injector)
    episode.robot._step_for(1.0)  # a second of stillness at the end of the video

    for i, result in enumerate(run.results, start=1):
        args_text = ", ".join(map(str, result["args"].values()))
        print(f"  step {i}: {result['name']}({args_text}) -> {'ok' if result['success'] else 'FAILED'}: {result['message']}")
        if injector.injected and i == injector.step:
            print(f"    fault injected: {injector.detail}")
    for mismatch in run.mismatches:
        print(f"  verify mismatch: {mismatch}\n  replan: the model got the current scene and submitted a new plan")
    verdict = episode.verdict()
    print(f"plans: {len(run.plans)}, replans: {run.replans}, task {'solved' if verdict.success else 'NOT solved'}: {verdict.detail}")

    video.save(args.out)
    print(f"saved {len(video.frames)} frames to {args.out} (MUJOCO_GL={os.environ['MUJOCO_GL']})")
    sys.exit(0 if verdict.success else 1)


if __name__ == "__main__":
    main()
