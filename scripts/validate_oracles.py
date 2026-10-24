"""Check that every task is solvable: run each oracle plan through the real primitives.

    uv run python scripts/validate_oracles.py                 # all 110 tasks
    uv run python scripts/validate_oracles.py --families stack tower
"""

import argparse
import sys
import time

from graspgraph.tasks import FAMILY_NAMES, build_suite
from graspgraph.tasks.oracle import run_oracle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families", nargs="+", choices=FAMILY_NAMES, help="default: all")
    args = parser.parse_args()

    start = time.time()
    tasks = build_suite(args.families)
    failures = []
    for task in tasks:
        verdict = run_oracle(task)
        print(f"{'ok  ' if verdict.success else 'FAIL'} {task.task_id:16s} {verdict.detail}")
        if not verdict.success:
            failures.append(task.task_id)
    passed = len(tasks) - len(failures)
    print(f"\noracle plans: {passed}/{len(tasks)} solved in {time.time() - start:.0f}s")
    if failures:
        print("failed: " + ", ".join(failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
