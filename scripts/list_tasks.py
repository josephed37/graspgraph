"""Print every benchmark task: id, family, expected outcome and instruction.

    uv run python scripts/list_tasks.py
"""

from graspgraph.tasks import build_suite


def main() -> None:
    tasks = build_suite()
    print(f"{'task_id':16s} {'family':11s} {'expect':6s} instruction")
    for task in tasks:
        print(f"{task.task_id:16s} {task.family:11s} {task.expected_outcome:6s} {task.instruction}")
    print(f"\n{len(tasks)} tasks")


if __name__ == "__main__":
    main()
