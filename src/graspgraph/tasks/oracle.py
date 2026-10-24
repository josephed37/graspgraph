"""Run a task's oracle plan through the real primitives and check its predicate."""

from __future__ import annotations

from graspgraph.actions import STOP_TOOLS, execute
from graspgraph.tasks.predicates import Verdict
from graspgraph.tasks.task import Task, start_episode


def run_oracle(task: Task) -> Verdict:
    """Execute the oracle plan on a fresh episode; fail on any primitive failure or a false predicate."""
    episode = start_episode(task)
    for call in task.oracle:
        if call.name in STOP_TOOLS:
            break
        result = execute(episode.robot, call)
        if not result.success:
            return Verdict(False, f"{call} failed: {result.message}")
    return episode.verdict()
