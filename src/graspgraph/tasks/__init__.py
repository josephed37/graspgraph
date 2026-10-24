"""The benchmark task suite: 10 families x 11 seeds = 110 reproducible tasks."""

from __future__ import annotations

from graspgraph.tasks.families import FAMILIES, FAMILY_NAMES, SEEDS_PER_FAMILY
from graspgraph.tasks.task import Episode, Task, start_episode


def build_suite(families: list[str] | None = None) -> list[Task]:
    """All tasks of the given families (default: all), in family then seed order."""
    names = families or list(FAMILY_NAMES)
    unknown = [f for f in names if f not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown families {unknown}; choose from {list(FAMILY_NAMES)}")
    return [FAMILIES[f](seed) for f in names for seed in range(SEEDS_PER_FAMILY)]


def get_task(task_id: str) -> Task:
    """Rebuild one task from its id, e.g. "stack-007"."""
    family_slug, _, seed = task_id.rpartition("-")
    family = family_slug.replace("-", "_")
    if family not in FAMILIES or not seed.isdigit() or int(seed) >= SEEDS_PER_FAMILY:
        raise ValueError(f"unknown task id '{task_id}'")
    return FAMILIES[family](int(seed))


__all__ = ["FAMILY_NAMES", "SEEDS_PER_FAMILY", "Episode", "Task", "build_suite", "get_task", "start_episode"]
