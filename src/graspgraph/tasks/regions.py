"""Named table regions, defined once and shared by the task generator, predicates and prompt.

Directions are from the robot's point of view, in the robot frame: "front" is away from
the robot base (+x), "back" is towards it (-x), "left" is +y and "right" is -y. Every
region lies inside the arm's top-down reach at table height.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    name: str  # identifier, e.g. "front_left"
    label: str  # how instructions refer to it, e.g. "front left corner"
    x: tuple[float, float]  # metres, robot frame
    y: tuple[float, float]

    def contains(self, x: float, y: float, slack: float = 0.0) -> bool:
        """Whether (x, y) lies inside the region, grown by `slack` metres on every side."""
        return self.x[0] - slack <= x <= self.x[1] + slack and self.y[0] - slack <= y <= self.y[1] + slack

    @property
    def centre(self) -> tuple[float, float]:
        return (sum(self.x) / 2, sum(self.y) / 2)


REGIONS: dict[str, Region] = {
    r.name: r
    for r in (
        Region("front_left", "front left corner", (0.56, 0.68), (0.18, 0.32)),
        Region("front_right", "front right corner", (0.56, 0.68), (-0.32, -0.18)),
        Region("back_left", "back left corner", (0.32, 0.44), (0.18, 0.32)),
        Region("back_right", "back right corner", (0.32, 0.44), (-0.32, -0.18)),
        Region("centre", "centre", (0.44, 0.56), (-0.07, 0.07)),
    )
}


def describe_regions() -> str:
    """One line per region, for the planner prompt."""
    return "\n".join(
        f"- {r.label} ({r.name}): x {r.x[0]:.2f} to {r.x[1]:.2f}, y {r.y[0]:.2f} to {r.y[1]:.2f}"
        for r in REGIONS.values()
    )
