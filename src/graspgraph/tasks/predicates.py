"""Success predicates. They read only simulator state, never anything the planner says.

Each predicate compares two `Robot.get_scene()` snapshots: the state when the planner
started (after any task setup) and the final state. Tolerances are named constants below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from graspgraph.sim.scene import BLOCK_HALF, BOWL_HALF, BOWL_WALL_T, TABLE_HEIGHT
from graspgraph.tasks.regions import REGIONS

ON_TABLE_Z = TABLE_HEIGHT + BLOCK_HALF  # centre height of a block resting on the table
IN_BOWL_Z = TABLE_HEIGHT + BOWL_WALL_T + BLOCK_HALF  # centre height of a block resting in a bowl

Z_TOL = 0.01  # how far a "resting" block's centre may sit from its expected height
IN_BOWL_XY = BOWL_HALF  # block centre must lie inside the bowl's square footprint
STACK_XY_TOL = 0.02  # top block centre within 2 cm (xy) of the block below
REGION_SLACK = 0.01  # a block may overhang a named region's edge by 1 cm
LINE_X_SPREAD = 0.04  # blocks "in a line" left-to-right: their x values span at most 4 cm
LINE_MIN_GAP = 0.05  # neighbours in a line are at least 5 cm apart along y
UNDO_XY_TOL = 0.02  # an undone block is back within 2 cm of where it was
UNCHANGED_TOL = 0.015  # objects that should not move stay within 1.5 cm

State = dict[str, Any]  # a Robot.get_scene() snapshot


@dataclass(frozen=True)
class Verdict:
    success: bool
    detail: str


class Predicate(Protocol):
    def check(self, start: State, final: State) -> Verdict: ...


def positions(state: State) -> dict[str, np.ndarray]:
    """Object name -> position array from a get_scene() snapshot."""
    return {o["name"]: np.array(o["position"], dtype=float) for o in state["objects"]}


def _in_bowl(block: np.ndarray, bowl: np.ndarray) -> bool:
    return bool(np.all(np.abs(block[:2] - bowl[:2]) < IN_BOWL_XY)) and abs(block[2] - IN_BOWL_Z) < Z_TOL


def _on_top(top: np.ndarray, base: np.ndarray) -> bool:
    return np.linalg.norm(top[:2] - base[:2]) < STACK_XY_TOL and abs(top[2] - base[2] - 2 * BLOCK_HALF) < Z_TOL


def _on_table(block: np.ndarray) -> bool:
    return abs(block[2] - ON_TABLE_Z) < Z_TOL


def _moved(start: State, final: State, names: list[str], tol: float = UNCHANGED_TOL) -> list[str]:
    p0, p1 = positions(start), positions(final)
    return [n for n in names if np.linalg.norm(p1[n] - p0[n]) > tol]


@dataclass(frozen=True)
class InBowl:
    """`block` rests inside `bowl`."""

    block: str
    bowl: str

    def check(self, start: State, final: State) -> Verdict:
        p = positions(final)
        if final["held"] == self.block:
            return Verdict(False, f"{self.block} is still in the gripper")
        ok = _in_bowl(p[self.block], p[self.bowl])
        return Verdict(ok, f"{self.block} {'is' if ok else 'is not'} in {self.bowl}")


@dataclass(frozen=True)
class OnTop:
    """`top` rests directly on `base`."""

    top: str
    base: str

    def check(self, start: State, final: State) -> Verdict:
        p = positions(final)
        ok = _on_top(p[self.top], p[self.base])
        return Verdict(ok, f"{self.top} {'is' if ok else 'is not'} on {self.base}")


@dataclass(frozen=True)
class Tower:
    """Blocks stacked in this order, bottom first; the bottom block rests on the table."""

    bottom_to_top: tuple[str, ...]

    def check(self, start: State, final: State) -> Verdict:
        p = positions(final)
        if not _on_table(p[self.bottom_to_top[0]]):
            return Verdict(False, f"{self.bottom_to_top[0]} is not on the table")
        for base, top in zip(self.bottom_to_top, self.bottom_to_top[1:]):
            if not _on_top(p[top], p[base]):
                return Verdict(False, f"{top} is not on {base}")
        return Verdict(True, "tower is " + " < ".join(self.bottom_to_top))


@dataclass(frozen=True)
class AllInBowls:
    """Every listed block rests in its assigned bowl."""

    assignment: tuple[tuple[str, str], ...]  # (block, bowl) pairs

    def check(self, start: State, final: State) -> Verdict:
        p = positions(final)
        missing = [f"{b} not in {w}" for b, w in self.assignment if not _in_bowl(p[b], p[w])]
        return Verdict(not missing, "; ".join(missing) or "every block is in its bowl")


@dataclass(frozen=True)
class BowlCounts:
    """Each listed bowl holds exactly the given number of blocks."""

    counts: tuple[tuple[str, int], ...]  # (bowl, number of blocks) pairs

    def check(self, start: State, final: State) -> Verdict:
        p = positions(final)
        blocks = [o["name"] for o in final["objects"] if o["kind"] == "block"]
        wrong = []
        for bowl, n in self.counts:
            have = sum(_in_bowl(p[b], p[bowl]) for b in blocks)
            if have != n:
                wrong.append(f"{bowl} holds {have}, wanted {n}")
        return Verdict(not wrong, "; ".join(wrong) or "every bowl holds the right number")


@dataclass(frozen=True)
class InRegion:
    """`block` rests on the table inside a named region."""

    block: str
    region: str

    def check(self, start: State, final: State) -> Verdict:
        p = positions(final)[self.block]
        region = REGIONS[self.region]
        ok = region.contains(p[0], p[1], REGION_SLACK) and _on_table(p)
        return Verdict(ok, f"{self.block} {'is' if ok else 'is not'} on the table in the {region.label}")


@dataclass(frozen=True)
class InLine:
    """Blocks rest on the table in a left-to-right row (left is +y) in the given order."""

    left_to_right: tuple[str, ...]

    def check(self, start: State, final: State) -> Verdict:
        p = positions(final)
        pts = [p[b] for b in self.left_to_right]
        if not all(_on_table(q) for q in pts):
            return Verdict(False, "not every block is resting on the table")
        ys = [q[1] for q in pts]
        if any(a - b < LINE_MIN_GAP for a, b in zip(ys, ys[1:])):
            return Verdict(False, "blocks are not in the stated left-to-right order")
        spread = max(q[0] for q in pts) - min(q[0] for q in pts)
        if spread > LINE_X_SPREAD:
            return Verdict(False, f"blocks are not in a line (x spread {spread * 100:.1f} cm)")
        return Verdict(True, "blocks are lined up " + ", ".join(self.left_to_right))


@dataclass(frozen=True)
class BackAt:
    """`block` is back on the table at (x, y) and nothing else moved."""

    block: str
    x: float
    y: float

    def check(self, start: State, final: State) -> Verdict:
        p = positions(final)[self.block]
        if np.hypot(p[0] - self.x, p[1] - self.y) > UNDO_XY_TOL or not _on_table(p):
            return Verdict(False, f"{self.block} is not back at ({self.x:.3f}, {self.y:.3f})")
        others = [o["name"] for o in start["objects"] if o["name"] != self.block]
        moved = _moved(start, final, others)
        if moved:
            return Verdict(False, f"{', '.join(moved)} also moved")
        return Verdict(True, f"{self.block} is back where it was")


@dataclass(frozen=True)
class Unchanged:
    """Nothing in the world moved (used for tasks where the planner must ask or refuse)."""

    def check(self, start: State, final: State) -> Verdict:
        moved = _moved(start, final, [o["name"] for o in start["objects"]])
        if final["held"] is not None:
            moved.append(f"{final['held']} (held)")
        return Verdict(not moved, f"moved: {', '.join(moved)}" if moved else "world unchanged")
