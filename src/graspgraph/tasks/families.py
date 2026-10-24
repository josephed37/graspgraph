"""Task families: 10 families x 11 seeds, each seed giving one reproducible task.

Every family builder takes a seed and returns a Task. It looks for a suitable scene by
trying a fixed sequence of scene seeds (derived from the family and task seed) until the
scene fits the task: e.g. "sort" needs blocks whose colours match bowls. All choices
(objects, phrasing, regions) come from a generator seeded the same way, so a task id always
rebuilds the identical scene, instruction, predicate and oracle plan.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from graspgraph.actions import ASK, REFUSE, ToolCall
from graspgraph.sim import COLOURS, SceneConfig, build_scene
from graspgraph.sim.robot import Robot
from graspgraph.sim.scene import footprint_radius
from graspgraph.tasks.predicates import (
    AllInBowls,
    BackAt,
    BowlCounts,
    InBowl,
    InLine,
    InRegion,
    OnTop,
    Predicate,
    State,
    Tower,
    Unchanged,
)
from graspgraph.tasks.regions import REGIONS, Region
from graspgraph.tasks.task import ExpectedOutcome, Task

SEEDS_PER_FAMILY = 11
MAX_SCENE_ATTEMPTS = 100

# Extra clearance (beyond footprints touching) when the oracle chooses a table spot. Blocks get
# more room because the open fingers reach ~5 cm from a block's centre when it is picked later.
SPOT_MARGIN_BOWL = 0.01
SPOT_MARGIN_BLOCK = 0.035
REFERENCE_MARGIN = 0.04  # "left-most" etc. must win by at least 4 cm to be unambiguous


@dataclass(frozen=True)
class SceneView:
    """What a family builder may look at when designing a task for a candidate scene."""

    blocks: tuple[str, ...]
    bowls: tuple[str, ...]
    colour: dict[str, str]  # object name -> colour
    xy: dict[str, np.ndarray]
    kind: dict[str, str]
    state: State  # get_scene() at the start


@dataclass(frozen=True)
class Draft:
    instruction: str
    predicate: Predicate
    oracle: tuple[ToolCall, ...]
    expected_outcome: ExpectedOutcome = "act"
    setup: tuple[ToolCall, ...] = ()


Builder = Callable[[SceneView, np.random.Generator], Draft | None]


def _pick(rng: np.random.Generator, options: tuple | list) -> object:
    return options[int(rng.integers(len(options)))]


def _pick_place(block: str, target: str) -> tuple[ToolCall, ToolCall]:
    return ToolCall("pick", {"obj_name": block}), ToolCall("place_on", {"obj_name": target})


def _generate(family: str, seed: int, n_blocks: int, n_bowls: int, build: Builder) -> Task:
    """Try scene seeds in a fixed order until `build` accepts one."""
    family_index = FAMILY_NAMES.index(family)
    for attempt in range(MAX_SCENE_ATTEMPTS):
        config = SceneConfig(seed=100_000 * family_index + 100 * seed + attempt, n_blocks=n_blocks, n_bowls=n_bowls)
        try:
            scene = build_scene(config)
        except RuntimeError:  # objects did not fit on the table
            continue
        state = Robot(scene).get_scene()
        view = SceneView(
            blocks=tuple(o.name for o in scene.objects if o.kind == "block"),
            bowls=tuple(o.name for o in scene.objects if o.kind == "bowl"),
            colour={o.name: o.colour for o in scene.objects},
            xy={o.name: np.array(o.xy) for o in scene.objects},
            kind={o.name: o.kind for o in scene.objects},
            state=state,
        )
        draft = build(view, np.random.default_rng([family_index, seed, attempt]))
        if draft is None:
            continue
        # An "act" task must not already be solved at the start (setup tasks change the start).
        if draft.expected_outcome == "act" and not draft.setup and draft.predicate.check(state, state).success:
            continue
        return Task(
            task_id=f"{family.replace('_', '-')}-{seed:03d}",
            family=family,
            seed=seed,
            scene_config=config,
            instruction=draft.instruction,
            expected_outcome=draft.expected_outcome,
            predicate=draft.predicate,
            oracle=draft.oracle,
            setup=draft.setup,
        )
    raise RuntimeError(f"no suitable scene for {family} seed {seed} in {MAX_SCENE_ATTEMPTS} attempts")


def _clear_of(view: SceneView, spot: np.ndarray, ignore: tuple[str, ...]) -> bool:
    """Whether a block could be put down at `spot` and still leave room to pick its neighbours."""
    for name, xy in view.xy.items():
        if name in ignore:
            continue
        margin = SPOT_MARGIN_BOWL if view.kind[name] == "bowl" else SPOT_MARGIN_BLOCK
        if np.linalg.norm(spot - xy) < footprint_radius(view.kind[name]) + footprint_radius("block") + margin:
            return False
    return True


# ------------------------------------------------------------------ families


def pick_place(seed: int) -> Task:
    templates = (
        "Put the {b} block in the {w} bowl.",
        "Place the {b} block into the {w} bowl.",
        "Move the {b} block to the {w} bowl.",
        "Could you drop the {b} block in the {w} bowl?",
    )

    def build(view: SceneView, rng: np.random.Generator) -> Draft:
        block, bowl = _pick(rng, view.blocks), _pick(rng, view.bowls)
        text = _pick(rng, templates).format(b=view.colour[block], w=view.colour[bowl])
        return Draft(text, InBowl(block, bowl), _pick_place(block, bowl))

    return _generate("pick_place", seed, 3, 2, build)


def stack(seed: int) -> Task:
    templates = (
        "Stack the {t} block on the {b} one.",
        "Put the {t} block on top of the {b} block.",
        "Place the {t} block onto the {b} block.",
    )

    def build(view: SceneView, rng: np.random.Generator) -> Draft:
        top, base = (str(n) for n in rng.choice(view.blocks, size=2, replace=False))
        text = _pick(rng, templates).format(t=view.colour[top], b=view.colour[base])
        return Draft(text, OnTop(top, base), _pick_place(top, base))

    return _generate("stack", seed, 3, 2, build)


def tower(seed: int) -> Task:
    templates = (
        "Build a tower of all three blocks: {a} at the bottom, then {b}, with {c} on top.",
        "Stack the blocks into a tower, bottom to top: {a}, {b}, {c}.",
        "Make a three-block tower with {a} at the bottom, {b} in the middle and {c} at the top.",
    )

    def build(view: SceneView, rng: np.random.Generator) -> Draft:
        a, b, c = (str(n) for n in rng.permutation(view.blocks))
        text = _pick(rng, templates).format(a=view.colour[a], b=view.colour[b], c=view.colour[c])
        return Draft(text, Tower((a, b, c)), _pick_place(b, a) + _pick_place(c, b))

    return _generate("tower", seed, 3, 1, build)


def sort(seed: int) -> Task:
    templates = (
        "Put each block in the bowl of the same colour.",
        "Sort the blocks into the bowls that match their colours.",
        "Match the blocks to the bowls by colour.",
    )

    def build(view: SceneView, rng: np.random.Generator) -> Draft | None:
        bowl_of = {view.colour[w]: w for w in view.bowls}
        pairs = tuple((b, bowl_of[view.colour[b]]) for b in view.blocks if view.colour[b] in bowl_of)
        if len(pairs) < 2:  # need at least two blocks to sort
            return None
        oracle = tuple(call for b, w in pairs for call in _pick_place(b, w))
        return Draft(str(_pick(rng, templates)), AllInBowls(pairs), oracle)

    return _generate("sort", seed, 3, 3, build)


def spatial(seed: int) -> Task:
    templates = (
        "Move the {ref} to the {region} of the table.",
        "Put the {ref} in the {region} of the table.",
        "Take the {ref} and place it in the {region} of the table.",
    )
    # phrase -> score to maximise over blocks (None means "refer to it by colour")
    references = {
        "left-most block": lambda xy: xy[1],
        "right-most block": lambda xy: -xy[1],
        "block closest to the robot": lambda xy: -np.hypot(*xy),
        "block furthest from the robot": lambda xy: np.hypot(*xy),
        "colour": None,
    }

    def build(view: SceneView, rng: np.random.Generator) -> Draft | None:
        phrase = str(_pick(rng, list(references)))
        score = references[phrase]
        if score is None:
            block = str(_pick(rng, view.blocks))
            phrase = f"{view.colour[block]} block"
        else:
            ranked = sorted(view.blocks, key=lambda b: score(view.xy[b]), reverse=True)
            if score(view.xy[ranked[0]]) - score(view.xy[ranked[1]]) < REFERENCE_MARGIN:
                return None  # too close to call: the reference would be ambiguous
            block = ranked[0]
        region = REGIONS[str(_pick(rng, list(REGIONS)))]
        spot = _free_spot(view, region, block)
        if spot is None:
            return None
        text = _pick(rng, templates).format(ref=phrase, region=region.label)
        oracle = (ToolCall("pick", {"obj_name": block}), ToolCall("place_at", {"x": spot[0], "y": spot[1]}))
        return Draft(text, InRegion(block, region.name), oracle)

    return _generate("spatial", seed, 3, 2, build)


def _free_spot(view: SceneView, region: Region, block: str) -> tuple[float, float] | None:
    """Clear spot inside `region` (at least 1.5 cm from its edges), nearest its centre."""
    inset, step = 0.015, 0.01
    xs = np.arange(region.x[0] + inset, region.x[1] - inset + 1e-9, step)
    ys = np.arange(region.y[0] + inset, region.y[1] - inset + 1e-9, step)
    candidates = sorted(
        (np.array([x, y]) for x in xs for y in ys), key=lambda p: np.linalg.norm(p - np.array(region.centre))
    )
    for spot in candidates:
        if _clear_of(view, spot, ignore=(block,)):
            return (round(float(spot[0]), 3), round(float(spot[1]), 3))
    return None


def arrange(seed: int) -> Task:
    templates = (
        "Line the blocks up from left to right: {a}, {b}, {c}.",
        "Arrange the blocks in a row, left to right: {a}, {b}, {c}.",
        "Put the blocks in a line with {a} on the left, {b} in the middle and {c} on the right.",
    )
    spacing = 0.10

    def build(view: SceneView, rng: np.random.Generator) -> Draft | None:
        order = tuple(str(n) for n in rng.permutation(view.blocks))
        for x in np.arange(0.40, 0.621, 0.02):
            for centre in np.arange(-0.10, 0.101, 0.025):
                spots = [np.array([x, centre + spacing * (1 - i)]) for i in range(3)]  # left (+y) first
                if all(_clear_of(view, s, ignore=(order[i],)) for i, s in enumerate(spots)):
                    oracle = tuple(
                        call
                        for block, s in zip(order, spots)
                        for call in (
                            ToolCall("pick", {"obj_name": block}),
                            ToolCall("place_at", {"x": round(float(s[0]), 3), "y": round(float(s[1]), 3)}),
                        )
                    )
                    text = _pick(rng, templates).format(**{k: view.colour[b] for k, b in zip("abc", order)})
                    return Draft(text, InLine(order), oracle)
        return None

    return _generate("arrange", seed, 3, 1, build)


def count(seed: int) -> Task:
    templates = (
        "Put {n} {blocks} in each bowl.",
        "Each bowl should end up holding exactly {n} {blocks}.",
        "Fill every bowl with {n} {blocks}.",
    )
    per_bowl = 1 + seed % 2  # alternate one and two blocks per bowl

    def build(view: SceneView, rng: np.random.Generator) -> Draft:
        remaining = list(view.blocks)
        oracle: list[ToolCall] = []
        for bowl in view.bowls:  # each bowl takes the nearest free blocks
            for _ in range(per_bowl):
                block = min(remaining, key=lambda b: np.linalg.norm(view.xy[b] - view.xy[bowl]))
                remaining.remove(block)
                oracle += _pick_place(block, bowl)
        text = _pick(rng, templates).format(n=["one", "two"][per_bowl - 1], blocks="block" if per_bowl == 1 else "blocks")
        return Draft(text, BowlCounts(tuple((w, per_bowl) for w in view.bowls)), tuple(oracle))

    return _generate("count", seed, 2 * per_bowl + (1 if per_bowl == 1 else 0), 2, build)


def ambiguous(seed: int) -> Task:
    variant = seed % 4  # which kind of missing information

    def build(view: SceneView, rng: np.random.Generator) -> Draft:
        c = view.colour[str(_pick(rng, view.blocks))]
        if variant == 0:
            text, question = "Move the block into the bowl.", "Which block should I move into the bowl?"
        elif variant == 1:
            bowls = " or the ".join(f"{view.colour[w]} bowl" for w in view.bowls)
            text, question = f"Put the {c} block in the bowl.", f"Which bowl: the {bowls}?"
        elif variant == 2:
            text, question = f"Stack the {c} block on the other block.", f"Which block should the {c} block go on?"
        else:
            text, question = f"Put the block on the {c} block.", f"Which block should I put on the {c} block?"
        return Draft(text, Unchanged(), (ToolCall(ASK, {"question": question}),), expected_outcome="ask")

    return _generate("ambiguous", seed, 3, 1 if variant == 0 else 2, build)


def impossible(seed: int) -> Task:
    variant = seed % 3  # what makes it impossible

    def build(view: SceneView, rng: np.random.Generator) -> Draft:
        block, bowl = view.colour[str(_pick(rng, view.blocks))], view.colour[str(_pick(rng, view.bowls))]
        if variant == 0:
            missing = _pick(rng, [c for c in COLOURS if c not in {view.colour[b] for b in view.blocks}])
            text, reason = f"Put the {missing} block in the {bowl} bowl.", f"There is no {missing} block."
        elif variant == 1:
            missing = _pick(rng, [c for c in COLOURS if c not in {view.colour[w] for w in view.bowls}])
            text, reason = f"Put the {block} block in the {missing} bowl.", f"There is no {missing} bowl."
        else:
            text = f"Pick up the {bowl} bowl and put it on the {block} block."
            reason = "Bowls are fixed to the table and cannot be picked up."
        return Draft(text, Unchanged(), (ToolCall(REFUSE, {"reason": reason}),), expected_outcome="refuse")

    return _generate("impossible", seed, 3, 2, build)


def undo(seed: int) -> Task:
    templates = ("Undo the last move.", "Put the block you just moved back where it was.", "Reverse your last move.")
    two_moves = seed % 2 == 1

    def build(view: SceneView, rng: np.random.Generator) -> Draft:
        a, b, c = (str(n) for n in rng.permutation(view.blocks))
        bowl_a, bowl_b = (str(n) for n in rng.permutation(view.bowls))
        setup = _pick_place(a, bowl_a)
        moved = a
        if two_moves:  # the second move goes onto a block or into the other bowl
            setup += _pick_place(b, c if rng.integers(2) else bowl_b)
            moved = b
        x, y = (round(float(v), 3) for v in view.xy[moved])
        oracle = (ToolCall("pick", {"obj_name": moved}), ToolCall("place_at", {"x": x, "y": y}))
        return Draft(str(_pick(rng, templates)), BackAt(moved, x, y), oracle, setup=setup)

    return _generate("undo", seed, 3, 2, build)


FAMILIES: dict[str, Callable[[int], Task]] = {
    "pick_place": pick_place,
    "stack": stack,
    "tower": tower,
    "sort": sort,
    "spatial": spatial,
    "arrange": arrange,
    "count": count,
    "ambiguous": ambiguous,
    "impossible": impossible,
    "undo": undo,
}
FAMILY_NAMES = tuple(FAMILIES)
