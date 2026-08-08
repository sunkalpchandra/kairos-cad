"""Draw a closed profile as ADD_LINE actions the codec can express.

Six of the eight families sketch their outline with a single `ADD_POLYGON`
carrying an arbitrary vertex list. The action codec can only express a polygon
as a *regular* n-gon, so those steps are unrepresentable: behavioral cloning
drops them (7.8% of all expert steps), and an oracle replaying the expert
through the codec cannot rebuild the part. Which is why the measured ceiling on
BUILD tasks is 0.431 rather than 1.0.

Both `kairos/rl/action_space.py` and `kairos/training/bc_dataset.py` already
prescribe the fix in their docstrings, "expand those into ADD_LINE actions", and this is
that expansion. One `ADD_LINE` per edge, closing back to the first
vertex, each of which round-trips through the codec exactly.

The cost is honest and worth stating: a 6-vertex L profile becomes 6 actions
instead of 1, so expert trajectories get longer. That is the point. The policy
now sees every edge it must draw, in an action it can actually emit, rather than
one atomic step it can never reproduce.
"""

from __future__ import annotations

from collections.abc import Sequence

from kairos.actions.schema import Action, Operation

#: Vertices closer than this are treated as duplicates, in mm.
_MIN_EDGE = 1e-6


def profile_actions(points: Sequence[Sequence[float]]) -> list[Action]:
    """One ADD_LINE per edge of a closed profile.

    Args:
        points: the profile's vertices in order. The closing edge back to the
            first vertex is added automatically, so callers pass an open list.

    Raises:
        ValueError: fewer than three distinct vertices, that is not a profile,
            and emitting it would produce a sketch that cannot be padded.
    """
    cleaned: list[tuple[float, float]] = []
    for point in points:
        vertex = (float(point[0]), float(point[1]))
        if not cleaned or _distance(cleaned[-1], vertex) > _MIN_EDGE:
            cleaned.append(vertex)
    # A profile whose last vertex repeats the first is already closed.
    if len(cleaned) > 1 and _distance(cleaned[0], cleaned[-1]) <= _MIN_EDGE:
        cleaned.pop()

    if len(cleaned) < 3:
        raise ValueError(f"a closed profile needs 3+ distinct vertices, got {len(cleaned)}")

    actions: list[Action] = []
    for index, start in enumerate(cleaned):
        end = cleaned[(index + 1) % len(cleaned)]
        actions.append(
            Action(
                Operation.ADD_LINE,
                # Unrounded: the families assert closed-form volumes to rel=1e-6,
                # and rounding vertices to 4 decimals shifts a revolved solid
                # by ~2e-6 -- enough to fail the analytic check.
                parameters={
                    "x1": start[0], "y1": start[1],
                    "x2": end[0], "y2": end[1],
                },
            )
        )
    return actions


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def is_regular(points: Sequence[Sequence[float]], tolerance: float = 1e-3) -> bool:
    """Whether a profile is a regular n-gon the codec can express directly.

    Regular profiles keep their single ADD_POLYGON action: expanding them would
    lengthen trajectories for no gain, since the codec already round-trips them.
    """
    from kairos.rl.action_space import UnrepresentableAction, _fit_regular_polygon

    try:
        _fit_regular_polygon([list(p) for p in points])
        return True
    except (UnrepresentableAction, ValueError, IndexError):
        return False


def draw_profile(points: Sequence[Sequence[float]]) -> list[Action]:
    """Emit the cheapest codec-expressible drawing of a closed profile.

    A regular n-gon stays one ADD_POLYGON; anything else becomes ADD_LINE edges.
    """
    if is_regular(points):
        return [Action(Operation.ADD_POLYGON, parameters={"points": [list(p) for p in points]})]
    return profile_actions(points)
