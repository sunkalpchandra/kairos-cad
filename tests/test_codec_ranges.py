"""Slot ranges must cover the dataset without dwarfing it.

Two failure directions, and the tests here pin both because each has already
shipped:

Too narrow, and encode clips. That corrupted 260 expert steps across 19.1% of
designs while every audit reported the trajectories as fully representable
(covered by the overflow tests in test_action_codec.py).

Too wide, and the policy loses precision. Widening every range on the reasoning
that headroom is free moved a mid-range fillet from 5 mm to 25 mm, and PPO's
invalid-action rate went 0.001 -> 0.422 with BC validity 0.957 -> 0.485. Nothing
failed; the codec audit still reported 0% unrepresentable, because a too-wide
range is not an encoding error.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from kairos.rl.action_space import (
    _COORD,
    _LENGTH,
    _OFFSET,
    _RADIUS,
    _SIDE,
    _SMALL,
)

DATASET = Path(__file__).resolve().parent.parent / "dataset"

#: How much wider than the observed data a range may be. Enough to absorb a
#: resampled dataset, tight enough that a blanket widening fails here.
MAX_HEADROOM = 3.0

#: Parameters grouped by the slot range that encodes them.
COORD_KEYS = {"cx", "cy", "x1", "y1", "x2", "y2", "x", "y"}


def _observed() -> dict[str, tuple[float, float]]:
    """Min and max of each parameter group across every expert trajectory."""
    spans: dict[str, list[float]] = defaultdict(list)
    for path in DATASET.glob("designs/design_*/trajectory.json"):
        try:
            actions = json.loads(path.read_text()).get("actions", [])
        except (OSError, json.JSONDecodeError):
            continue
        for entry in actions:
            # Trajectories record operations upper-case (ADD_CIRCLE); comparing
            # against lower-case names matches nothing and silently drops the
            # whole group, which is how the radius slot went unchecked here.
            operation = str(entry.get("operation", "")).lower()
            for key, value in (entry.get("parameters") or {}).items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if key in COORD_KEYS:
                    spans["coord"].append(value)
                elif key == "offset":
                    spans["offset"].append(value)
                elif key == "radius" and operation in ("add_circle", "add_arc"):
                    spans["radius"].append(value)
                elif key in ("width", "height") and operation == "add_rectangle":
                    spans["side"].append(value)
                elif key == "length" and operation in ("pad", "revolve"):
                    spans["length"].append(value)
                elif key == "size" or (key == "radius" and operation == "fillet"):
                    spans["small"].append(value)
    return {name: (min(v), max(v)) for name, v in spans.items() if v}


@pytest.fixture(scope="module")
def observed() -> dict[str, tuple[float, float]]:
    if not DATASET.exists():
        pytest.skip("no dataset/ to measure against")
    data = _observed()
    if not data:
        pytest.skip("dataset/ has no trajectories to measure")
    return data


RANGES = {
    "coord": _COORD,
    "offset": _OFFSET,
    "radius": _RADIUS,
    "side": _SIDE,
    "length": _LENGTH,
    "small": _SMALL,
}


@pytest.mark.parametrize("group", sorted(RANGES))
def test_range_covers_the_data(group, observed):
    """Anything outside its range raises, so a gap here means dropped steps."""
    if group not in observed:
        pytest.skip(f"no {group} parameters in this dataset")
    low, high = RANGES[group]
    seen_low, seen_high = observed[group]
    assert low <= seen_low, (
        f"{group}: data reaches {seen_low:.3f}, below the range floor {low}. "
        "encode() will raise and those expert steps will be dropped."
    )
    assert seen_high <= high, (
        f"{group}: data reaches {seen_high:.3f}, above the range ceiling {high}. "
        "encode() will raise and those expert steps will be dropped."
    )


@pytest.mark.parametrize("group", sorted(RANGES))
def test_range_is_not_absurdly_wider_than_the_data(group, observed):
    """A range far wider than the data costs the policy resolution.

    The policy emits a normalized value, so its error in millimetres scales with
    the span. Widening a range is a real change to the learning problem and
    should be a deliberate one.
    """
    if group not in observed:
        pytest.skip(f"no {group} parameters in this dataset")
    low, high = RANGES[group]
    seen_low, seen_high = observed[group]
    span = high - low
    used = max(seen_high - seen_low, 1e-6)
    assert span <= used * MAX_HEADROOM, (
        f"{group}: range spans {span:.1f} for data spanning {used:.1f} "
        f"({span / used:.1f}x). Every millimetre of unused range costs the "
        f"policy resolution; narrow it or raise MAX_HEADROOM deliberately."
    )


def test_every_expert_action_encodes(observed):
    """The end-to-end statement the ranges exist to support.

    Duplicates `scripts/audit_codec.py`, on purpose: the audit is a manual step
    and this one runs in CI.
    """
    from kairos.actions.schema import Action, Operation
    from kairos.rl.action_space import UnrepresentableAction, encode

    failures: list[str] = []
    checked = 0
    for path in sorted(DATASET.glob("designs/design_*/trajectory.json"))[:120]:
        try:
            actions = json.loads(path.read_text()).get("actions", [])
        except (OSError, json.JSONDecodeError):
            continue
        for entry in actions:
            try:
                operation = Operation(entry["operation"])
            except (KeyError, ValueError):
                continue
            action = Action(
                operation,
                target=entry.get("target"),
                parameters=entry.get("parameters") or {},
            )
            checked += 1
            try:
                encode(action)
            except UnrepresentableAction as err:
                failures.append(f"{path.parent.name} {operation.value}: {err}")

    assert checked > 0, "no expert actions were checked"
    assert not failures, (
        f"{len(failures)} of {checked} expert actions cannot be encoded:\n  "
        + "\n  ".join(failures[:5])
    )
