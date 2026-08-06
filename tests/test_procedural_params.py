"""Pure-python tests of procedural recipe parameters and action emission."""

import random

from kairos.actions.schema import Operation
from kairos.data.procedural import (
    LBracketParams,
    PlateParams,
    expected_hole_count,
    l_bracket_hole_actions,
    l_bracket_profile_actions,
    plate_actions,
)


def test_default_bracket_is_feasible():
    assert LBracketParams().is_feasible()


def test_infeasible_when_holes_hit_corner():
    p = LBracketParams(leg1=20.0, thickness=8.0, hole_margin=2.0, hole_diameter=6.0)
    assert not p.is_feasible()


def test_infeasible_when_adjacent_holes_would_merge():
    # 3 holes of d=6 on a short leg: spacing < d + 2 -> slot, not holes.
    p = LBracketParams(
        leg1=42.0, leg2=80.0, thickness=4.0, hole_diameter=6.0,
        holes_per_leg=3, hole_margin=9.0,
    )
    positions = p.hole_positions(p.leg1)
    assert positions[1] - positions[0] <= p.hole_diameter + 2.0  # scenario is real
    assert not p.is_feasible()


def test_hole_positions_stay_inside_leg():
    p = LBracketParams()
    for leg in (p.leg1, p.leg2):
        for x in p.hole_positions(leg):
            assert p.thickness < x < leg


def test_sampled_params_reproducible_by_seed():
    a = LBracketParams.sample(random.Random(42))
    b = LBracketParams.sample(random.Random(42))
    assert a == b


def test_profile_actions_shape():
    ops = [a.operation for a in l_bracket_profile_actions(LBracketParams())]
    assert ops == [Operation.CREATE_SKETCH, Operation.ADD_POLYGON, Operation.PAD]


def test_hole_actions_count_matches_holes_per_leg():
    p = LBracketParams(holes_per_leg=3)
    actions = l_bracket_hole_actions(p)
    circles = [a for a in actions if a.operation is Operation.ADD_CIRCLE]
    pockets = [a for a in actions if a.operation is Operation.POCKET]
    assert len(circles) == 6 and len(pockets) == 2
    assert all(a.parameters["radius"] == p.hole_diameter / 2 for a in circles)


def test_expected_hole_count():
    assert expected_hole_count(LBracketParams(holes_per_leg=2)) == 4
    assert expected_hole_count(PlateParams(holes_x=4, holes_y=2)) == 8


def test_plate_grid_and_actions_agree():
    p = PlateParams(holes_x=3, holes_y=2)
    assert len(p.hole_grid()) == 6
    circles = [a for a in plate_actions(p) if a.operation is Operation.ADD_CIRCLE]
    assert len(circles) == 6


def test_plate_infeasible_when_holes_overlap():
    p = PlateParams(length=30.0, width=20.0, holes_x=4, holes_y=2, hole_diameter=8.0, hole_margin=6.0)
    assert not p.is_feasible()


def test_plate_actions_serialize():
    for action in plate_actions(PlateParams()):
        assert action.to_dict()["operation"] in Operation._value2member_map_
