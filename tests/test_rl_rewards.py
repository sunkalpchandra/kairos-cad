

def test_an_empty_sketch_does_not_earn_the_constrained_bonus():
    """An empty sketch reports fully_constrained: no geometry, no free DOF.

    Without a geometry guard this paid out in every episode, including ones
    that drew nothing at all.
    """
    from kairos.actions.executor import ActionResult
    from kairos.actions.schema import Operation
    from kairos.language import parse_requirement
    from kairos.rl.rewards import RewardTracker

    tracker = RewardTracker(parse_requirement("Design a plate 40 x 40 x 5 mm"))
    result = ActionResult(
        ok=True, operation=Operation.CREATE_SKETCH, message="", info={}, done=False
    )
    observation = {
        "summary": {"has_solid": False, "valid": False},
        "sketch": {"fully_constrained": True, "geometry_count": 0},
        "holes": [],
    }
    breakdown = tracker.step(result, observation)
    assert "fully_constrained_sketch" not in breakdown.components


def test_a_constrained_sketch_with_geometry_still_earns_it():
    from kairos.actions.executor import ActionResult
    from kairos.actions.schema import Operation
    from kairos.language import parse_requirement
    from kairos.rl.rewards import RewardTracker

    tracker = RewardTracker(parse_requirement("Design a plate 40 x 40 x 5 mm"))
    result = ActionResult(
        ok=True, operation=Operation.ADD_RECTANGLE, message="", info={}, done=False
    )
    observation = {
        "summary": {"has_solid": False, "valid": False},
        "sketch": {"fully_constrained": True, "geometry_count": 4},
        "holes": [],
    }
    breakdown = tracker.step(result, observation)
    assert "fully_constrained_sketch" in breakdown.components
