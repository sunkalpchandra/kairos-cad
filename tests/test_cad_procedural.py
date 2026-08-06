"""CAD integration tests: procedural recipes and validated dataset generation."""

import json
import random

import pytest

from kairos.actions.executor import ActionExecutor
from kairos.data import procedural
from kairos.data.generator import GenerationStats, generate_dataset, generate_design

pytestmark = pytest.mark.cad


def test_l_bracket_recipe_end_to_end(engine):
    params = procedural.LBracketParams(fillet_radius=2.0)
    executor = ActionExecutor(engine)
    actions = procedural.build_l_bracket(executor, params)
    assert engine.check_validity().is_valid
    holes = engine.find_holes(diameter=params.hole_diameter)
    assert len(holes) == procedural.expected_hole_count(params)
    ops = [a.operation.value for a in actions]
    assert ops[0] == "CREATE_SKETCH" and ops[-1] == "FINISH_DESIGN"
    assert "FILLET" in ops


def test_l_bracket_mass_matches_analytic(engine):
    p = procedural.LBracketParams(fillet_radius=0.0)
    procedural.build_l_bracket(ActionExecutor(engine), p)
    import math

    profile_area = p.leg1 * p.thickness + (p.leg2 - p.thickness) * p.thickness
    hole_volume = (
        math.pi
        * (p.hole_diameter / 2) ** 2
        * p.thickness
        * procedural.expected_hole_count(p)
    )
    expected = profile_area * p.width - hole_volume
    assert engine.measure_volume() == pytest.approx(expected, rel=1e-6)


def test_plate_recipe_end_to_end(engine):
    params = procedural.PlateParams(holes_x=3, holes_y=2)
    procedural.build_plate(ActionExecutor(engine), params)
    assert engine.check_validity().is_valid
    assert len(engine.find_holes(diameter=params.hole_diameter)) == 6


def test_sampled_recipes_build_validly():
    """Property test: every feasible sampled design must build and validate."""
    from kairos.cad.engine import CADEngine

    rng = random.Random(7)
    built = 0
    for _ in range(6):
        for cls, builder in (
            (procedural.LBracketParams, procedural.build_l_bracket),
            (procedural.PlateParams, procedural.build_plate),
        ):
            params = cls.sample(rng)
            if not params.is_feasible():
                continue
            engine = CADEngine("prop")
            try:
                builder(ActionExecutor(engine), params)
                assert engine.check_validity().is_valid
                assert len(engine.find_holes(diameter=params.hole_diameter)) == (
                    procedural.expected_hole_count(params)
                )
                built += 1
            finally:
                engine.close()
    assert built >= 4, "too few feasible samples exercised"


def test_generate_design_writes_complete_layout(tmp_path):
    rng = random.Random(3)
    stats = GenerationStats()
    written = False
    design_id = 0
    while not written and design_id < 20:
        written = generate_design("l_bracket", rng, tmp_path, design_id, stats)
        design_id += 1
    assert written
    design_dir = next(tmp_path.glob("design_*"))
    expected_files = {
        "model.FCStd",
        "model.step",
        "model.stl",
        "iso.png",
        "front.png",
        "top.png",
        "right.png",
        "state.json",
        "requirements.json",
        "trajectory.json",
    }
    assert expected_files <= {p.name for p in design_dir.iterdir()}
    state = json.loads((design_dir / "state.json").read_text())
    assert state["valid"] is True
    requirements = json.loads((design_dir / "requirements.json").read_text())
    assert requirements["spec"]["hole_count"] == state["hole_count"]
    trajectory = json.loads((design_dir / "trajectory.json").read_text())
    assert trajectory["actions"][0]["operation"] == "CREATE_SKETCH"


def test_generate_dataset_hits_target_count(tmp_path):
    stats = generate_dataset(tmp_path, count=2, seed=11)
    assert stats.written == 2
    assert len(list(tmp_path.glob("design_*"))) == 2
    assert (tmp_path / "generation_stats.json").exists()
