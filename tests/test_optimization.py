"""Surrogate and optimizer tests (pure numpy, no torch, no FreeCAD)."""

import math

import numpy as np
import pytest

from kairos.optimization import (
    Bounds,
    RidgeSurrogate,
    Sample,
    SurrogateData,
    optimize_design,
    penalized_objective,
    train_surrogate,
    verify_result,
)
from kairos.optimization.optimizer import OptimizationResult


def _truth(w, h, t):
    """Ground truth: mass is a three-way product; thickness IS t."""
    return w * h * t * 0.0027, t


def _data(n=200, seed=0, thickness_none=0):
    rng = np.random.default_rng(seed)
    data = SurrogateData("synthetic", ["w", "h", "t"])
    for i in range(n):
        w, h, t = rng.uniform(20, 80), rng.uniform(20, 60), rng.uniform(1.0, 8.0)
        mass, thickness = _truth(w, h, t)
        data.samples.append(
            Sample({"w": w, "h": h, "t": t}, mass,
                   None if i < thickness_none else thickness, True)
        )
    return data


# ------------------------------------------------------------------ surrogate


def test_unmeasured_rows_are_dropped_not_imputed():
    """A fabricated target would teach a value the geometry never had."""
    data = _data(n=50, thickness_none=10)
    x, y = data.matrix()
    assert len(x) == 40 and len(y) == 40


def test_invalid_rows_are_dropped():
    data = _data(n=20)
    data.samples[0].valid = False
    assert len(data.matrix()[0]) == 19


def test_cubic_surrogate_captures_a_three_way_product():
    """Mass is width x height x thickness; a quadratic cannot represent it."""
    data = _data()
    cubic, metrics = train_surrogate(data, degree=3, seed=0)
    assert metrics.mass_r2 > 0.999
    assert metrics.thickness_r2 > 0.999

    _, quadratic_metrics = train_surrogate(data, degree=2, seed=0)
    assert quadratic_metrics.mass_mae > metrics.mass_mae


def test_surrogate_predicts_one_design():
    model, _ = train_surrogate(_data(), seed=0)
    mass, thickness = model.predict_one({"w": 40.0, "h": 30.0, "t": 5.0})
    true_mass, true_thickness = _truth(40.0, 30.0, 5.0)
    assert mass == pytest.approx(true_mass, rel=0.02)
    assert thickness == pytest.approx(true_thickness, abs=0.05)


def test_surrogate_round_trips_through_json(tmp_path):
    model, _ = train_surrogate(_data(), seed=0)
    path = model.save(tmp_path / "surrogate.json")
    restored = RidgeSurrogate.load(path)
    for point in ({"w": 25.0, "h": 25.0, "t": 3.0}, {"w": 70.0, "h": 55.0, "t": 7.0}):
        assert restored.predict_one(point) == pytest.approx(model.predict_one(point))


def test_dataset_round_trips_through_json(tmp_path):
    data = _data(n=20)
    restored = SurrogateData.load(data.save(tmp_path / "samples.json"))
    assert restored.family == data.family
    assert restored.parameter_names == data.parameter_names
    assert len(restored.samples) == len(data.samples)


def test_too_few_samples_is_an_error_not_a_bad_fit():
    with pytest.raises(ValueError, match="at least 8"):
        train_surrogate(_data(n=5))


def test_a_constant_parameter_does_not_divide_by_zero():
    data = SurrogateData("flat", ["a", "b"])
    rng = np.random.default_rng(0)
    for _ in range(30):
        b = rng.uniform(1, 10)
        data.samples.append(Sample({"a": 5.0, "b": b}, 2.0 * b, b, True))
    model, metrics = train_surrogate(data, seed=0)
    assert math.isfinite(metrics.mass_mae)
    assert math.isfinite(model.predict_one({"a": 5.0, "b": 4.0})[0])


# ------------------------------------------------------------------ optimizer


def test_penalty_is_scale_relative():
    """An additive penalty tuned for grams misprices a millimetre shortfall."""
    assert penalized_objective(100.0, 5.0, 3.0) == pytest.approx(100.0)  # no shortfall
    small = penalized_objective(10.0, 2.9, 3.0)
    large = penalized_objective(1000.0, 2.9, 3.0)
    # The same 0.1 mm shortfall costs proportionally the same at any part size.
    assert small / 10.0 == pytest.approx(large / 1000.0)


def test_penalty_grows_with_the_shortfall():
    mild = penalized_objective(50.0, 2.9, 3.0)
    severe = penalized_objective(50.0, 1.0, 3.0)
    assert 50.0 < mild < severe


def test_optimizer_finds_the_known_optimum():
    """Lightest design is the smallest one whose wall still clears the floor."""
    model, _ = train_surrogate(_data(), seed=0)
    result = optimize_design(
        model,
        Bounds({"w": 20, "h": 20, "t": 1.0}, {"w": 80, "h": 60, "t": 8.0}),
        min_thickness_mm=3.0,
        iterations=30,
        seed=0,
    )
    p = result.parameters
    assert p["t"] == pytest.approx(3.0, abs=0.25)  # sits on the constraint
    assert p["w"] == pytest.approx(20.0, abs=1.5)  # and at the size floor
    ideal = _truth(20, 20, 3.0)[0]
    assert _truth(p["w"], p["h"], p["t"])[0] < ideal * 1.2


def test_optimizer_respects_the_manufacturing_floor():
    model, _ = train_surrogate(_data(), seed=0)
    for floor in (2.0, 4.0, 6.0):
        result = optimize_design(
            model,
            Bounds({"w": 20, "h": 20, "t": 1.0}, {"w": 80, "h": 60, "t": 8.0}),
            min_thickness_mm=floor,
            iterations=25,
            seed=1,
        )
        assert result.parameters["t"] >= floor - 0.3, f"floor {floor}"


def test_infeasible_draws_are_never_scored():
    model, _ = train_surrogate(_data(), seed=0)
    seen: list[dict] = []

    def is_feasible(values):
        seen.append(values)
        return values["t"] >= 5.0

    result = optimize_design(
        model,
        Bounds({"w": 20, "h": 20, "t": 1.0}, {"w": 80, "h": 60, "t": 8.0}),
        min_thickness_mm=3.0,
        is_feasible=is_feasible,
        iterations=10,
        seed=0,
    )
    assert seen, "the guard was never consulted"
    assert result.parameters["t"] >= 5.0
    # Rejected draws cost nothing: fewer evaluations than candidates seen.
    assert result.evaluations < len(seen)


def test_no_feasible_candidate_raises_rather_than_returning_junk():
    model, _ = train_surrogate(_data(), seed=0)
    with pytest.raises(RuntimeError, match="no feasible candidate"):
        optimize_design(
            model,
            Bounds({"w": 20, "h": 20, "t": 1.0}, {"w": 80, "h": 60, "t": 8.0}),
            min_thickness_mm=3.0,
            is_feasible=lambda values: False,
            iterations=3,
            seed=0,
        )


def test_search_improves_over_its_own_first_round():
    model, _ = train_surrogate(_data(), seed=0)
    result = optimize_design(
        model,
        Bounds({"w": 20, "h": 20, "t": 1.0}, {"w": 80, "h": 60, "t": 8.0}),
        min_thickness_mm=3.0,
        iterations=25,
        seed=0,
    )
    assert result.history[-1] < result.history[0]


def test_bounds_are_never_exceeded():
    model, _ = train_surrogate(_data(), seed=0)
    bounds = Bounds({"w": 30, "h": 25, "t": 2.0}, {"w": 50, "h": 40, "t": 6.0})
    p = optimize_design(model, bounds, min_thickness_mm=3.0, iterations=15, seed=2).parameters
    for name, value in p.items():
        assert bounds.lower[name] - 1e-9 <= value <= bounds.upper[name] + 1e-9


# ----------------------------------------------------------------- verification


def test_verification_records_reality_and_exposes_surrogate_error():
    result = OptimizationResult(
        parameters={"w": 20.0, "h": 20.0, "t": 3.0},
        predicted_mass_g=3.0,
        predicted_thickness_mm=3.0,
        baseline_mass_g=10.0,
    )
    verified = verify_result(result, lambda p: (3.3, 3.05, True), min_thickness_mm=3.0)
    assert verified.verified_mass_g == pytest.approx(3.3)
    assert verified.verified_feasible is True
    assert verified.mass_saving_pct == pytest.approx(67.0, abs=0.1)
    # The prediction was 3.0 against a true 3.3: a 9% surrogate error, visible.
    assert verified.surrogate_error_pct == pytest.approx(9.09, abs=0.1)


def test_verification_can_overturn_a_predicted_pass():
    """The surrogate said fine, the geometry says otherwise, reality wins."""
    result = OptimizationResult(
        parameters={"t": 3.0}, predicted_mass_g=5.0, predicted_thickness_mm=3.2
    )
    verified = verify_result(result, lambda p: (5.0, 2.1, True), min_thickness_mm=3.0)
    assert verified.verified_feasible is False


def test_an_unmeasurable_verification_is_not_a_pass():
    result = OptimizationResult(
        parameters={"t": 3.0}, predicted_mass_g=5.0, predicted_thickness_mm=3.2
    )
    verified = verify_result(result, lambda p: (5.0, None, True), min_thickness_mm=3.0)
    assert verified.verified_feasible is False


def test_result_serializes_with_its_derived_numbers():
    result = OptimizationResult(
        parameters={"t": 3.0}, predicted_mass_g=5.0, predicted_thickness_mm=3.2,
        baseline_mass_g=8.0,
    )
    verify_result(result, lambda p: (5.2, 3.1, True), min_thickness_mm=3.0)
    payload = result.to_dict()
    assert payload["mass_saving_pct"] == pytest.approx(35.0, abs=0.1)
    assert payload["surrogate_error_pct"] is not None
    import json

    json.dumps(payload)  # must stay serializable


def test_package_exports_are_all_importable():
    import kairos.optimization as package

    assert package.__all__
    for name in package.__all__:
        assert hasattr(package, name), f"{name} is exported but missing"
