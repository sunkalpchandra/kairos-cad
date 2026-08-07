"""Pure-python tests of the constraint checker on synthetic observations."""


from kairos.evaluation.constraints import check_constraints
from kairos.language import parse_requirement
from kairos.language.spec import Constraint, EngineeringSpec


def _obs(holes=(), faces=(), bbox=None, mass=None):
    summary = {"has_solid": bool(bbox), "valid": bool(bbox)}
    if bbox:
        summary["bounding_box"] = {
            "x_len": bbox[0], "y_len": bbox[1], "z_len": bbox[2],
        }
    if mass is not None:
        summary["mass_g"] = mass
    return {"summary": summary, "holes": list(holes), "faces": list(faces), "sketch": None}


def _hole(d, at=(0, 0, 0), axis=(0, 0, 1)):
    return {"diameter": d, "axis": axis, "axis_point": list(at), "faces": ["FaceX"]}


def test_hole_count_uses_spec_diameter():
    """hole_count is a TOTAL: a requirement's stated count spans size groups.

    A flange's "12 mm central bore, and 6 bolt holes of 5 mm" is seven holes,
    so filtering by the nominal diameter would find six and call a correct part
    violated. Here the part has three holes where two were asked for.
    """
    spec = parse_requirement("Bracket with 2 x M5 holes")
    obs = _obs(holes=[_hole(5.0), _hole(5.0), _hole(8.0)], bbox=(10, 10, 10))
    report = check_constraints(obs, spec)
    by_kind = {r.constraint.kind: r for r in report.results}
    assert by_kind["hole_count"].status == "violated"  # 3 holes, 2 requested
    # An unmentioned extra bore is not a diameter violation: parts legitimately
    # carry bores the requirement never named (a flange's central bore beside
    # its bolt holes). Wrong *totals* are hole_count's job, checked above.
    assert by_kind["hole_diameter"].status == "satisfied"


def test_hole_diameter_only_requires_the_named_size_to_be_present():
    """The total is hole_count's job; this asks whether d=5 exists at all."""
    spec = parse_requirement("Bracket with 4 x M5 holes")
    obs = _obs(holes=[_hole(5.0), _hole(5.0), _hole(8.0), _hole(8.0)], bbox=(10, 10, 10))
    by_kind = {r.constraint.kind: r for r in check_constraints(obs, spec).results}
    assert by_kind["hole_diameter"].status == "satisfied"
    assert by_kind["hole_count"].status == "satisfied"  # 4 holes total


def test_hole_diameter_violated_when_the_named_size_is_absent():
    spec = parse_requirement("Bracket with 4 x M5 holes")
    obs = _obs(holes=[_hole(8.0)] * 4, bbox=(10, 10, 10))
    by_kind = {r.constraint.kind: r for r in check_constraints(obs, spec).results}
    assert by_kind["hole_diameter"].status == "violated"


def test_hole_count_violation():
    spec = parse_requirement("Plate with 4 M5 holes")
    obs = _obs(holes=[_hole(5.0)], bbox=(10, 10, 10))
    report = check_constraints(obs, spec)
    assert any(r.status == "violated" and r.constraint.kind == "hole_count"
               for r in report.results)
    assert report.satisfaction_rate < 1.0


def test_bbox_exact_orientation_invariant():
    spec = EngineeringSpec(constraints=[Constraint("bounding_box_exact", [60, 40, 5], 1.0)])
    assert check_constraints(_obs(bbox=(5, 60, 40)), spec).all_measured_satisfied
    assert not check_constraints(_obs(bbox=(5, 60, 48)), spec).all_measured_satisfied


def test_bbox_max_upper_bound():
    spec = EngineeringSpec(constraints=[Constraint("bounding_box_max", [80, 60, 20])])
    assert check_constraints(_obs(bbox=(20, 79, 59)), spec).all_measured_satisfied
    assert not check_constraints(_obs(bbox=(20, 81, 59)), spec).all_measured_satisfied


def test_mounting_angle_from_planar_faces():
    spec = EngineeringSpec(constraints=[Constraint("mounting_angle", 90.0, 1.0)])
    faces = [
        {"surface": "Plane", "area": 100.0, "normal": (0, 0, 1)},
        {"surface": "Plane", "area": 90.0, "normal": (1, 0, 0)},
    ]
    assert check_constraints(_obs(faces=faces, bbox=(1, 1, 1)), spec).all_measured_satisfied
    slanted = [
        {"surface": "Plane", "area": 100.0, "normal": (0, 0, 1)},
        {"surface": "Plane", "area": 90.0, "normal": (0.707, 0, 0.707)},
    ]
    assert not check_constraints(_obs(faces=slanted, bbox=(1, 1, 1)), spec).all_measured_satisfied


def test_cylindrical_interface_counts_convex_groups():
    spec = EngineeringSpec(
        constraints=[Constraint("cylindrical_interface", {"count": 2, "diameter": 12.0}, 0.1)]
    )
    # Two coaxial convex faces (one shaft split into halves) + one distinct shaft.
    faces = [
        {"surface": "Cylinder", "area": 10, "radius": 6.0, "concave": False,
         "axis": (0, 0, 1), "axis_point": (0, 0, 0)},
        {"surface": "Cylinder", "area": 10, "radius": 6.0, "concave": False,
         "axis": (0, 0, 1), "axis_point": (0, 0, 5)},
        {"surface": "Cylinder", "area": 10, "radius": 6.0, "concave": False,
         "axis": (0, 0, 1), "axis_point": (30, 0, 0)},
        # A concave bore must not count as an interface.
        {"surface": "Cylinder", "area": 10, "radius": 6.0, "concave": True,
         "axis": (0, 0, 1), "axis_point": (60, 0, 0)},
    ]
    report = check_constraints(_obs(faces=faces, bbox=(1, 1, 1)), spec)
    assert report.results[0].measured == 2
    assert report.all_measured_satisfied


def test_mass_reduction_requires_context_baseline():
    spec = EngineeringSpec(constraints=[Constraint("mass_reduction_pct", 20.0)])
    obs = _obs(bbox=(1, 1, 1), mass=75.0)
    assert check_constraints(obs, spec).unmeasured  # no baseline
    ok = check_constraints(obs, spec, context={"initial_mass_g": 100.0})
    assert ok.all_measured_satisfied  # 25% >= 20%
    short = check_constraints(obs, spec, context={"initial_mass_g": 90.0})
    assert not short.all_measured_satisfied  # 16.7% < 20%


def test_hole_positions_preserved():
    spec = EngineeringSpec(
        constraints=[Constraint("hole_positions_preserved", [[0, 0, 0], [10, 0, 0]], 0.5)]
    )
    obs = _obs(holes=[_hole(5.0, at=(0, 0, 0)), _hole(5.0, at=(10.1, 0, 0))], bbox=(1, 1, 1))
    assert check_constraints(obs, spec).all_measured_satisfied
    moved = _obs(holes=[_hole(5.0, at=(0, 0, 0)), _hole(5.0, at=(12, 0, 0))], bbox=(1, 1, 1))
    assert not check_constraints(moved, spec).all_measured_satisfied


def test_unmeasured_kinds_excluded_from_rate_and_success():
    spec = parse_requirement("Bracket with minimum wall thickness: 3 mm")
    report = check_constraints(_obs(bbox=(1, 1, 1)), spec)
    assert report.unmeasured and not report.violated
    assert report.satisfaction_rate == 1.0  # nothing measured
    assert not report.all_measured_satisfied  # but no credit either


def test_empty_spec_trivially_satisfied():
    report = check_constraints(_obs(bbox=(1, 1, 1)), EngineeringSpec())
    assert report.all_measured_satisfied
    assert report.satisfaction_rate == 1.0
