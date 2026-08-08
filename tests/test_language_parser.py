"""Pure-python tests of the requirement parser (no FreeCAD)."""

from kairos.language import EngineeringSpec, parse_requirement


def test_demo_requirement_full_extraction():
    spec = parse_requirement(
        "Create an L-bracket with:\n"
        "- 4 M5 mounting holes\n"
        "- 3 mm minimum wall thickness\n"
        "- 90 degree angle\n"
        "- symmetric hole placement\n"
        "- minimum possible mass"
    )
    assert spec.hole_count == 4
    assert spec.hole_diameter == 5.0
    assert spec.min_wall_thickness == 3.0
    assert spec.value("mounting_angle") == 90.0
    assert spec.value("symmetry") == "XZ"
    assert spec.has_objective("minimize_mass")


def test_plate_requirement():
    spec = parse_requirement(
        "Design a rectangular mounting plate 60 x 43 x 3.7 mm with 3 "
        "through-holes of 8 mm diameter. Minimize mass."
    )
    assert spec.hole_count == 3
    assert spec.hole_diameter == 8.0
    assert spec.value("bounding_box_exact") == [60.0, 43.0, 3.7]
    assert spec.has_objective("minimize_mass")


def test_metric_thread_table():
    for thread, diameter in (("M3", 3.0), ("M6", 6.0), ("M8", 8.0)):
        spec = parse_requirement(f"Bracket with 2 x {thread} holes")
        assert spec.hole_count == 2
        assert spec.hole_diameter == diameter


def test_envelope_upper_bound():
    spec = parse_requirement("The part must fit within 80 x 60 x 20 mm.")
    assert spec.value("bounding_box_max") == [80.0, 60.0, 20.0]
    assert spec.get("bounding_box_exact") is None


def test_cylindrical_interfaces_and_symmetry_plane():
    spec = parse_requirement(
        "Connector with two mounting faces; 2 cylindrical interfaces of 12 mm "
        "diameter; symmetry about the YZ plane."
    )
    interface = spec.value("cylindrical_interface")
    assert interface == {"count": 2, "diameter": 12.0}
    assert spec.value("symmetry") == "YZ"


def test_mass_reduction_task():
    spec = parse_requirement(
        "Reduce mass by at least 20% while maintaining all mounting interfaces "
        "and minimum wall thickness."
    )
    assert spec.value("mass_reduction_pct") == 20.0
    assert spec.has_objective("minimize_mass")


def test_material_and_tolerance():
    spec = parse_requirement("Steel spacer, tolerance: 0.1 mm, minimize volume.")
    assert spec.material == "steel"
    assert spec.tolerance == 0.1
    assert spec.has_objective("minimize_volume")


def test_no_invented_values():
    spec = parse_requirement("Make something nice.")
    assert spec.constraints == []
    assert spec.objectives == []
    assert spec.material is None


def test_thread_designation_is_not_read_as_a_hole_count():
    """"M4" states a diameter; the 4 must not become a quantity."""
    spec = parse_requirement("Design a bracket with M4 mounting holes on the base.")
    assert spec.hole_count is None
    assert spec.hole_diameter == 4.0


def test_sub_component_dimensions_are_not_the_part_envelope():
    """A triple that sizes a sub-component the design then stacks material on
    is left unextracted, the finished part is meant to be taller than it."""
    reinforced = parse_requirement(
        "Design a reinforced rectangular plate 100 x 60 x 6.0 mm stiffened by 2 "
        "full-length ribs 8 mm wide and 8 mm tall, with 4 corner through-holes "
        "of 6 mm diameter. Minimize mass."
    )
    assert reinforced.get("bounding_box_exact") is None

    braced = parse_requirement(
        "Design a support bracket with a 90 x 50 x 8.0 mm base plate and a 5.0 mm "
        "thick vertical wall 50 mm tall, braced by a 20 mm triangular rib."
    )
    assert braced.get("bounding_box_exact") is None

    # A plate really is its stated block, so that envelope still parses.
    plain = parse_requirement(
        "Design a rectangular mounting plate 100 x 60 x 6.0 mm with 12 "
        "through-holes of 5 mm diameter in a 4x3 grid."
    )
    assert plain.get("bounding_box_exact").value == [100.0, 60.0, 6.0]


def test_round_trip_serialization():
    spec = parse_requirement("Plate with 4 M5 holes, minimum wall thickness: 3 mm.")
    restored = EngineeringSpec.from_dict(spec.to_dict())
    assert restored.to_dict() == spec.to_dict()
    assert restored.hole_count == 4
