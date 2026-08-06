"""CAD integration tests: measurements, validation, boolean ops, export, render."""

import math

import pytest

from kairos.cad import boolean, measurements
from kairos.cad.errors import GeometryInvalidError

pytestmark = pytest.mark.cad


@pytest.fixture
def plate(engine):
    """A 40x20x10 plate with one 6mm through-hole at (20, 10)."""
    engine.create_sketch("XY")
    engine.add_rectangle(0, 0, 40, 20)
    engine.pad(10)
    engine.create_sketch("XY", offset=10)
    engine.add_circle(20, 10, 3)
    engine.pocket(through_all=True)
    return engine


def test_summary_contains_engineering_state(plate):
    summary = plate.summary()
    assert summary["valid"] is True
    assert summary["hole_count"] == 1
    assert summary["mass_g"] == pytest.approx(
        summary["volume_mm3"] * 2.70e-3, rel=1e-9
    )
    assert summary["topology"]["solids"] == 1
    assert "Pad" in summary["feature_history"]


def test_hole_detection_diameter_filter(plate):
    assert len(plate.find_holes(diameter=6.0)) == 1
    assert plate.find_holes(diameter=6.0)[0]["diameter"] == pytest.approx(6.0)
    assert plate.find_holes(diameter=3.0) == []


def test_find_edges_by_direction_and_position(plate):
    vertical = plate.find_edges(curve="Line", direction=(0, 0, 1))
    assert len(vertical) == 4
    near_origin = plate.find_edges(
        curve="Line", direction=(0, 0, 1), near=(0, 0, 5), near_tol=0.5
    )
    assert len(near_origin) == 1


def test_validation_report_structure(plate):
    report = plate.check_validity()
    assert report.is_valid
    assert report.checks["kernel_valid"]
    assert report.checks["closed_shells"]
    assert report.issues == []
    assert report.to_dict()["is_valid"] is True


def test_mass_material_lookup(plate):
    shape = plate.document.tip_shape()
    aluminum = measurements.mass(shape, "aluminum")
    steel = measurements.mass(shape, "steel")
    assert steel / aluminum == pytest.approx(7.85 / 2.70, rel=1e-9)
    with pytest.raises(Exception):
        measurements.mass(shape, "unobtainium")


def test_boolean_union_cut_intersection(engine):
    import Part
    from FreeCAD import Vector

    a = Part.makeBox(10, 10, 10)
    b = Part.makeBox(10, 10, 10, Vector(5, 0, 0))
    assert boolean.union(a, b).Volume == pytest.approx(1500.0, rel=1e-9)
    assert boolean.cut(a, b).Volume == pytest.approx(500.0, rel=1e-9)
    assert boolean.intersection(a, b).Volume == pytest.approx(500.0, rel=1e-9)


def test_boolean_empty_intersection_raises(engine):
    import Part
    from FreeCAD import Vector

    a = Part.makeBox(5, 5, 5)
    b = Part.makeBox(5, 5, 5, Vector(50, 50, 50))
    with pytest.raises(GeometryInvalidError, match="empty"):
        boolean.intersection(a, b)


def test_exports_produce_parseable_files(plate, tmp_path):
    step = plate.export_step(tmp_path / "model")
    stl = plate.export_stl(tmp_path / "model")
    fcstd = plate.save(tmp_path / "model")
    assert step.stat().st_size > 500
    assert b"ISO-10303" in step.read_bytes()[:200]
    assert stl.stat().st_size > 100
    assert fcstd.stat().st_size > 1000
    # Re-import the STEP and confirm identical volume.
    import Part

    reimported = Part.Shape()
    reimported.read(str(step))
    assert reimported.Volume == pytest.approx(8000 - math.pi * 9 * 10, rel=1e-4)


def test_render_views_written(plate, tmp_path):
    paths = plate.render(tmp_path, size=96)
    assert set(paths) == {"iso", "front", "top", "right"}
    for path in paths.values():
        data = path.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
