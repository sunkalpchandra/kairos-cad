"""CAD integration tests: document, sketches, features, transactional rollback."""

import math

import pytest
from conftest import freecad_version

from kairos.cad.errors import FeatureError, SketchError

pytestmark = pytest.mark.cad


def test_document_starts_empty(engine):
    assert not engine.has_solid()
    assert engine.feature_history() == []
    assert engine.active_sketch_name is None


def test_sketch_creation_on_each_plane(engine):
    for plane in ("XY", "XZ", "YZ"):
        name = engine.create_sketch(plane=plane)
        assert engine.active_sketch_name == name
        status = engine.sketch_status()
        assert status["geometry_count"] == 0


def test_rectangle_pad_volume_exact(engine):
    engine.create_sketch("XY")
    engine.add_rectangle(0, 0, 40, 20)
    engine.pad(10)
    assert engine.measure_volume() == pytest.approx(8000.0, rel=1e-9)
    bbox = engine.measure_bounding_box()
    assert bbox["x_len"] == pytest.approx(40)
    assert bbox["y_len"] == pytest.approx(20)
    assert bbox["z_len"] == pytest.approx(10)


def test_circle_pad_volume_matches_cylinder(engine):
    engine.create_sketch("XY")
    engine.add_circle(0, 0, 5)
    engine.pad(8)
    assert engine.measure_volume() == pytest.approx(math.pi * 25 * 8, rel=1e-6)


def test_pocket_removes_hole_volume(engine):
    engine.create_sketch("XY")
    engine.add_rectangle(0, 0, 40, 20)
    engine.pad(10)
    engine.create_sketch("XY", offset=10)
    engine.add_circle(20, 10, 3)
    engine.pocket(through_all=True)
    assert engine.measure_volume() == pytest.approx(8000 - math.pi * 9 * 10, rel=1e-6)


def test_pad_empty_sketch_fails_and_rolls_back(engine):
    engine.create_sketch("XY")
    with pytest.raises(FeatureError):
        engine.pad(10)
    # Document must remain consistent: no solid, no leftover broken feature.
    assert not engine.has_solid()
    types = [f["type"] for f in engine.feature_history()]
    assert "PartDesign::Pad" not in types


def test_pocket_without_solid_fails(engine):
    engine.create_sketch("XY")
    engine.add_circle(0, 0, 4)
    with pytest.raises(FeatureError):
        engine.pocket(through_all=True)


def test_oversized_fillet_fails_and_rolls_back(engine):
    engine.create_sketch("XY")
    engine.add_rectangle(0, 0, 20, 20)
    engine.pad(5)
    volume_before = engine.measure_volume()
    with pytest.raises(FeatureError):
        engine.fillet(["Edge1"], radius=500.0)
    assert engine.measure_volume() == pytest.approx(volume_before)
    types = [f["type"] for f in engine.feature_history()]
    assert "PartDesign::Fillet" not in types


def test_fillet_reduces_volume(engine):
    engine.create_sketch("XY")
    engine.add_rectangle(0, 0, 20, 20)
    engine.pad(10)
    before = engine.measure_volume()
    vertical = engine.find_edges(curve="Line", direction=(0, 0, 1))
    assert len(vertical) == 4
    engine.fillet(vertical[:1], radius=3.0)
    after = engine.measure_volume()
    # A r=3 vertical-edge fillet removes (9 - 9*pi/4) * height.
    assert before - after == pytest.approx((9 - 9 * math.pi / 4) * 10, rel=1e-6)


def test_chamfer_reduces_volume(engine):
    engine.create_sketch("XY")
    engine.add_rectangle(0, 0, 20, 20)
    engine.pad(10)
    before = engine.measure_volume()
    edge = engine.find_edges(curve="Line", direction=(0, 0, 1))[:1]
    engine.chamfer(edge, size=2.0)
    assert before - engine.measure_volume() == pytest.approx(2 * 10, rel=1e-6)


def test_degenerate_sketch_geometry_rejected(engine):
    engine.create_sketch("XY")
    with pytest.raises(SketchError):
        engine.add_line(1, 1, 1, 1)
    with pytest.raises(SketchError):
        engine.add_circle(0, 0, -2)


def test_constraints_and_dof(engine):
    engine.create_sketch("XY")
    engine.add_rectangle(0, 0, 30, 10)
    status = engine.sketch_status()
    # Rectangle helper adds closure + axis alignment constraints.
    assert status["constraint_count"] == 8
    dof = status["degrees_of_freedom"]
    assert dof is None or dof > 0  # position/size still free
    engine.add_constraint("Distance", [0, 30.0])
    assert engine.sketch_status()["constraint_count"] == 9


def test_mirror_doubles_volume(engine):
    engine.create_sketch("XY")
    engine.add_rectangle(0, 0, 20, 10)  # touches the YZ plane so halves connect
    engine.pad(6)
    before = engine.measure_volume()
    pad_name = engine.last_feature_name
    engine.mirror([pad_name], plane="YZ")
    assert engine.measure_volume() == pytest.approx(2 * before, rel=1e-6)
    assert engine.measure_bounding_box()["x_min"] == pytest.approx(-20)


@pytest.mark.skipif(
    freecad_version() < (1, 1),
    reason="FreeCAD below 1.1 discards disjoint pattern instances rather than "
           "keeping them as a multi-solid compound",
)
def test_disjoint_mirror_yields_two_solid_compound(engine):
    # FreeCAD 1.1 permits disjoint pattern results as multi-solid compounds;
    # the summary must expose the solid count so rewards can reason about it.
    # Debian bookworm ships 0.20, where mirror() raises FeatureError instead,
    # which is how CI surfaced this as a portability difference.
    engine.create_sketch("XY")
    engine.add_rectangle(5, 0, 20, 10)  # clear of the YZ plane: halves disjoint
    engine.pad(6)
    engine.mirror([engine.last_feature_name], plane="YZ")
    summary = engine.summary()
    assert summary["volume_mm3"] == pytest.approx(2400.0, rel=1e-6)
    assert summary["topology"]["solids"] == 2
    assert summary["valid"] is True


def test_revolve_produces_solid_of_revolution(engine):
    engine.create_sketch("XZ")
    engine.add_rectangle(2, 0, 3, 8)  # annulus: inner r=2, outer r=5, h=8
    engine.revolve(angle=360.0, axis="V")
    expected = math.pi * (25 - 4) * 8
    assert engine.measure_volume() == pytest.approx(expected, rel=1e-6)
