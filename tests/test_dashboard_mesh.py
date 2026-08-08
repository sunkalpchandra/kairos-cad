"""Tests for dashboard mesh extraction.

These run under the system interpreter: the STL path is pure python by design,
so the dashboard can be built without FreeCAD.
"""

import struct

import pytest

from kairos.dashboard.mesh import QUANTUM, mesh_from_stl


def _binary_stl(triangles: list[tuple[tuple[float, float, float], ...]]) -> bytes:
    out = bytearray(b"\0" * 80)
    out += struct.pack("<I", len(triangles))
    for tri in triangles:
        out += struct.pack("<3f", 0.0, 0.0, 1.0)  # normal, ignored by the reader
        for vertex in tri:
            out += struct.pack("<3f", *vertex)
        out += struct.pack("<H", 0)
    return bytes(out)


def _square() -> bytes:
    """Two triangles sharing an edge: 6 raw corners, 4 distinct vertices."""
    a, b, c, d = (0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0), (0.0, 10.0, 0.0)
    return _binary_stl([(a, b, c), (a, c, d)])


def test_welding_collapses_the_shared_edge(tmp_path):
    path = tmp_path / "square.stl"
    path.write_bytes(_square())
    mesh = mesh_from_stl(path)
    assert mesh["triangle_count"] == 2
    assert mesh["vertex_count"] == 4  # not 6: the shared edge is welded


def test_bounds_are_reported_in_mm(tmp_path):
    path = tmp_path / "square.stl"
    path.write_bytes(_square())
    mesh = mesh_from_stl(path)
    assert mesh["bounds"]["min"] == [0.0, 0.0, 0.0]
    assert mesh["bounds"]["max"] == pytest.approx([10.0, 10.0, 0.0])


def test_positions_are_integers_scaled_by_the_quantum(tmp_path):
    path = tmp_path / "square.stl"
    path.write_bytes(_square())
    mesh = mesh_from_stl(path)
    assert all(isinstance(value, int) for value in mesh["positions"])
    assert max(mesh["positions"]) * QUANTUM == pytest.approx(10.0)


def test_degenerate_triangles_are_dropped(tmp_path):
    """A sliver that quantization collapses onto a line has no normal.

    Keeping it would render a black shard across the part.
    """
    flat = ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (5.0, 1e-6, 0.0))
    path = tmp_path / "sliver.stl"
    path.write_bytes(_binary_stl([flat]))
    with pytest.raises(ValueError, match="no non-degenerate triangles"):
        mesh_from_stl(path)


def test_ascii_stl_is_rejected_rather_than_misread(tmp_path):
    """Reading ASCII bytes as binary yields a huge facet count and garbage."""
    path = tmp_path / "ascii.stl"
    path.write_text("solid part\n facet normal 0 0 1\n" + "x" * 600)
    with pytest.raises(ValueError, match="ASCII STL"):
        mesh_from_stl(path)


def test_truncated_stl_is_rejected(tmp_path):
    path = tmp_path / "short.stl"
    path.write_bytes(_square()[:-20])
    with pytest.raises(ValueError, match="declares"):
        mesh_from_stl(path)


def test_indices_stay_inside_the_position_array(tmp_path):
    path = tmp_path / "square.stl"
    path.write_bytes(_square())
    mesh = mesh_from_stl(path)
    assert max(mesh["indices"]) < mesh["vertex_count"]
    assert len(mesh["indices"]) % 3 == 0
