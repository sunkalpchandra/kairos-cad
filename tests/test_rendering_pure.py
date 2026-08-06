"""Rasterizer and PNG writer tests on synthetic meshes (no FreeCAD)."""

import numpy as np
import pytest

from kairos.cad.rendering import VIEW_DIRECTIONS, rasterize, write_png

#: A unit cube as 8 vertices and 12 triangles.
CUBE_VERTICES = np.array(
    [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ],
    dtype=float,
)
CUBE_TRIANGLES = np.array(
    [
        [0, 2, 1], [0, 3, 2],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [1, 2, 6], [1, 6, 5],  # right
        [3, 0, 4], [3, 4, 7],  # left
    ]
)


@pytest.mark.parametrize("view", sorted(VIEW_DIRECTIONS))
def test_rasterize_produces_shaded_pixels(view):
    image = rasterize(CUBE_VERTICES, CUBE_TRIANGLES, view=view, size=96)
    assert image.shape == (96, 96, 3)
    assert image.dtype == np.uint8
    background = np.all(image == 245, axis=2)
    filled = (~background).sum()
    assert filled > 96 * 96 * 0.2, f"{view} view rendered almost nothing"


def test_unknown_view_rejected():
    from kairos.cad.errors import MeasurementError

    with pytest.raises(MeasurementError, match="unknown view"):
        rasterize(CUBE_VERTICES, CUBE_TRIANGLES, view="behind")


def test_write_png_emits_valid_signature(tmp_path):
    image = rasterize(CUBE_VERTICES, CUBE_TRIANGLES, view="iso", size=64)
    path = write_png(image, tmp_path / "cube.png")
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"IHDR" in data[:32] and data.endswith(b"IEND\xaeB`\x82")


def test_front_view_of_cube_is_square():
    image = rasterize(CUBE_VERTICES, CUBE_TRIANGLES, view="front", size=100)
    background = np.all(image == 245, axis=2)
    rows = np.where(~background.all(axis=1))[0]
    cols = np.where(~background.all(axis=0))[0]
    height = rows.max() - rows.min() + 1
    width = cols.max() - cols.min() + 1
    assert abs(height - width) <= 2, "front view of a cube should be square"
