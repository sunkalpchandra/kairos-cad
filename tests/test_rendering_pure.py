"""Rasterizer and PNG writer tests on synthetic meshes (no FreeCAD)."""

import numpy as np
import pytest

from kairos.cad.rendering import VIEW_DIRECTIONS, rasterize, read_png, write_png

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


def test_png_round_trips_through_read_png(tmp_path):
    rng = np.random.default_rng(0)
    original = rng.integers(0, 256, size=(23, 17, 3), dtype=np.uint8)
    path = write_png(original, tmp_path / "round.png")
    assert np.array_equal(read_png(path), original)


@pytest.mark.parametrize("filter_type", [0, 1, 2, 3, 4])
def test_read_png_reconstructs_every_row_filter(tmp_path, filter_type):
    """Our writer only emits filter 0, but re-encoded views may use any of them."""
    import struct
    import zlib

    rng = np.random.default_rng(filter_type)
    height, width = 6, 5
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)

    # Encode with the requested filter applied to every row.
    stride = width * 3
    raw = bytearray()
    prior = np.zeros(stride, dtype=np.int16)
    for row in range(height):
        line = image[row].reshape(-1).astype(np.int16)
        encoded = np.zeros(stride, dtype=np.int16)
        for i in range(stride):
            left = line[i - 3] if i >= 3 else 0
            up = prior[i]
            up_left = prior[i - 3] if i >= 3 else 0
            if filter_type == 0:
                pred = 0
            elif filter_type == 1:
                pred = left
            elif filter_type == 2:
                pred = up
            elif filter_type == 3:
                pred = (int(left) + int(up)) >> 1
            else:
                p = int(left) + int(up) - int(up_left)
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
                pred = left if (pa <= pb and pa <= pc) else (up if pb <= pc else up_left)
            encoded[i] = (line[i] - pred) & 0xFF
        raw += bytes([filter_type]) + encoded.astype(np.uint8).tobytes()
        prior = line

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    path = tmp_path / f"filter{filter_type}.png"
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )
    assert np.array_equal(read_png(path), image)
