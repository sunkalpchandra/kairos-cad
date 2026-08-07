"""Headless rendering of solids to PNG.

FreeCAD's own viewport requires the GUI, so this module implements a small
software rasterizer: tessellate the shape, orthographically project it for a
named view, z-buffer the triangles with Lambert shading, and write a PNG
(stdlib zlib only — usable from any FreeCAD-bundled interpreter).

Standard views: ``iso``, ``front``, ``top``, ``right``.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from kairos.cad.errors import MeasurementError

#: View direction the camera looks *along* (world coordinates, Z up).
VIEW_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "front": (0.0, 1.0, 0.0),
    "top": (0.0, 0.0, -1.0),
    "right": (-1.0, 0.0, 0.0),
    "iso": (-1.0, 1.0, -1.0),
}

_WORLD_UP = np.array([0.0, 0.0, 1.0])


def tessellate(shape, tolerance: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Tessellate a shape into (vertices [n,3], triangles [m,3] int)."""
    if shape is None or shape.isNull():
        raise MeasurementError("cannot tessellate a null shape")
    points, facets = shape.tessellate(tolerance)
    if not facets:
        raise MeasurementError("tessellation produced no triangles")
    vertices = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float64)
    triangles = np.array(facets, dtype=np.int64)
    return vertices, triangles


def _view_basis(view: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (right, up, forward) unit vectors for a named view."""
    try:
        forward = np.array(VIEW_DIRECTIONS[view], dtype=np.float64)
    except KeyError:
        raise MeasurementError(
            f"unknown view {view!r}; expected one of {sorted(VIEW_DIRECTIONS)}"
        ) from None
    forward /= np.linalg.norm(forward)
    up_hint = _WORLD_UP if abs(forward @ _WORLD_UP) < 0.99 else np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return right, up, forward


def rasterize(
    vertices: np.ndarray,
    triangles: np.ndarray,
    view: str = "iso",
    size: int = 512,
    margin: float = 0.08,
) -> np.ndarray:
    """Render triangles to an RGB uint8 image [size, size, 3]."""
    right, up, forward = _view_basis(view)
    # Camera coordinates: x to the right, y up, z into the screen.
    cam = np.stack(
        [vertices @ right, vertices @ up, vertices @ forward], axis=1
    )
    xy_min = cam[:, :2].min(axis=0)
    xy_max = cam[:, :2].max(axis=0)
    extent = float(max(xy_max - xy_min)) or 1.0
    scale = size * (1 - 2 * margin) / extent
    center = (xy_min + xy_max) / 2.0

    px = (cam[:, 0] - center[0]) * scale + size / 2.0
    py = size / 2.0 - (cam[:, 1] - center[1]) * scale
    depth_v = cam[:, 2]

    image = np.full((size, size, 3), 245, dtype=np.uint8)
    zbuf = np.full((size, size), np.inf)

    light = np.array([-0.4, 0.5, -0.75])
    light /= np.linalg.norm(light)
    base_color = np.array([120, 144, 180], dtype=np.float64)

    tri_xy = np.stack([px[triangles], py[triangles]], axis=2)  # [m,3,2]
    tri_z = depth_v[triangles]  # [m,3]

    # World-space triangle normals for shading.
    v0, v1, v2 = (vertices[triangles[:, i]] for i in range(3))
    normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid][:, None]

    for t in range(len(triangles)):
        if not valid[t]:
            continue
        pts = tri_xy[t]
        x_min = max(int(np.floor(pts[:, 0].min())), 0)
        x_max = min(int(np.ceil(pts[:, 0].max())), size - 1)
        y_min = max(int(np.floor(pts[:, 1].min())), 0)
        y_max = min(int(np.ceil(pts[:, 1].max())), size - 1)
        if x_min > x_max or y_min > y_max:
            continue
        xs = np.arange(x_min, x_max + 1) + 0.5
        ys = np.arange(y_min, y_max + 1) + 0.5
        gx, gy = np.meshgrid(xs, ys)

        (x0, y0), (x1, y1), (x2, y2) = pts
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue
        w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w0 * tri_z[t, 0] + w1 * tri_z[t, 1] + w2 * tri_z[t, 2]

        region_z = zbuf[y_min : y_max + 1, x_min : x_max + 1]
        win = inside & (z < region_z)
        if not win.any():
            continue
        intensity = 0.30 + 0.70 * abs(float(normals[t] @ light))
        shade = np.clip(base_color * intensity, 0, 255).astype(np.uint8)
        region_z[win] = z[win]
        image[y_min : y_max + 1, x_min : x_max + 1][win] = shade
    return image


def write_png(image: np.ndarray, path: str | Path) -> Path:
    """Write an RGB uint8 array as a PNG using stdlib zlib only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _ = image.shape
    # Prepend filter byte 0 to each row.
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(payload)
    return path


def read_png(path: str | Path) -> np.ndarray:
    """Read an 8-bit RGB PNG back into an ``[H, W, 3]`` uint8 array.

    The inverse of :func:`write_png`, kept dependency-free for the same reason:
    the training stack must be able to load rendered views without pulling an
    image library into FreeCAD's interpreter. All five PNG row filters are
    reconstructed, so re-encoded files load too — not just our own output.
    """
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")

    pos = 8
    header: tuple[int, ...] | None = None
    compressed = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        pos += 12 + length  # length + tag + payload + crc
        if tag == b"IHDR":
            header = struct.unpack(">IIBBBBB", payload)
        elif tag == b"IDAT":
            compressed += payload
        elif tag == b"IEND":
            break

    if header is None:
        raise ValueError(f"{path} has no IHDR chunk")
    width, height, depth, color_type = header[0], header[1], header[2], header[3]
    if depth != 8 or color_type != 2:
        raise ValueError(f"{path}: only 8-bit RGB PNGs are supported (got {depth}/{color_type})")

    raw = zlib.decompress(bytes(compressed))
    stride = width * 3
    out = np.zeros((height, stride), dtype=np.uint8)
    prior = np.zeros(stride, dtype=np.uint8)

    for row in range(height):
        start = row * (stride + 1)
        filter_type = raw[start]
        line = np.frombuffer(raw[start + 1 : start + 1 + stride], dtype=np.uint8).astype(np.int16)
        if filter_type == 0:
            recon = line
        elif filter_type == 1:  # Sub
            recon = line.copy()
            for i in range(3, stride):
                recon[i] = (recon[i] + recon[i - 3]) & 0xFF
        elif filter_type == 2:  # Up
            recon = (line + prior) & 0xFF
        elif filter_type == 3:  # Average
            recon = line.copy()
            for i in range(stride):
                left = recon[i - 3] if i >= 3 else 0
                recon[i] = (recon[i] + ((left + int(prior[i])) >> 1)) & 0xFF
        elif filter_type == 4:  # Paeth
            recon = line.copy()
            for i in range(stride):
                left = int(recon[i - 3]) if i >= 3 else 0
                up = int(prior[i])
                up_left = int(prior[i - 3]) if i >= 3 else 0
                p = left + up - up_left
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
                pred = left if (pa <= pb and pa <= pc) else (up if pb <= pc else up_left)
                recon[i] = (recon[i] + pred) & 0xFF
        else:
            raise ValueError(f"{path}: unknown PNG row filter {filter_type}")
        out[row] = prior = recon.astype(np.uint8)

    return out.reshape(height, width, 3)


def render_views(
    shape,
    out_dir: str | Path,
    views: tuple[str, ...] = ("iso", "front", "top", "right"),
    size: int = 512,
    tolerance: float = 0.2,
) -> dict[str, Path]:
    """Render the standard observation views; returns {view: png_path}."""
    vertices, triangles = tessellate(shape, tolerance)
    out_dir = Path(out_dir)
    paths = {}
    for view in views:
        image = rasterize(vertices, triangles, view=view, size=size)
        paths[view] = write_png(image, out_dir / f"{view}.png")
    return paths
