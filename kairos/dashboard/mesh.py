"""Tessellate solids into a compact mesh payload for the dashboard's 3D viewer.

This module runs **under FreeCAD's interpreter only**, it opens documents and
touches `Shape`. Everything it emits is plain JSON, so the bundling side
(`bundle.py`, either interpreter) never needs FreeCAD.

Size is the whole design problem here. A dashboard embeds two dozen parts, and
`Shape.tessellate` hands back float64 coordinates that serialize to ~20 chars
each. Two things keep the payload small enough to inline:

  * **Vertex welding.** FreeCAD tessellates face by face, so every vertex on a
    shared edge appears once per adjoining face. Welding on the quantized
    coordinate typically removes a third of them and, more importantly, makes
    the mesh watertight enough for smooth-normal shading.
  * **Quantization.** Coordinates are rounded to `QUANTUM` mm and emitted as
    integers, which the viewer scales back. At 0.01 mm this is far below the
    0.1 mm tolerance every constraint check in this repo uses, so it cannot
    change what the viewer shows about a part that passed or failed.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

#: Coordinate quantum in mm. Well below the 0.1 mm constraint tolerance.
QUANTUM = 0.01

#: Chord error for tessellation, in mm. Coarser than the 0.2 mm used for
#: rasterized observations: the viewer is for looking at, not for measuring.
TOLERANCE = 0.35


def _solids(shape) -> list:
    """PartDesign tips are Compounds; meshing one directly yields nothing."""
    solids = list(getattr(shape, "Solids", []) or [])
    return solids if solids else [shape]


def _degenerate(positions: list[int], a: int, b: int, c: int) -> bool:
    """Whether a triangle has zero area, and so no usable normal.

    Two ways a triangle degenerates. Welding can collapse two corners onto one
    index; and quantization can flatten a thin sliver until its three *distinct*
    vertices are collinear. Both leave the vertex normal undefined, which shades
    as a black shard across the part, so both have to go.

    The cross product runs on the quantized integers, so this is exact, no
    epsilon to tune, and no sensitivity to part scale.
    """
    if a == b or b == c or a == c:
        return True
    ax, ay, az = positions[a * 3], positions[a * 3 + 1], positions[a * 3 + 2]
    bx, by, bz = positions[b * 3], positions[b * 3 + 1], positions[b * 3 + 2]
    cx, cy, cz = positions[c * 3], positions[c * 3 + 1], positions[c * 3 + 2]
    ux, uy, uz = bx - ax, by - ay, bz - az
    vx, vy, vz = cx - ax, cy - ay, cz - az
    return (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) == (0, 0, 0)


def tessellate_shape(shape, tolerance: float = TOLERANCE) -> dict[str, Any]:
    """Weld and quantize a shape into flat `positions`/`indices` arrays.

    Returns a dict with integer `positions` (multiply by `quantum` for mm),
    triangle `indices`, and the `bounds` in mm.

    Raises:
        ValueError: the shape produced no triangles, which means the caller
            handed over an empty or null shape rather than a part.
    """
    scale = 1.0 / QUANTUM
    welded: dict[tuple[int, int, int], int] = {}
    positions: list[int] = []
    indices: list[int] = []

    for solid in _solids(shape):
        points, facets = solid.tessellate(tolerance)
        remap: list[int] = []
        for point in points:
            key = (round(point.x * scale), round(point.y * scale), round(point.z * scale))
            index = welded.get(key)
            if index is None:
                index = len(positions) // 3
                welded[key] = index
                positions.extend(key)
            remap.append(index)
        for facet in facets:
            a, b, c = remap[facet[0]], remap[facet[1]], remap[facet[2]]
            if not _degenerate(positions, a, b, c):
                indices.extend((a, b, c))

    if not indices:
        raise ValueError("tessellation produced no triangles")
    return _payload(positions, indices)


def _payload(positions: list[int], indices: list[int]) -> dict[str, Any]:
    """Assemble the viewer payload from quantized positions and triangles."""
    xs = positions[0::3]
    ys = positions[1::3]
    zs = positions[2::3]
    return {
        "positions": positions,
        "indices": indices,
        "quantum": QUANTUM,
        "vertex_count": len(positions) // 3,
        "triangle_count": len(indices) // 3,
        "bounds": {
            "min": [min(xs) * QUANTUM, min(ys) * QUANTUM, min(zs) * QUANTUM],
            "max": [max(xs) * QUANTUM, max(ys) * QUANTUM, max(zs) * QUANTUM],
        },
    }


#: Binary STL layout: 80-byte header, uint32 count, then 50 bytes per facet
#: (3 normal floats + 9 vertex floats + a uint16 attribute word).
_STL_HEADER = 84
_STL_FACET = 50


def mesh_from_stl(path: str | Path) -> dict[str, Any]:
    """Weld and quantize a binary STL into the same payload as `tessellate_shape`.

    Every generated design already ships `model.stl`, so the dashboard can be
    built entirely under the torch interpreter, no FreeCAD subprocess, and no
    re-meshing of geometry that was already meshed at generation time.

    STL is a triangle soup with no shared vertices at all, which makes the
    welding pass here do more work than it does on FreeCAD tessellation, not
    less: a typical bracket drops from ~3n to ~n vertices.

    Raises:
        ValueError: the file is ASCII STL, truncated, or has no triangles.
    """
    raw = Path(path).read_bytes()
    if raw[:5].lstrip().lower().startswith(b"solid") and b"facet normal" in raw[:512]:
        raise ValueError(f"{path} is ASCII STL; only binary STL is supported")
    if len(raw) < _STL_HEADER:
        raise ValueError(f"{path} is too short to be a binary STL")

    (count,) = struct.unpack_from("<I", raw, 80)
    expected = _STL_HEADER + count * _STL_FACET
    if len(raw) < expected:
        raise ValueError(f"{path} declares {count} facets but holds {len(raw)} bytes")

    scale = 1.0 / QUANTUM
    welded: dict[tuple[int, int, int], int] = {}
    positions: list[int] = []
    indices: list[int] = []

    for facet in range(count):
        # Skip the stored normal: it is redundant with the winding order, and
        # the viewer recomputes smooth normals from the welded mesh anyway.
        offset = _STL_HEADER + facet * _STL_FACET + 12
        corners = struct.unpack_from("<9f", raw, offset)
        triangle: list[int] = []
        for corner in range(3):
            key = (
                round(corners[corner * 3] * scale),
                round(corners[corner * 3 + 1] * scale),
                round(corners[corner * 3 + 2] * scale),
            )
            index = welded.get(key)
            if index is None:
                index = len(positions) // 3
                welded[key] = index
                positions.extend(key)
            triangle.append(index)
        a, b, c = triangle
        if not _degenerate(positions, a, b, c):
            indices.extend((a, b, c))

    if not indices:
        raise ValueError(f"{path} contains no non-degenerate triangles")
    return _payload(positions, indices)


def mesh_from_document(path: str, tolerance: float = TOLERANCE) -> dict[str, Any]:
    """Open an FCStd file and mesh its body tip.

    Raises:
        ValueError: the document has no body tip to mesh.
    """
    import FreeCAD  # noqa: PLC0415 - FreeCAD interpreter only

    document = FreeCAD.openDocument(str(path))
    try:
        tip = None
        for obj in document.Objects:
            if obj.TypeId == "PartDesign::Body" and getattr(obj, "Tip", None) is not None:
                tip = obj.Tip
                break
        if tip is None:
            raise ValueError(f"{path} has no body tip to mesh")
        return tessellate_shape(tip.Shape, tolerance=tolerance)
    finally:
        FreeCAD.closeDocument(document.Name)
