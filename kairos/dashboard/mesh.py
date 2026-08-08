"""Tessellate solids into a compact mesh payload for the dashboard's 3D viewer.

This module runs **under FreeCAD's interpreter only** — it opens documents and
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
            # Welding can collapse a sliver triangle onto a line. A degenerate
            # triangle has no normal, so it renders as a black shard.
            if a != b and b != c and a != c:
                indices.extend((a, b, c))

    if not indices:
        raise ValueError("tessellation produced no triangles")

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
