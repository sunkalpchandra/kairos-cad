"""Observation extraction: one dict snapshot of engine state per step.

The observation is plain JSON data (no FreeCAD objects), so every consumer —
constraint checker, reward tracker, numerical encoder, trajectory files — is
pure-python testable on recorded observations.
"""

from __future__ import annotations

from typing import Any


def observe(engine, wall_thickness: bool = False) -> dict[str, Any]:
    """Snapshot the engine into a JSON-ready observation dict.

    Keys:
        summary: engine.summary() (volume/mass/bbox/topology/validity/...)
        holes: cylindrical hole groups (diameter/axis/axis_point/faces)
        faces: face inventory (surface type, area, normals, cylinder data)
        sketch: active sketch status or None
        edge_count: number of solid edges (0 when no solid)

    ``wall_thickness=True`` adds ``summary["min_wall_thickness_mm"]``. It is
    off by default because the measurement ray-casts against the solid and
    costs far more than the rest of an observation; callers that need the
    manufacturing check (dataset generation, an episode's final state) opt in.
    """
    summary = engine.summary()
    if wall_thickness and summary.get("has_solid"):
        from kairos.evaluation.wall_thickness import measure_min_wall_thickness

        try:
            shape = engine.document.tip_shape()
            measurement = measure_min_wall_thickness(shape)
            if measurement.measured:
                summary["min_wall_thickness_mm"] = measurement.min_thickness_mm
                summary["wall_thickness_rays"] = measurement.rays_hit
        except Exception:
            # A failed measurement must leave the key absent, so the
            # constraint reads `unmeasured` rather than silently passing.
            pass
    observation: dict[str, Any] = {
        "summary": summary,
        "holes": [],
        "faces": [],
        "sketch": None,
        "edge_count": 0,
    }
    if summary.get("has_solid"):
        observation["holes"] = engine.find_holes()
        observation["faces"] = engine.list_faces()
        observation["edge_count"] = len(engine.list_edges())
    if engine.active_sketch_name is not None:
        try:
            observation["sketch"] = engine.sketch_status()
        except Exception:
            observation["sketch"] = None
    return observation
