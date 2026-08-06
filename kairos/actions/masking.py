"""Action masking: which operations are legal in a given CAD state.

Masking is computed from cheap boolean state flags so it can run every step
without touching geometry, and it is pure logic — unit-testable without
FreeCAD. The RL environment multiplies the policy's operation logits by this
mask, collapsing the search space.
"""

from __future__ import annotations

from dataclasses import dataclass

from kairos.actions.schema import (
    BOOLEAN_OPS,
    CONSTRAINT_OPS,
    Operation,
)


@dataclass(frozen=True)
class StateFlags:
    """Cheap summary of engine state sufficient to decide legality."""

    has_sketch: bool = False
    sketch_has_geometry: bool = False
    has_solid: bool = False
    has_edges: bool = False
    has_faces: bool = False
    has_features: bool = False
    body_count: int = 1


#: Operations legal even in an empty document.
_ALWAYS = {Operation.CREATE_SKETCH, Operation.CHECK_VALIDITY, Operation.FINISH_DESIGN}

#: Sketch ops that need existing sketch geometry (not just a sketch).
_NEEDS_SKETCH_GEOMETRY = {Operation.DELETE_GEOMETRY, Operation.MOVE_GEOMETRY} | CONSTRAINT_OPS

#: Sketch ops that only need an open sketch.
_NEEDS_SKETCH = {
    Operation.ADD_LINE,
    Operation.ADD_RECTANGLE,
    Operation.ADD_CIRCLE,
    Operation.ADD_ARC,
    Operation.ADD_POLYGON,
}

#: Profile features consuming the active sketch.
_NEEDS_PROFILE = {Operation.PAD, Operation.REVOLVE}

#: Features that additionally require an existing solid.
_NEEDS_SOLID_AND_PROFILE = {Operation.POCKET}

_NEEDS_EDGES = {Operation.FILLET, Operation.CHAMFER}
_NEEDS_FACES = {Operation.SHELL}
_NEEDS_FEATURES = {
    Operation.MIRROR,
    Operation.LINEAR_PATTERN,
    Operation.CIRCULAR_PATTERN,
}
_NEEDS_SOLID_INSPECTION = {
    Operation.MEASURE_DISTANCE,
    Operation.MEASURE_VOLUME,
    Operation.MEASURE_AREA,
    Operation.MEASURE_BOUNDING_BOX,
    Operation.RENDER_VIEW,
}


def legal_operations(flags: StateFlags) -> set[Operation]:
    """Return the set of operations that may legally be attempted."""
    legal = set(_ALWAYS)
    if flags.has_sketch:
        legal |= _NEEDS_SKETCH
    if flags.has_sketch and flags.sketch_has_geometry:
        legal |= _NEEDS_SKETCH_GEOMETRY
        legal |= _NEEDS_PROFILE
    if flags.has_sketch and flags.sketch_has_geometry and flags.has_solid:
        legal |= _NEEDS_SOLID_AND_PROFILE
    if flags.has_solid and flags.has_edges:
        legal |= _NEEDS_EDGES
    if flags.has_solid and flags.has_faces:
        legal |= _NEEDS_FACES
    if flags.has_solid and flags.has_features:
        legal |= _NEEDS_FEATURES
    if flags.has_solid:
        legal |= _NEEDS_SOLID_INSPECTION
    if flags.body_count >= 2:
        legal |= BOOLEAN_OPS
    return legal


def operation_mask(flags: StateFlags, ordering: list[Operation] | None = None) -> list[bool]:
    """Boolean mask over an operation ordering (default: enum order)."""
    ordering = ordering or list(Operation)
    legal = legal_operations(flags)
    return [op in legal for op in ordering]


def flags_from_engine(engine) -> StateFlags:
    """Compute StateFlags from a live CADEngine."""
    has_sketch = engine.active_sketch_name is not None
    sketch_has_geometry = False
    if has_sketch:
        try:
            sketch_has_geometry = engine.sketch_status()["geometry_count"] > 0
        except Exception:
            has_sketch = False
    has_solid = engine.has_solid()
    counts = {"edges": 0, "faces": 0}
    if has_solid:
        try:
            counts["edges"] = len(engine.list_edges())
            counts["faces"] = len(engine.list_faces())
        except Exception:
            pass
    has_features = engine.last_feature_name is not None
    return StateFlags(
        has_sketch=has_sketch,
        sketch_has_geometry=sketch_has_geometry,
        has_solid=has_solid,
        has_edges=counts["edges"] > 0,
        has_faces=counts["faces"] > 0,
        has_features=has_features,
        body_count=1,
    )
