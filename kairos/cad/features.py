"""Parametric PartDesign features: pad, pocket, revolve, dress-ups, patterns.

Every feature builder is transactional: if the recompute fails or produces an
error state, the partially-created feature is removed and the previous body
tip restored, then a ``FeatureError`` is raised. The action executor converts
that into an invalid-action penalty without corrupting the document.
"""

from __future__ import annotations

from kairos.cad.document import CADDocument
from kairos.cad.errors import FeatureError


def _add_feature(
    cad_doc: CADDocument,
    type_id: str,
    name: str,
    configure,
    require_volume_change: bool = False,
) -> object:
    """Create a body feature transactionally.

    Args:
        cad_doc: managed document.
        type_id: FreeCAD type, e.g. 'PartDesign::Pad'.
        name: object base name.
        configure: callback(feature) applying parameters before recompute.
        require_volume_change: reject features that leave the solid volume
            untouched, PartDesign silently drops pattern/mirror instances
            that are disjoint from the base solid, which must surface as a
            failure rather than a successful no-op.
    """
    body = cad_doc.body
    prev_tip = getattr(body, "Tip", None)
    volume_before = None
    if require_volume_change:
        prev_shape = cad_doc.tip_shape()
        volume_before = prev_shape.Volume if prev_shape is not None else 0.0
    try:
        feature = body.newObject(type_id, name)
    except Exception as err:
        raise FeatureError(f"could not create {type_id}: {err}") from err

    try:
        configure(feature)
        errors = cad_doc.recompute()
        shape = getattr(feature, "Shape", None)
        if errors:
            raise FeatureError(f"{name} failed to build: {errors}")
        if shape is None or shape.isNull():
            raise FeatureError(f"{name} produced a null shape")
        if not shape.isValid():
            raise FeatureError(f"{name} produced an invalid shape")
        if (
            require_volume_change
            and volume_before is not None
            and abs(shape.Volume - volume_before) < 1e-6
        ):
            raise FeatureError(
                f"{name} left the solid unchanged, pattern instances are "
                "likely disjoint from the base solid and were discarded"
            )
        # newObject does not advance the tip for transform features
        # (Mirrored/patterns), leaving measurements reading the old solid.
        if getattr(body, "Tip", None) is not feature:
            body.Tip = feature
            cad_doc.recompute()
        return feature
    except Exception as err:
        feature_name = feature.Name
        try:
            if prev_tip is not None:
                body.Tip = prev_tip
            cad_doc.doc.removeObject(feature_name)
            cad_doc.recompute()
        except Exception:
            pass
        if isinstance(err, FeatureError):
            raise
        raise FeatureError(f"{name} failed: {err}") from err


# ------------------------------------------------------------------ additive


def pad(
    cad_doc: CADDocument,
    sketch,
    length: float,
    reversed_: bool = False,
    midplane: bool = False,
    name: str = "Pad",
):
    """Extrude a sketch profile into a solid."""
    if length <= 0:
        raise FeatureError(f"pad length must be positive, got {length}")

    def configure(f):
        f.Profile = sketch
        f.Length = float(length)
        f.Reversed = bool(reversed_)
        f.Midplane = bool(midplane)
        sketch.Visibility = False

    return _add_feature(cad_doc, "PartDesign::Pad", name, configure)


def revolve(
    cad_doc: CADDocument,
    sketch,
    angle: float = 360.0,
    axis: str = "V",
    name: str = "Revolution",
):
    """Revolve a sketch profile around its own V (y) or H (x) axis."""
    if not 0 < angle <= 360:
        raise FeatureError(f"revolve angle must be in (0, 360], got {angle}")
    axis_ref = {"V": "V_Axis", "H": "H_Axis"}.get(axis.upper())
    if axis_ref is None:
        raise FeatureError(f"unknown revolve axis {axis!r}; expected 'V' or 'H'")

    def configure(f):
        f.Profile = sketch
        f.ReferenceAxis = (sketch, [axis_ref])
        f.Angle = float(angle)
        sketch.Visibility = False

    return _add_feature(cad_doc, "PartDesign::Revolution", name, configure)


# --------------------------------------------------------------- subtractive


def pocket(
    cad_doc: CADDocument,
    sketch,
    depth: float | None = None,
    through_all: bool = False,
    reversed_: bool = False,
    name: str = "Pocket",
):
    """Cut a sketch profile into the existing solid."""
    if not through_all and (depth is None or depth <= 0):
        raise FeatureError(f"pocket needs positive depth or through_all, got {depth}")
    if not cad_doc.has_solid():
        raise FeatureError("pocket requires an existing solid")

    def configure(f):
        f.Profile = sketch
        if through_all:
            f.Type = "ThroughAll"
        else:
            f.Type = "Length"
            f.Length = float(depth)
        f.Reversed = bool(reversed_)
        sketch.Visibility = False

    return _add_feature(cad_doc, "PartDesign::Pocket", name, configure)


# ------------------------------------------------------------------ dressups


def fillet(cad_doc: CADDocument, edges: list[str], radius: float, name: str = "Fillet"):
    """Round the given edges ('EdgeN' subelement names of the current tip)."""
    if radius <= 0:
        raise FeatureError(f"fillet radius must be positive, got {radius}")
    if not edges:
        raise FeatureError("fillet requires at least one edge")
    base = getattr(cad_doc.body, "Tip", None)
    if base is None:
        raise FeatureError("fillet requires an existing solid feature")

    def configure(f):
        f.Base = (base, list(edges))
        f.Radius = float(radius)

    return _add_feature(cad_doc, "PartDesign::Fillet", name, configure)


def chamfer(cad_doc: CADDocument, edges: list[str], size: float, name: str = "Chamfer"):
    """Chamfer the given edges of the current tip."""
    if size <= 0:
        raise FeatureError(f"chamfer size must be positive, got {size}")
    if not edges:
        raise FeatureError("chamfer requires at least one edge")
    base = getattr(cad_doc.body, "Tip", None)
    if base is None:
        raise FeatureError("chamfer requires an existing solid feature")

    def configure(f):
        f.Base = (base, list(edges))
        f.Size = float(size)

    return _add_feature(cad_doc, "PartDesign::Chamfer", name, configure)


def shell(cad_doc: CADDocument, faces: list[str], thickness: float, name: str = "Thickness"):
    """Hollow the solid, removing the given faces ('FaceN' names)."""
    if thickness <= 0:
        raise FeatureError(f"shell thickness must be positive, got {thickness}")
    base = getattr(cad_doc.body, "Tip", None)
    if base is None:
        raise FeatureError("shell requires an existing solid feature")

    def configure(f):
        f.Base = (base, list(faces))
        f.Value = float(thickness)

    return _add_feature(cad_doc, "PartDesign::Thickness", name, configure)


# ------------------------------------------------------------------ patterns


def mirror(cad_doc: CADDocument, features: list, plane: str = "XZ", name: str = "Mirrored"):
    """Mirror the given features across a body origin plane."""
    if not features:
        raise FeatureError("mirror requires at least one feature")
    plane_obj = cad_doc.origin_plane(plane)

    def configure(f):
        f.Originals = list(features)
        f.MirrorPlane = (plane_obj, [""])

    return _add_feature(
        cad_doc, "PartDesign::Mirrored", name, configure, require_volume_change=True
    )


def linear_pattern(
    cad_doc: CADDocument,
    features: list,
    axis: str,
    length: float,
    occurrences: int,
    name: str = "LinearPattern",
):
    """Repeat features along a body origin axis over ``length`` mm."""
    if occurrences < 2:
        raise FeatureError(f"linear pattern needs >= 2 occurrences, got {occurrences}")
    if length <= 0:
        raise FeatureError(f"linear pattern length must be positive, got {length}")
    if not features:
        raise FeatureError("linear pattern requires at least one feature")
    axis_obj = cad_doc.origin_axis(axis)

    def configure(f):
        f.Originals = list(features)
        f.Direction = (axis_obj, [""])
        f.Length = float(length)
        f.Occurrences = int(occurrences)

    return _add_feature(
        cad_doc, "PartDesign::LinearPattern", name, configure, require_volume_change=True
    )


def polar_pattern(
    cad_doc: CADDocument,
    features: list,
    axis: str,
    angle: float,
    occurrences: int,
    name: str = "PolarPattern",
):
    """Repeat features around a body origin axis over ``angle`` degrees."""
    if occurrences < 2:
        raise FeatureError(f"polar pattern needs >= 2 occurrences, got {occurrences}")
    if not 0 < angle <= 360:
        raise FeatureError(f"polar pattern angle must be in (0, 360], got {angle}")
    if not features:
        raise FeatureError("polar pattern requires at least one feature")
    axis_obj = cad_doc.origin_axis(axis)

    def configure(f):
        f.Originals = list(features)
        f.Axis = (axis_obj, [""])
        f.Angle = float(angle)
        f.Occurrences = int(occurrences)

    return _add_feature(
        cad_doc, "PartDesign::PolarPattern", name, configure, require_volume_change=True
    )
