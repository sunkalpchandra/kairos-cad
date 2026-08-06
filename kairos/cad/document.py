"""Document lifecycle management: creation, bodies, recompute, feature tree.

Wraps a single FreeCAD ``App.Document`` and its active ``PartDesign::Body``.
All higher-level modules (sketches, features, measurements) operate through
this wrapper so the engine holds exactly one source of truth for model state.
"""

from __future__ import annotations

from typing import Any

from kairos.cad.backend import load_freecad
from kairos.cad.errors import DocumentError

#: Origin plane roles as exposed by PartDesign bodies.
PLANE_ROLES = {"XY": "XY_Plane", "XZ": "XZ_Plane", "YZ": "YZ_Plane"}

#: Origin axis roles.
AXIS_ROLES = {"X": "X_Axis", "Y": "Y_Axis", "Z": "Z_Axis"}


class CADDocument:
    """A managed FreeCAD document with one active PartDesign body."""

    def __init__(self, name: str = "kairos") -> None:
        self._app = load_freecad()
        try:
            self._doc = self._app.newDocument(name)
        except Exception as err:  # pragma: no cover - App-level failure
            raise DocumentError(f"could not create document {name!r}: {err}") from err
        self._body = None

    # ------------------------------------------------------------------ core

    @property
    def app(self):
        return self._app

    @property
    def doc(self):
        return self._doc

    @property
    def name(self) -> str:
        return self._doc.Name

    def close(self) -> None:
        """Close and discard the underlying document."""
        try:
            self._app.closeDocument(self._doc.Name)
        except Exception:
            pass
        self._doc = None
        self._body = None

    # ------------------------------------------------------------------ body

    @property
    def body(self):
        """The active PartDesign body, created lazily on first access."""
        if self._body is None:
            self._body = self._doc.addObject("PartDesign::Body", "Body")
            self.recompute()
        return self._body

    def has_body(self) -> bool:
        return self._body is not None

    def origin_plane(self, plane: str):
        """Return the body origin plane object for 'XY' | 'XZ' | 'YZ'."""
        role = PLANE_ROLES.get(plane.upper())
        if role is None:
            raise DocumentError(f"unknown plane {plane!r}; expected XY, XZ, or YZ")
        for feature in self.body.Origin.OriginFeatures:
            if getattr(feature, "Role", None) == role:
                return feature
        raise DocumentError(f"origin plane {role} not found in body")

    def origin_axis(self, axis: str):
        """Return the body origin axis object for 'X' | 'Y' | 'Z'."""
        role = AXIS_ROLES.get(axis.upper())
        if role is None:
            raise DocumentError(f"unknown axis {axis!r}; expected X, Y, or Z")
        for feature in self.body.Origin.OriginFeatures:
            if getattr(feature, "Role", None) == role:
                return feature
        raise DocumentError(f"origin axis {role} not found in body")

    # ------------------------------------------------------------- recompute

    def recompute(self) -> list[str]:
        """Recompute the document and return error descriptions (empty = ok)."""
        try:
            self._doc.recompute()
        except Exception as err:
            return [f"recompute raised: {err}"]
        errors = []
        for obj in self._doc.Objects:
            state = getattr(obj, "State", []) or []
            if any(s in ("Invalid", "Error") for s in state):
                errors.append(f"{obj.Name}: state={list(state)}")
        return errors

    # ------------------------------------------------------------- inspection

    def tip_shape(self):
        """Return the current solid shape at the body tip, or None."""
        if self._body is None:
            return None
        tip = getattr(self._body, "Tip", None)
        if tip is None:
            return None
        shape = getattr(tip, "Shape", None)
        if shape is None or shape.isNull():
            return None
        return shape

    def has_solid(self) -> bool:
        shape = self.tip_shape()
        return shape is not None and len(shape.Solids) > 0

    def find_object(self, name: str):
        obj = self._doc.getObject(name)
        if obj is None:
            raise DocumentError(f"no object named {name!r} in document")
        return obj

    def feature_tree(self) -> list[dict[str, Any]]:
        """Ordered summary of the body's feature history.

        Each entry: ``{"name", "type", "label", "is_tip"}``. Sketches are
        listed where they appear in the body group.
        """
        if self._body is None:
            return []
        tip_name = getattr(getattr(self._body, "Tip", None), "Name", None)
        tree = []
        for obj in self._body.Group:
            type_id = obj.TypeId
            if type_id.startswith("App::") and "Origin" in type_id:
                continue
            tree.append(
                {
                    "name": obj.Name,
                    "type": type_id,
                    "label": obj.Label,
                    "is_tip": obj.Name == tip_name,
                }
            )
        return tree

    # ------------------------------------------------------------------ save

    def save(self, path: str) -> None:
        """Save the document as a .FCStd file."""
        try:
            self._doc.saveAs(str(path))
        except Exception as err:
            raise DocumentError(f"could not save document to {path}: {err}") from err
