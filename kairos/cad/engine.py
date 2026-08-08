"""CADEngine: the single facade the action executor drives.

Owns one document, tracks named sketches and features, and exposes the
controlled API surface (sketch, constrain, feature, measure, render, export).
Targets are referenced by *name strings*, sketch object names ('Sketch001')
and subelement names ('Edge7', 'Face3'), never by raw Python objects, so
every call is serializable into a trajectory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kairos.cad import (
    constraints as constraints_api,
)
from kairos.cad import (
    export as export_api,
)
from kairos.cad import (
    features as features_api,
)
from kairos.cad import (
    measurements as measurements_api,
)
from kairos.cad import (
    rendering as rendering_api,
)
from kairos.cad import (
    sketches as sketches_api,
)
from kairos.cad.document import CADDocument
from kairos.cad.errors import CADError, SketchError
from kairos.cad.validation import ValidationReport, validate_document


class CADEngine:
    """A managed CAD session: one document, one body, a controlled API."""

    def __init__(self, name: str = "kairos", material: str = "aluminum") -> None:
        self._doc = CADDocument(name)
        self.material = material
        self._active_sketch_name: str | None = None
        self._feature_names: list[str] = []

    # ------------------------------------------------------------- lifecycle

    @property
    def document(self) -> CADDocument:
        return self._doc

    def close(self) -> None:
        self._doc.close()

    # -------------------------------------------------------------- sketches

    def create_sketch(self, plane: str = "XY", offset: float = 0.0) -> str:
        sketch = sketches_api.create_sketch(self._doc, plane=plane, offset=offset)
        self._active_sketch_name = sketch.Name
        return sketch.Name

    def _sketch(self, name: str | None = None):
        target = name or self._active_sketch_name
        if target is None:
            raise SketchError("no sketch exists; CREATE_SKETCH first")
        obj = self._doc.find_object(target)
        if obj.TypeId != "Sketcher::SketchObject":
            raise SketchError(f"{target!r} is not a sketch (type {obj.TypeId})")
        return obj

    @property
    def active_sketch_name(self) -> str | None:
        return self._active_sketch_name

    def add_line(self, x1, y1, x2, y2, sketch: str | None = None) -> int:
        return sketches_api.add_line(self._sketch(sketch), x1, y1, x2, y2)

    def add_rectangle(self, x, y, width, height, sketch: str | None = None) -> list[int]:
        return sketches_api.add_rectangle(self._sketch(sketch), x, y, width, height)

    def add_circle(self, cx, cy, radius, sketch: str | None = None) -> int:
        return sketches_api.add_circle(self._sketch(sketch), cx, cy, radius)

    def add_arc(self, cx, cy, radius, start_deg, end_deg, sketch: str | None = None) -> int:
        return sketches_api.add_arc(self._sketch(sketch), cx, cy, radius, start_deg, end_deg)

    def add_polygon(self, points, closed: bool = True, sketch: str | None = None) -> list[int]:
        return sketches_api.add_polygon(self._sketch(sketch), points, closed=closed)

    def delete_geometry(self, index: int, sketch: str | None = None) -> None:
        sketches_api.delete_geometry(self._sketch(sketch), index)

    def move_geometry(self, index: int, dx: float, dy: float, sketch: str | None = None) -> None:
        sketches_api.move_geometry(self._sketch(sketch), index, dx, dy)

    def add_constraint(self, kind: str, args: list, sketch: str | None = None) -> int:
        return constraints_api.add_constraint(self._sketch(sketch), kind, *args)

    def sketch_status(self, sketch: str | None = None) -> dict[str, Any]:
        sk = self._sketch(sketch)
        return {
            "name": sk.Name,
            "geometry_count": sketches_api.geometry_count(sk),
            "constraint_count": constraints_api.constraint_count(sk),
            "fully_constrained": constraints_api.is_fully_constrained(sk),
            "degrees_of_freedom": constraints_api.degrees_of_freedom(sk),
        }

    # -------------------------------------------------------------- features

    def _record(self, feature) -> str:
        self._feature_names.append(feature.Name)
        return feature.Name

    def pad(self, length, sketch: str | None = None, reversed_=False, midplane=False) -> str:
        return self._record(
            features_api.pad(self._doc, self._sketch(sketch), length, reversed_, midplane)
        )

    def pocket(
        self, depth=None, sketch: str | None = None, through_all=False, reversed_=False
    ) -> str:
        return self._record(
            features_api.pocket(
                self._doc, self._sketch(sketch), depth, through_all, reversed_
            )
        )

    def revolve(self, angle=360.0, axis="V", sketch: str | None = None) -> str:
        return self._record(
            features_api.revolve(self._doc, self._sketch(sketch), angle, axis)
        )

    def fillet(self, edges: list[str], radius: float) -> str:
        return self._record(features_api.fillet(self._doc, edges, radius))

    def chamfer(self, edges: list[str], size: float) -> str:
        return self._record(features_api.chamfer(self._doc, edges, size))

    def shell(self, faces: list[str], thickness: float) -> str:
        return self._record(features_api.shell(self._doc, faces, thickness))

    def mirror(self, feature_names: list[str], plane: str = "XZ") -> str:
        feats = [self._doc.find_object(n) for n in feature_names]
        return self._record(features_api.mirror(self._doc, feats, plane))

    def linear_pattern(self, feature_names: list[str], axis: str, length: float, count: int) -> str:
        feats = [self._doc.find_object(n) for n in feature_names]
        return self._record(
            features_api.linear_pattern(self._doc, feats, axis, length, count)
        )

    def polar_pattern(self, feature_names: list[str], axis: str, angle: float, count: int) -> str:
        feats = [self._doc.find_object(n) for n in feature_names]
        return self._record(
            features_api.polar_pattern(self._doc, feats, axis, angle, count)
        )

    @property
    def last_feature_name(self) -> str | None:
        return self._feature_names[-1] if self._feature_names else None

    # ------------------------------------------------------------ inspection

    def _shape(self):
        shape = self._doc.tip_shape()
        if shape is None:
            raise CADError("no solid geometry yet")
        return shape

    def has_solid(self) -> bool:
        return self._doc.has_solid()

    def measure_volume(self) -> float:
        return measurements_api.volume(self._shape())

    def measure_surface_area(self) -> float:
        return measurements_api.surface_area(self._shape())

    def measure_mass(self) -> float:
        return measurements_api.mass(self._shape(), self.material)

    def measure_bounding_box(self) -> dict[str, float]:
        return measurements_api.bounding_box(self._shape())

    def measure_distance(self, sub_a: str, sub_b: str) -> float:
        shape = self._shape()
        return measurements_api.distance(
            shape.getElement(sub_a), shape.getElement(sub_b)
        )

    def list_edges(self) -> list[dict[str, Any]]:
        return measurements_api.list_edges(self._shape())

    def list_faces(self) -> list[dict[str, Any]]:
        return measurements_api.list_faces(self._shape())

    def find_edges(self, **kwargs) -> list[str]:
        return measurements_api.find_edges(self._shape(), **kwargs)

    def find_holes(self, diameter: float | None = None) -> list[dict[str, Any]]:
        return measurements_api.find_cylindrical_holes(self._shape(), diameter)

    def check_validity(self) -> ValidationReport:
        return validate_document(self._doc)

    def feature_history(self) -> list[dict[str, Any]]:
        return self._doc.feature_tree()

    def summary(self) -> dict[str, Any]:
        """Numerical engineering state + validity + history, for observations."""
        base: dict[str, Any] = {
            "has_solid": self.has_solid(),
            "feature_history": [f["type"].split("::")[-1] for f in self.feature_history()],
            "active_sketch": self._active_sketch_name,
        }
        if base["has_solid"]:
            base.update(measurements_api.summary(self._shape(), self.material))
            base["valid"] = self.check_validity().is_valid
        else:
            base["valid"] = False
        return base

    # --------------------------------------------------------------- outputs

    def render(self, out_dir: str | Path, views=("iso", "front", "top", "right"), size=512):
        return rendering_api.render_views(self._shape(), out_dir, views=views, size=size)

    def export_step(self, path: str | Path) -> Path:
        return export_api.export_step(self._doc, path)

    def export_stl(self, path: str | Path) -> Path:
        return export_api.export_stl(self._doc, path)

    def save(self, path: str | Path) -> Path:
        return export_api.save_fcstd(self._doc, path)
