"""Export solids to STEP / STL and documents to FCStd."""

from __future__ import annotations

from pathlib import Path

from kairos.cad.document import CADDocument
from kairos.cad.errors import ExportError


def _tip_object(cad_doc: CADDocument):
    tip = getattr(cad_doc.body, "Tip", None) if cad_doc.has_body() else None
    if tip is None:
        raise ExportError("nothing to export: body has no tip feature")
    return tip


def export_step(cad_doc: CADDocument, path: str | Path) -> Path:
    """Export the body tip solid as STEP (AP214)."""
    path = Path(path).with_suffix(".step")
    tip = _tip_object(cad_doc)
    try:
        import Import  # FreeCAD STEP module

        path.parent.mkdir(parents=True, exist_ok=True)
        Import.export([tip], str(path))
    except Exception as err:
        raise ExportError(f"STEP export to {path} failed: {err}") from err
    if not path.exists():
        raise ExportError(f"STEP export silently produced no file at {path}")
    return path


def export_stl(cad_doc: CADDocument, path: str | Path, linear_deflection: float = 0.3) -> Path:
    """Export the body tip solid as a binary STL mesh.

    ``linear_deflection`` (mm) bounds the tessellation chord error, it
    controls file size directly (0.3 mm keeps bracket-scale parts well under
    1 MB; ``Shape.exportStl`` ignores deflection entirely, so meshing goes
    through MeshPart when available).
    """
    path = Path(path).with_suffix(".stl")
    tip = _tip_object(cad_doc)
    shape = tip.Shape
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import MeshPart

            mesh = MeshPart.meshFromShape(
                Shape=shape,
                LinearDeflection=float(linear_deflection),
                AngularDeflection=0.5,
                Relative=False,
            )
            mesh.write(str(path))
        except ImportError:  # pragma: no cover - minimal FreeCAD builds
            shape.exportStl(str(path))
    except Exception as err:
        raise ExportError(f"STL export to {path} failed: {err}") from err
    if not path.exists():
        raise ExportError(f"STL export silently produced no file at {path}")
    return path


def save_fcstd(cad_doc: CADDocument, path: str | Path) -> Path:
    """Save the native FreeCAD document."""
    path = Path(path).with_suffix(".FCStd")
    path.parent.mkdir(parents=True, exist_ok=True)
    cad_doc.save(str(path))
    if not path.exists():
        raise ExportError(f"FCStd save silently produced no file at {path}")
    return path
