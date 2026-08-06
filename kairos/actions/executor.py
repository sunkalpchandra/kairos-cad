"""ActionExecutor: validated dispatch of structured actions onto a CADEngine.

Failures never raise out of ``execute`` — they return ``ActionResult(ok=False)``
with a typed message, so the RL environment can convert them into penalties
while the document remains consistent (feature builders roll back internally).
"""

from __future__ import annotations

from typing import Any, Callable

from kairos.actions.parameters import ActionValidationError, validate_action
from kairos.actions.schema import Action, ActionResult, Operation
from kairos.cad.engine import CADEngine
from kairos.cad.errors import CADError


def _split_targets(target: str) -> list[str]:
    """Targets referencing multiple subelements are comma-separated."""
    return [t.strip() for t in target.split(",") if t.strip()]


class ActionExecutor:
    """Executes validated actions against one engine, recording a trajectory."""

    def __init__(self, engine: CADEngine, render_dir: str | None = None) -> None:
        self.engine = engine
        self.render_dir = render_dir
        self.history: list[dict[str, Any]] = []
        self.finished = False
        self._dispatch: dict[Operation, Callable[[Action, dict], dict[str, Any]]] = {
            Operation.CREATE_SKETCH: self._create_sketch,
            Operation.ADD_LINE: self._add_line,
            Operation.ADD_RECTANGLE: self._add_rectangle,
            Operation.ADD_CIRCLE: self._add_circle,
            Operation.ADD_ARC: self._add_arc,
            Operation.ADD_POLYGON: self._add_polygon,
            Operation.DELETE_GEOMETRY: self._delete_geometry,
            Operation.MOVE_GEOMETRY: self._move_geometry,
            Operation.ADD_HORIZONTAL: self._constraint("Horizontal", "geo"),
            Operation.ADD_VERTICAL: self._constraint("Vertical", "geo"),
            Operation.ADD_PARALLEL: self._constraint("Parallel", "geo1", "geo2"),
            Operation.ADD_PERPENDICULAR: self._constraint("Perpendicular", "geo1", "geo2"),
            Operation.ADD_TANGENT: self._constraint("Tangent", "geo1", "geo2"),
            Operation.ADD_EQUAL: self._constraint("Equal", "geo1", "geo2"),
            Operation.ADD_DISTANCE: self._constraint(
                "Distance", "geo1", "pos1", "geo2", "pos2", "value"
            ),
            Operation.ADD_RADIUS: self._constraint("Radius", "geo", "value"),
            Operation.ADD_DIAMETER: self._constraint("Diameter", "geo", "value"),
            Operation.ADD_COINCIDENT: self._constraint(
                "Coincident", "geo1", "pos1", "geo2", "pos2"
            ),
            Operation.ADD_SYMMETRY: self._constraint(
                "Symmetric", "geo1", "pos1", "geo2", "pos2", "axis_geo"
            ),
            Operation.PAD: self._pad,
            Operation.POCKET: self._pocket,
            Operation.REVOLVE: self._revolve,
            Operation.FILLET: self._fillet,
            Operation.CHAMFER: self._chamfer,
            Operation.SHELL: self._shell,
            Operation.MIRROR: self._mirror,
            Operation.LINEAR_PATTERN: self._linear_pattern,
            Operation.CIRCULAR_PATTERN: self._circular_pattern,
            Operation.MEASURE_DISTANCE: self._measure_distance,
            Operation.MEASURE_VOLUME: lambda a, p: {"volume_mm3": self.engine.measure_volume()},
            Operation.MEASURE_AREA: lambda a, p: {
                "surface_area_mm2": self.engine.measure_surface_area()
            },
            Operation.MEASURE_BOUNDING_BOX: lambda a, p: {
                "bounding_box": self.engine.measure_bounding_box()
            },
            Operation.CHECK_VALIDITY: lambda a, p: self.engine.check_validity().to_dict(),
            Operation.RENDER_VIEW: self._render_view,
            Operation.FINISH_DESIGN: self._finish,
        }

    # ------------------------------------------------------------------ api

    def execute(self, action: Action) -> ActionResult:
        """Validate and run one action; always returns, never raises."""
        if self.finished:
            result = ActionResult(
                False, action.operation, "design already finished", done=True
            )
            self._record(action, result)
            return result
        try:
            params = validate_action(action)
        except ActionValidationError as err:
            result = ActionResult(False, action.operation, f"validation: {err}")
            self._record(action, result)
            return result

        handler = self._dispatch.get(action.operation)
        if handler is None:
            result = ActionResult(
                False,
                action.operation,
                f"{action.operation.value} is not executable in Phase 1",
            )
            self._record(action, result)
            return result

        try:
            info = handler(action, params)
            result = ActionResult(
                True, action.operation, "ok", info, done=self.finished
            )
        except (CADError, ActionValidationError) as err:
            result = ActionResult(False, action.operation, str(err))
        except Exception as err:  # unexpected backend failure: still no raise
            result = ActionResult(
                False, action.operation, f"unexpected {type(err).__name__}: {err}"
            )
        self._record(action, result)
        return result

    def run(self, actions: list[Action], stop_on_error: bool = False) -> list[ActionResult]:
        """Execute a sequence; optionally stop at the first failure."""
        results = []
        for action in actions:
            result = self.execute(action)
            results.append(result)
            if stop_on_error and not result.ok:
                break
            if result.done:
                break
        return results

    def trajectory(self) -> list[dict[str, Any]]:
        """JSON-serializable record of every attempted action and outcome."""
        return list(self.history)

    def _record(self, action: Action, result: ActionResult) -> None:
        self.history.append({"action": action.to_dict(), "result": result.to_dict()})

    # ------------------------------------------------------------- handlers

    def _create_sketch(self, action: Action, p: dict) -> dict:
        name = self.engine.create_sketch(plane=p["plane"], offset=p["offset"])
        return {"sketch": name}

    def _add_line(self, action: Action, p: dict) -> dict:
        idx = self.engine.add_line(p["x1"], p["y1"], p["x2"], p["y2"], sketch=action.target)
        return {"geometry_index": idx}

    def _add_rectangle(self, action: Action, p: dict) -> dict:
        idx = self.engine.add_rectangle(
            p["x"], p["y"], p["width"], p["height"], sketch=action.target
        )
        return {"geometry_indices": idx}

    def _add_circle(self, action: Action, p: dict) -> dict:
        idx = self.engine.add_circle(p["cx"], p["cy"], p["radius"], sketch=action.target)
        return {"geometry_index": idx}

    def _add_arc(self, action: Action, p: dict) -> dict:
        idx = self.engine.add_arc(
            p["cx"], p["cy"], p["radius"], p["start_deg"], p["end_deg"], sketch=action.target
        )
        return {"geometry_index": idx}

    def _add_polygon(self, action: Action, p: dict) -> dict:
        points = [(float(pt[0]), float(pt[1])) for pt in p["points"]]
        idx = self.engine.add_polygon(points, closed=p["closed"], sketch=action.target)
        return {"geometry_indices": idx}

    def _delete_geometry(self, action: Action, p: dict) -> dict:
        self.engine.delete_geometry(p["index"], sketch=action.target)
        return {}

    def _move_geometry(self, action: Action, p: dict) -> dict:
        self.engine.move_geometry(p["index"], p["dx"], p["dy"], sketch=action.target)
        return {}

    def _constraint(self, kind: str, *param_names: str):
        def handler(action: Action, p: dict) -> dict:
            args = [p[name] for name in param_names]
            idx = self.engine.add_constraint(kind, args, sketch=action.target)
            return {"constraint_index": idx}

        return handler

    def _pad(self, action: Action, p: dict) -> dict:
        name = self.engine.pad(
            p["length"], sketch=action.target, reversed_=p["reversed"], midplane=p["midplane"]
        )
        return {"feature": name, "volume_mm3": self.engine.measure_volume()}

    def _pocket(self, action: Action, p: dict) -> dict:
        name = self.engine.pocket(
            p.get("depth"),
            sketch=action.target,
            through_all=p["through_all"],
            reversed_=p["reversed"],
        )
        return {"feature": name, "volume_mm3": self.engine.measure_volume()}

    def _revolve(self, action: Action, p: dict) -> dict:
        name = self.engine.revolve(p["angle"], p["axis"], sketch=action.target)
        return {"feature": name, "volume_mm3": self.engine.measure_volume()}

    def _fillet(self, action: Action, p: dict) -> dict:
        name = self.engine.fillet(_split_targets(action.target), p["radius"])
        return {"feature": name, "volume_mm3": self.engine.measure_volume()}

    def _chamfer(self, action: Action, p: dict) -> dict:
        name = self.engine.chamfer(_split_targets(action.target), p["size"])
        return {"feature": name, "volume_mm3": self.engine.measure_volume()}

    def _shell(self, action: Action, p: dict) -> dict:
        name = self.engine.shell(_split_targets(action.target), p["thickness"])
        return {"feature": name, "volume_mm3": self.engine.measure_volume()}

    def _mirror(self, action: Action, p: dict) -> dict:
        name = self.engine.mirror(_split_targets(action.target), p["plane"])
        return {"feature": name, "volume_mm3": self.engine.measure_volume()}

    def _linear_pattern(self, action: Action, p: dict) -> dict:
        name = self.engine.linear_pattern(
            _split_targets(action.target), p["axis"], p["length"], p["count"]
        )
        return {"feature": name, "volume_mm3": self.engine.measure_volume()}

    def _circular_pattern(self, action: Action, p: dict) -> dict:
        name = self.engine.polar_pattern(
            _split_targets(action.target), p["axis"], p["angle"], p["count"]
        )
        return {"feature": name, "volume_mm3": self.engine.measure_volume()}

    def _measure_distance(self, action: Action, p: dict) -> dict:
        return {"distance_mm": self.engine.measure_distance(p["sub_a"], p["sub_b"])}

    def _render_view(self, action: Action, p: dict) -> dict:
        if self.render_dir is None:
            raise ActionValidationError("executor has no render_dir configured")
        paths = self.engine.render(self.render_dir, views=(p["view"],), size=p["size"])
        return {"paths": {k: str(v) for k, v in paths.items()}}

    def _finish(self, action: Action, p: dict) -> dict:
        self.finished = True
        return {"summary": self.engine.summary()}
