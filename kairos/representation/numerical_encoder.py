"""Numerical state encoder: observation + spec + progress → fixed vector.

The vector layout is frozen and documented here because BC datasets, the RL
environment, and any trained policy must agree on it. Scaling constants
assume the KAIROS workspace envelope (parts within ~200 mm, masses within
~1 kg); values are squashed with log1p where ranges are heavy-tailed.
"""

from __future__ import annotations

import math

import numpy as np

from kairos.language.spec import EngineeringSpec

#: Ordered feature names; ENCODING_DIM == len(FEATURE_NAMES).
FEATURE_NAMES: tuple[str, ...] = (
    "has_solid",
    "is_valid",
    "log_volume",  # log1p(mm^3) / 15
    "log_mass",  # log1p(g) / 8
    "log_area",  # log1p(mm^2) / 12
    "bbox_x",  # mm / 200
    "bbox_y",
    "bbox_z",
    "solids",  # count / 4
    "faces",  # count / 64
    "edges",  # count / 128
    "hole_count",  # count / 12
    "feature_count",  # count / 16
    "sketch_open",
    "sketch_geometry",  # count / 12
    "sketch_fully_constrained",
    "step_fraction",  # step / max_steps
    "spec_hole_count",  # / 12
    "spec_hole_diameter",  # mm / 12
    "spec_wall_thickness",  # mm / 12
    "spec_has_angle",
    "objective_minimize_mass",
    "constraint_satisfaction",  # measured satisfaction rate
    "constraints_all_satisfied",
)

ENCODING_DIM = len(FEATURE_NAMES)


def encode_numeric(
    observation: dict,
    spec: EngineeringSpec,
    step: int = 0,
    max_steps: int = 1,
    satisfaction_rate: float | None = None,
    all_satisfied: bool | None = None,
) -> np.ndarray:
    """Encode one observation into the fixed float32 vector."""
    summary = observation.get("summary", {})
    sketch = observation.get("sketch") or {}
    bbox = summary.get("bounding_box") or {}

    def log_scaled(value, denom):
        return math.log1p(max(float(value or 0.0), 0.0)) / denom

    values = [
        1.0 if summary.get("has_solid") else 0.0,
        1.0 if summary.get("valid") else 0.0,
        log_scaled(summary.get("volume_mm3"), 15.0),
        log_scaled(summary.get("mass_g"), 8.0),
        log_scaled(summary.get("surface_area_mm2"), 12.0),
        float(bbox.get("x_len", 0.0)) / 200.0,
        float(bbox.get("y_len", 0.0)) / 200.0,
        float(bbox.get("z_len", 0.0)) / 200.0,
        float(summary.get("topology", {}).get("solids", 0)) / 4.0,
        float(summary.get("topology", {}).get("faces", 0)) / 64.0,
        float(summary.get("topology", {}).get("edges", 0)) / 128.0,
        float(summary.get("hole_count", 0)) / 12.0,
        float(len(summary.get("feature_history", []))) / 16.0,
        1.0 if observation.get("sketch") is not None else 0.0,
        float(sketch.get("geometry_count", 0)) / 12.0,
        1.0 if sketch.get("fully_constrained") else 0.0,
        float(step) / float(max(max_steps, 1)),
        float(spec.hole_count or 0) / 12.0,
        float(spec.hole_diameter or 0.0) / 12.0,
        float(spec.min_wall_thickness or 0.0) / 12.0,
        1.0 if spec.get("mounting_angle") is not None else 0.0,
        1.0 if spec.has_objective("minimize_mass") else 0.0,
        float(satisfaction_rate if satisfaction_rate is not None else 0.0),
        1.0 if all_satisfied else 0.0,
    ]
    vector = np.asarray(values, dtype=np.float32)
    assert vector.shape == (ENCODING_DIM,)
    return vector
