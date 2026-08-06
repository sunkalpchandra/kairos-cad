"""Pure-python tests of numerical and feature-history encoders."""

import numpy as np

from kairos.language import parse_requirement
from kairos.representation.feature_encoder import (
    VOCAB_SIZE,
    encode_history,
    one_hot_history,
)
from kairos.representation.numerical_encoder import (
    ENCODING_DIM,
    FEATURE_NAMES,
    encode_numeric,
)

SPEC = parse_requirement("Plate with 4 M5 holes, minimum wall thickness: 3 mm. Minimize mass.")


def _obs():
    return {
        "summary": {
            "has_solid": True,
            "valid": True,
            "volume_mm3": 8000.0,
            "mass_g": 21.6,
            "surface_area_mm2": 2800.0,
            "bounding_box": {"x_len": 40.0, "y_len": 20.0, "z_len": 10.0},
            "topology": {"solids": 1, "faces": 7, "edges": 15},
            "hole_count": 1,
            "feature_history": ["SketchObject", "Pad", "SketchObject", "Pocket"],
        },
        "holes": [],
        "faces": [],
        "sketch": {"geometry_count": 3, "fully_constrained": True},
    }


def test_vector_shape_and_dtype():
    v = encode_numeric(_obs(), SPEC, step=5, max_steps=40)
    assert v.shape == (ENCODING_DIM,)
    assert v.dtype == np.float32
    assert len(FEATURE_NAMES) == ENCODING_DIM


def test_named_slots():
    v = encode_numeric(_obs(), SPEC, step=10, max_steps=40, satisfaction_rate=0.5)
    at = {name: v[i] for i, name in enumerate(FEATURE_NAMES)}
    assert at["has_solid"] == 1.0
    assert at["bbox_x"] == np.float32(40.0 / 200.0)
    assert at["step_fraction"] == np.float32(0.25)
    assert at["spec_hole_count"] == np.float32(4 / 12)
    assert at["spec_hole_diameter"] == np.float32(5 / 12)
    assert at["objective_minimize_mass"] == 1.0
    assert at["sketch_fully_constrained"] == 1.0
    assert at["constraint_satisfaction"] == np.float32(0.5)


def test_empty_observation_encodes_zeros():
    v = encode_numeric({"summary": {}, "sketch": None}, SPEC)
    at = {name: v[i] for i, name in enumerate(FEATURE_NAMES)}
    assert at["has_solid"] == 0.0
    assert at["log_mass"] == 0.0
    assert at["sketch_open"] == 0.0


def test_history_encoding_pads_and_masks():
    ids, mask = encode_history(["SketchObject", "Pad"], max_length=6)
    assert ids.shape == (6,) and mask.shape == (6,)
    assert mask.tolist() == [1, 1, 0, 0, 0, 0]
    assert ids[2:].tolist() == [0, 0, 0, 0]  # PAD id


def test_history_truncation_keeps_recent():
    history = ["SketchObject", "Pad", "Pocket", "Fillet"]
    ids, mask = encode_history(history, max_length=2)
    assert mask.tolist() == [1, 1]
    round_trip, _ = encode_history(["Pocket", "Fillet"], max_length=2)
    assert ids.tolist() == round_trip.tolist()


def test_unknown_feature_maps_to_unk():
    ids, _ = encode_history(["Warp9Drive"], max_length=2)
    assert ids[0] == 1  # UNK


def test_one_hot_shape_and_rows():
    m = one_hot_history(["Pad", "Pocket"], max_length=4)
    assert m.shape == (4, VOCAB_SIZE)
    assert m[:2].sum() == 2.0
    assert m[2:].sum() == 0.0
