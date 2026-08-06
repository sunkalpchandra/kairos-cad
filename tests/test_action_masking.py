"""Pure-python tests of action masking logic (no FreeCAD)."""

from kairos.actions.masking import StateFlags, legal_operations, operation_mask
from kairos.actions.schema import Operation


def test_empty_document_allows_only_bootstrap_ops():
    legal = legal_operations(StateFlags())
    assert legal == {
        Operation.CREATE_SKETCH,
        Operation.CHECK_VALIDITY,
        Operation.FINISH_DESIGN,
    }


def test_open_sketch_enables_geometry_but_not_constraints():
    legal = legal_operations(StateFlags(has_sketch=True))
    assert Operation.ADD_CIRCLE in legal
    assert Operation.ADD_RECTANGLE in legal
    assert Operation.ADD_HORIZONTAL not in legal  # nothing to constrain yet
    assert Operation.PAD not in legal  # empty profile


def test_sketch_with_geometry_enables_profile_features_and_constraints():
    legal = legal_operations(StateFlags(has_sketch=True, sketch_has_geometry=True))
    assert Operation.PAD in legal
    assert Operation.REVOLVE in legal
    assert Operation.ADD_RADIUS in legal
    assert Operation.POCKET not in legal  # nothing to cut into yet


def test_pocket_requires_solid_and_profile():
    legal = legal_operations(
        StateFlags(has_sketch=True, sketch_has_geometry=True, has_solid=True)
    )
    assert Operation.POCKET in legal


def test_dressups_require_edges_or_faces():
    base = StateFlags(has_solid=True)
    assert Operation.FILLET not in legal_operations(base)
    with_edges = StateFlags(has_solid=True, has_edges=True)
    assert Operation.FILLET in legal_operations(with_edges)
    assert Operation.CHAMFER in legal_operations(with_edges)
    assert Operation.SHELL not in legal_operations(with_edges)
    with_faces = StateFlags(has_solid=True, has_faces=True)
    assert Operation.SHELL in legal_operations(with_faces)


def test_patterns_require_features():
    flags = StateFlags(has_solid=True, has_features=True)
    legal = legal_operations(flags)
    assert Operation.MIRROR in legal
    assert Operation.LINEAR_PATTERN in legal
    assert Operation.CIRCULAR_PATTERN in legal


def test_booleans_require_multiple_bodies():
    assert Operation.UNION not in legal_operations(StateFlags(has_solid=True))
    assert Operation.UNION in legal_operations(StateFlags(has_solid=True, body_count=2))


def test_inspection_requires_solid():
    assert Operation.MEASURE_VOLUME not in legal_operations(StateFlags())
    assert Operation.MEASURE_VOLUME in legal_operations(StateFlags(has_solid=True))


def test_finish_always_legal():
    for flags in (StateFlags(), StateFlags(has_sketch=True), StateFlags(has_solid=True)):
        assert Operation.FINISH_DESIGN in legal_operations(flags)


def test_mask_ordering_matches_enum():
    mask = operation_mask(StateFlags())
    ops = list(Operation)
    assert len(mask) == len(ops)
    assert mask[ops.index(Operation.CREATE_SKETCH)] is True
    assert mask[ops.index(Operation.PAD)] is False
