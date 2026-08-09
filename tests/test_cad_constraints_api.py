"""The constraints module surface engine.sketch_status() depends on."""

def test_constraints_module_exposes_what_sketch_status_needs():
    """engine.sketch_status() calls into constraints_api by name.

    A range deletion in constraints.py once removed constraint_count and
    is_fully_constrained along with the unused wrappers above them. Nothing
    failed to import; sketch observations silently lost fully_constrained,
    which cost BC 26% of its benchmark progress.
    """
    from kairos.cad import constraints as constraints_api

    for name in ("constraint_count", "is_fully_constrained", "degrees_of_freedom"):
        assert callable(getattr(constraints_api, name, None)), (
            f"constraints.{name} is missing; engine.sketch_status() calls it"
        )
