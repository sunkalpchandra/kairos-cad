"""A stated requirement must be one the geometry can actually satisfy.

The constraint checker reads the bound out of the *requirement text*, not out
of the float the recipe used. So rounding direction is not cosmetic: rounding a
minimum to nearest can state a bound above what the part delivers, and the
expert then violates its own requirement while building exactly the right part.

That was 139 of 1080 designs (12.9%) before this fix.
"""

import pytest

from kairos.data.families.corner_bracket import CornerBracketParams, _min_wall
from kairos.data.families.corner_bracket import _requirements as corner_requirements
from kairos.data.families.l_bracket import LBracketParams
from kairos.data.families.l_bracket import _requirements as l_requirements
from kairos.data.families.wording import state_maximum, state_minimum, stated_minimum


class TestRoundingDirection:
    def test_minimum_rounds_down(self):
        """6.1866 stated as 6.2 is a bound the part misses by 0.013 mm."""
        assert state_minimum(6.186603) == "6.1"

    def test_maximum_rounds_up(self):
        assert state_maximum(6.181) == "6.2"

    def test_exact_values_are_unchanged(self):
        assert state_minimum(6.0) == "6.0"
        assert state_maximum(6.0) == "6.0"

    def test_the_printed_text_and_the_spec_number_agree(self):
        """A floored text with a full-precision spec re-splits the contract."""
        for value in (6.186603, 5.608921, 12.0, 0.9999):
            assert float(state_minimum(value)) == pytest.approx(stated_minimum(value))

    @pytest.mark.parametrize("value", [0.05, 3.14159, 6.186603, 89.937, 122.936])
    def test_a_floored_minimum_is_never_above_the_value(self, value):
        assert stated_minimum(value) <= value


class TestCornerBracketCountsTheRib:
    """The gusset rib is a wall; declaring only `thickness` overstated it."""

    def test_min_wall_is_the_thinner_of_plate_and_rib(self):
        p = CornerBracketParams(thickness=6.05, rib_width=5.61)
        assert _min_wall(p) == pytest.approx(5.61)

    def test_the_worst_recorded_case_is_now_satisfiable(self):
        """design_030104: rib 5.609 mm against a stated 6.0 mm minimum."""
        p = CornerBracketParams(
            thickness=6.047538320885954, gusset=14.857506867747018,
            rib_width=5.608921748499065, leg1=89.90382389233719,
            leg2=68.12758681158584, width=20.77379142243259,
            hole_diameter=6.0, holes_per_leg=2,
        )
        stated = corner_requirements(p)["spec"]["min_wall_thickness"]
        assert _min_wall(p) >= stated

    def test_a_plate_thinner_than_its_rib_still_governs(self):
        p = CornerBracketParams(thickness=4.0, rib_width=9.0)
        assert _min_wall(p) == pytest.approx(4.0)


class TestEveryStatedMinimumIsAchievable:
    """Sampled parameters must never produce a self-violating requirement."""

    @pytest.mark.parametrize("seed", range(25))
    def test_corner_bracket(self, seed):
        import random

        p = CornerBracketParams.sample(random.Random(seed))
        if not p.is_feasible():
            pytest.skip("infeasible draw")
        stated = corner_requirements(p)["spec"]["min_wall_thickness"]
        assert _min_wall(p) >= stated

    @pytest.mark.parametrize("seed", range(25))
    def test_l_bracket(self, seed):
        import random

        p = LBracketParams.sample(random.Random(seed))
        if not p.is_feasible():
            pytest.skip("infeasible draw")
        stated = l_requirements(p)["spec"]["min_wall_thickness"]
        assert p.thickness >= stated

    def test_the_text_carries_the_same_number_as_the_spec(self):
        """The parser reads the text; a mismatch reintroduces the bug."""
        import random

        p = CornerBracketParams.sample(random.Random(7))
        requirement = corner_requirements(p)
        stated = requirement["spec"]["min_wall_thickness"]
        assert f"wall thickness {stated:.1f} mm" in requirement["text"]
