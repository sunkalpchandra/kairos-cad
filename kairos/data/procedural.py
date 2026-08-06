"""Backward-compatible re-exports of the original procedural recipes.

The recipes now live in the ``kairos.data.families`` registry; this module
keeps the Phase 1 import surface stable.
"""

from __future__ import annotations

from typing import Any

from kairos.data.families.l_bracket import (  # noqa: F401
    LBracketParams,
    build_l_bracket,
    l_bracket_hole_actions,
    l_bracket_profile_actions,
)
from kairos.data.families.plate import (  # noqa: F401
    PlateParams,
    build_plate,
    plate_actions,
)


def expected_hole_count(params: LBracketParams | PlateParams) -> int:
    if isinstance(params, LBracketParams):
        return 2 * params.holes_per_leg
    return params.holes_x * params.holes_y


def params_to_dict(params: LBracketParams | PlateParams) -> dict[str, Any]:
    from dataclasses import asdict

    data = asdict(params)
    data["kind"] = "l_bracket" if isinstance(params, LBracketParams) else "plate"
    return data
