"""Round requirement text so the geometry can satisfy it.

The constraint checker reads bounds from the requirement text, not from the
float the recipe used, so the text must round in the direction that keeps the
bound satisfiable.

Rounding to nearest does not. A 6.1866 mm wall stated as "6.2 mm" makes the
expert violate its own requirement by 0.013 mm: correct geometry, wrong
sentence, scored as a failure. That was 139 designs (12.9%), all
min_wall_thickness on bracket families.

A minimum ("at least") rounds down; a maximum ("no more than") rounds up.
Stating less than the part delivers is always safe.
"""

from __future__ import annotations

import math

#: Decimals used when stating a dimension in requirement text.
DECIMALS = 1


def state_minimum(value: float, decimals: int = DECIMALS) -> str:
    """Format a lower bound, rounding **down** so the part always clears it.

    >>> state_minimum(6.186603)
    '6.1'
    >>> state_minimum(6.0)
    '6.0'
    """
    scale = 10.0**decimals
    return f"{math.floor(value * scale) / scale:.{decimals}f}"


def state_maximum(value: float, decimals: int = DECIMALS) -> str:
    """Format an upper bound, rounding **up** so the part always stays under it.

    >>> state_maximum(6.181)
    '6.2'
    """
    scale = 10.0**decimals
    return f"{math.ceil(value * scale) / scale:.{decimals}f}"


def stated_minimum(value: float, decimals: int = DECIMALS) -> float:
    """The numeric value `state_minimum` prints, for the spec dict.

    The spec and the text must agree. Keeping the spec at full precision while
    the text is floored would leave the parser and the recipe disagreeing about
    what was promised, the same split that caused the original bug.
    """
    scale = 10.0**decimals
    return math.floor(value * scale) / scale
