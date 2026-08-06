"""Design families. Importing this package populates the registry."""

from kairos.data.families import (  # noqa: F401  (self-registering)
    corner_bracket,
    flange,
    l_bracket,
    plate,
    reinforced_plate,
    spacer,
    support_bracket,
    u_bracket,
)
from kairos.data.families.base import (
    FAMILIES,
    Family,
    family_names,
    get_family,
    params_to_dict,
    register,
)

__all__ = [
    "FAMILIES",
    "Family",
    "register",
    "get_family",
    "family_names",
    "params_to_dict",
]
