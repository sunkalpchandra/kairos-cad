"""Natural-language requirement parsing into structured engineering specs."""

from kairos.language.parser import parse_requirement
from kairos.language.spec import Constraint, EngineeringSpec

__all__ = ["EngineeringSpec", "Constraint", "parse_requirement"]
