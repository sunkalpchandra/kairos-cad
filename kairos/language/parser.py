"""Rule-based requirement parser: natural language → EngineeringSpec.

This is deliberately a deterministic, testable extraction layer, not an LLM.
It covers the phrasing used by the KAIROS task families and benchmark
(§ hole counts, metric thread sizes, wall thickness, mounting angles,
dimensions, materials, objectives). Anything it cannot extract is simply
absent from the spec; it never invents values.
"""

from __future__ import annotations

import re

from kairos.language.spec import Constraint, EngineeringSpec

_MATERIALS = ("aluminum", "steel", "titanium", "abs", "pla")

#: metric thread designation → nominal hole diameter in mm (clearance ignored:
#: the procedural families drill nominal diameters).
_METRIC_THREADS = {"M3": 3.0, "M4": 4.0, "M5": 5.0, "M6": 6.0, "M8": 8.0, "M10": 10.0}

_NUM = r"(\d+(?:\.\d+)?)"

#: Features that add material on top of a stated block of dimensions, so a
#: "A x B x C mm" triple in the same sentence sizes a sub-component rather than
#: the finished part's envelope.
_STACKED_FEATURE_RE = re.compile(
    r"\b(ribs?|gussets?|bosses|boss|hubs?|walls?|stiffen\w*|braced|raised)\b", re.I
)


#: "<n> [up to two adjectives] holes", the adjectives matter because real
#: requirements say "6 bolt holes", "2 base mounting holes", "4 corner
#: through-holes". Intervening words must be alphabetic so a number belonging
#: to a different clause ("2 ribs 8 mm wide ... holes") cannot be captured.
_HOLE_GROUP_RE = re.compile(
    r"(?<![A-Za-z0-9])(\d+)\s+(?:[A-Za-z][\w-]*\s+){0,2}?(?:through[- ]?)?holes?\b",
    re.I,
)

#: A bore is a hole the geometry reports like any other, so it counts toward
#: the total: a flange's "12 mm central bore" plus "6 bolt holes" is 7 holes.
_BORE_RE = re.compile(r"\b(?:central|through)[- ]?bores?\b", re.I)


def _find_hole_spec(text: str) -> tuple[int | None, float | None]:
    """Extract (total hole count, nominal hole diameter in mm) from the text.

    Counts are **summed across every group the text names**. Requirements
    routinely split them ("2 base mounting holes and 2 cross-wall holes"), and
    reading only the first group under-counts the part, which then reads as a
    satisfied constraint against geometry that has twice as many holes.
    """
    count: int | None = None
    diameter: float | None = None

    # "4 x M5 holes", "4 M5 mounting holes", thread designations carry the
    # diameter, so they are matched before the generic count pattern.
    threaded = re.search(r"(\d+)\s*[xx]?\s*(M\d+)\s+(?:mounting\s+)?holes?", text, re.I)
    if threaded:
        count = int(threaded.group(1))
        diameter = _METRIC_THREADS.get(threaded.group(2).upper())
        return count, diameter

    groups = [int(m.group(1)) for m in _HOLE_GROUP_RE.finditer(text)]
    bores = len(_BORE_RE.findall(text))
    if groups or bores:
        count = sum(groups) + bores

    for pattern in (
        rf"holes?\s+of\s+{_NUM}\s*mm",  # "holes of 5 mm diameter"
        rf"all\s+{_NUM}\s*mm\s+diameter",  # "..., all 5 mm diameter"
        rf"{_NUM}\s*mm\s+diameter\s+(?:through[- ]?|central\s+)?bore",  # spacer
        rf"{_NUM}\s*mm\s+(?:diameter\s+)?holes?",  # "5 mm diameter holes"
        rf"hole\s+diameter\s*[:=]?\s*{_NUM}\s*mm",  # "hole diameter: 5 mm"
    ):
        match = re.search(pattern, text, re.I)
        if match:
            diameter = float(match.group(1))
            break
    else:
        # standalone metric thread mention, e.g. "M5 clearance"
        match = re.search(r"\b(M\d+)\b", text)
        if match:
            diameter = _METRIC_THREADS.get(match.group(1).upper())
    return count, diameter


def parse_requirement(text: str) -> EngineeringSpec:
    """Parse a natural-language engineering requirement into a spec."""
    spec = EngineeringSpec(text=text.strip())

    count, diameter = _find_hole_spec(text)
    if count is not None:
        spec.constraints.append(Constraint("hole_count", count))
    if diameter is not None:
        spec.constraints.append(Constraint("hole_diameter", diameter, tolerance=0.1))

    # "minimum wall thickness: 3 mm", "3 mm minimum wall thickness",
    # "wall thickness >= 3mm", "minimum 3 mm wall thickness"
    m = re.search(
        rf"(?:minimum|min\.?)\s+wall\s+thickness\s*(?:of\s+)?[:=>]*\s*{_NUM}\s*mm", text, re.I
    )
    if not m:
        m = re.search(rf"{_NUM}\s*mm\s+(?:minimum|min\.?)\s+wall\s+thickness", text, re.I)
    if not m:
        m = re.search(rf"(?:minimum|min\.?)\s+{_NUM}\s*mm\s+wall(?:\s+thickness)?", text, re.I)
    if not m:
        # Plain "wall thickness 6.2 mm". The separator is optional. Six
        # families phrase it exactly this way, so requiring ':' or '=' meant
        # the constraint was declared 940 times and extracted zero times.
        m = re.search(rf"wall\s+thickness\s*[:=>]*\s*{_NUM}\s*mm", text, re.I)
    if m:
        spec.constraints.append(Constraint("min_wall_thickness", float(m.group(1))))

    # "90 degree angle", "90-degree mounting geometry", "angle: 90"
    m = re.search(rf"{_NUM}\s*[-\s]?degree", text, re.I)
    if not m:
        m = re.search(rf"angle\s*[:=]\s*{_NUM}", text, re.I)
    if m:
        spec.constraints.append(Constraint("mounting_angle", float(m.group(1)), tolerance=0.5))

    # "60 x 40 x 5 mm" exact envelope, but only when the triple describes the
    # whole part. When the text also stacks material on it ("... 100 x 60 x 6 mm
    # plate stiffened by ribs 8 mm tall"), the triple sizes a sub-component and
    # the real envelope is larger; emitting it as the part envelope would invent
    # a requirement the part is meant to violate, so it is left unextracted.
    m = re.search(rf"{_NUM}\s*[xx]\s*{_NUM}\s*[xx]\s*{_NUM}\s*mm", text, re.I)
    if m and not _STACKED_FEATURE_RE.search(text):
        dims = [float(m.group(i)) for i in (1, 2, 3)]
        spec.constraints.append(Constraint("bounding_box_exact", dims, tolerance=1.0))

    # "fit within 80 x 60 x 20 mm"
    m = re.search(rf"(?:within|max(?:imum)?\s+envelope)\s+{_NUM}\s*[xx]\s*{_NUM}\s*[xx]\s*{_NUM}\s*mm", text, re.I)
    if m:
        dims = [float(m.group(i)) for i in (1, 2, 3)]
        # Replace the exact-box reading: "within" is an upper bound.
        spec.constraints = [c for c in spec.constraints if c.kind != "bounding_box_exact"]
        spec.constraints.append(Constraint("bounding_box_max", dims))

    # "symmetric hole placement", "symmetry about the XZ plane"
    m = re.search(r"symmetr(?:y|ic(?:al)?)(?:\s+(?:about|across)\s+the\s+(\w{2})\s+plane)?", text, re.I)
    if m:
        plane = (m.group(1) or "XZ").upper()
        spec.constraints.append(Constraint("symmetry", plane))

    # "two cylindrical interfaces of 12 mm diameter"
    m = re.search(rf"(\d+)\s+cylindrical\s+interfaces?\s+(?:of\s+)?{_NUM}\s*mm", text, re.I)
    if m:
        spec.constraints.append(
            Constraint(
                "cylindrical_interface",
                {"count": int(m.group(1)), "diameter": float(m.group(2))},
                tolerance=0.1,
            )
        )

    # Objectives.
    if re.search(r"minimi[sz]e\s+mass|lightweight|minimum\s+possible\s+mass|reduce\s+mass", text, re.I):
        spec.objectives.append("minimize_mass")
    if re.search(r"minimi[sz]e\s+volume", text, re.I):
        spec.objectives.append("minimize_volume")
    if re.search(r"(?:fewest|minimi[sz]e)\s+(?:number\s+of\s+)?(?:actions|operations|steps)", text, re.I):
        spec.objectives.append("minimize_actions")

    # "reduce mass by at least 20%"
    m = re.search(rf"reduce\s+mass\s+by\s+(?:at\s+least\s+)?{_NUM}\s*%", text, re.I)
    if m:
        spec.constraints.append(Constraint("mass_reduction_pct", float(m.group(1))))

    # Material.
    for material in _MATERIALS:
        if re.search(rf"\b{material}\b", text, re.I):
            spec.material = material
            break

    # "tolerance: 0.1 mm"
    m = re.search(rf"tolerance\s*[:=]?\s*(?:±\s*)?{_NUM}\s*mm", text, re.I)
    if m:
        spec.tolerance = float(m.group(1))

    return spec
