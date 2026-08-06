"""Design-family registry.

A *family* is a parametric design generator: a params dataclass with
``sample(rng)`` and ``is_feasible()``, a builder that drives an
``ActionExecutor`` (emitting the same structured actions a policy would),
and requirement metadata. Families register themselves at import time; the
dataset generator, benchmark tasks, and BC data pipeline all iterate the
registry rather than hardcoding kinds.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from kairos.actions.executor import ActionExecutor
from kairos.actions.schema import Action


class FamilyParams(Protocol):
    @classmethod
    def sample(cls, rng: random.Random) -> FamilyParams: ...

    def is_feasible(self) -> bool: ...


@dataclass(frozen=True)
class Family:
    """One registered design family."""

    name: str
    params_cls: type
    #: build(executor, params) -> list[Action]; must raise RuntimeError on
    #: failed actions (recipes are expected to succeed on feasible params).
    build: Callable[[ActionExecutor, Any], list[Action]]
    #: requirements(params) -> {"text": str, "spec": dict} (JSON-ready).
    requirements: Callable[[Any], dict[str, Any]]
    #: expected_holes(params) -> list[(diameter_mm, count)] used to validate
    #: generated geometry. Empty list = family has no holes to check.
    expected_holes: Callable[[Any], list[tuple[float, int]]]


FAMILIES: dict[str, Family] = {}


def register(family: Family) -> Family:
    if family.name in FAMILIES:
        raise ValueError(f"design family {family.name!r} already registered")
    FAMILIES[family.name] = family
    return family


def get_family(name: str) -> Family:
    try:
        return FAMILIES[name]
    except KeyError:
        raise ValueError(
            f"unknown design family {name!r}; registered: {sorted(FAMILIES)}"
        ) from None


def family_names() -> list[str]:
    return sorted(FAMILIES)


def params_to_dict(name: str, params: Any) -> dict[str, Any]:
    data = asdict(params)
    data["kind"] = name
    return data
