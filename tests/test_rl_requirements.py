"""Requirement pool tests (no torch, no FreeCAD)."""

import json

import pytest

from kairos.rl.requirements import (
    FALLBACK_REQUIREMENTS,
    load_requirements,
    requirement_pools,
)


def _dataset(tmp_path, texts, kinds=None):
    for i, text in enumerate(texts):
        design = tmp_path / "designs" / f"design_{i:06d}"
        design.mkdir(parents=True)
        payload = {"text": text, "spec": {"kind": (kinds or ["plate"] * len(texts))[i]}}
        (design / "requirements.json").write_text(json.dumps(payload))
    return tmp_path


def test_loads_requirement_texts_from_a_dataset(tmp_path):
    _dataset(tmp_path, ["plate one", "plate two"])
    assert load_requirements(tmp_path) == ["plate one", "plate two"]


def test_families_filter_selects_a_subset(tmp_path):
    _dataset(tmp_path, ["a", "b", "c"], kinds=["plate", "flange", "plate"])
    assert load_requirements(tmp_path, families=["flange"]) == ["b"]


def test_unreadable_entries_are_skipped_not_fatal(tmp_path):
    _dataset(tmp_path, ["good"])
    broken = tmp_path / "designs" / "design_000001"
    broken.mkdir(parents=True)
    (broken / "requirements.json").write_text("{not json")
    assert load_requirements(tmp_path) == ["good"]


def test_pools_fall_back_without_a_dataset(tmp_path):
    train, held_out = requirement_pools(tmp_path)
    assert train == list(FALLBACK_REQUIREMENTS)
    assert held_out == list(FALLBACK_REQUIREMENTS)


def test_fallback_covers_several_families():
    """A single-shape pool would not exercise the language encoder."""
    assert len(FALLBACK_REQUIREMENTS) >= 5
    assert len(set(FALLBACK_REQUIREMENTS)) == len(FALLBACK_REQUIREMENTS)


def test_pools_are_disjoint(tmp_path):
    """Evaluating on a trained-against requirement would be meaningless."""
    _dataset(tmp_path, [f"requirement number {i}" for i in range(20)])
    train, held_out = requirement_pools(tmp_path, held_out_fraction=0.25, seed=0)
    assert train and held_out
    assert set(train).isdisjoint(held_out)
    assert len(train) + len(held_out) == 20


def test_pools_are_deterministic_for_a_seed(tmp_path):
    _dataset(tmp_path, [f"requirement {i}" for i in range(12)])
    first = requirement_pools(tmp_path, seed=5)
    second = requirement_pools(tmp_path, seed=5)
    assert first == second


def test_duplicate_requirements_are_collapsed(tmp_path):
    _dataset(tmp_path, ["same", "same", "different"])
    train, held_out = requirement_pools(tmp_path, held_out_fraction=0.5)
    assert sorted(train + held_out) == ["different", "same"]


def test_a_single_requirement_yields_identical_pools(tmp_path):
    _dataset(tmp_path, ["only one"])
    train, held_out = requirement_pools(tmp_path)
    assert train == held_out == ["only one"]


@pytest.mark.parametrize("fraction", [0.1, 0.5])
def test_held_out_fraction_is_respected(tmp_path, fraction):
    _dataset(tmp_path, [f"r{i}" for i in range(10)])
    _, held_out = requirement_pools(tmp_path, held_out_fraction=fraction)
    assert len(held_out) == max(1, round(10 * fraction))
