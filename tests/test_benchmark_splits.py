"""Split tests, the contamination gate is the point of this module."""

import json

import pytest

from kairos.benchmark import (
    SPLIT_NAMES,
    ContaminationError,
    Split,
    SplitSet,
    assert_disjoint,
    build_splits,
    load_requirements_by_design,
    requirements_for,
    text_hash,
)


def _designs(n=100):
    return {f"design_{i:06d}": f"Design a plate {i} x 40 x 5 mm" for i in range(n)}


def test_split_covers_every_design_exactly_once():
    designs = _designs()
    splits = build_splits(designs, seed=0)
    covered = [d for name in SPLIT_NAMES for d in splits[name].design_ids]
    assert sorted(covered) == sorted(designs)
    assert len(covered) == len(set(covered))


def test_splits_are_disjoint_by_id_and_by_text():
    assert_disjoint(build_splits(_designs(), seed=0))  # must not raise


def test_duplicate_texts_never_straddle_a_boundary():
    """Two designs with the same requirement must land in the same split.

    Splitting by design id alone lets a policy that memorized one score on the
    other, the near-duplicate version of the leak that already shipped once.
    """
    designs = {f"design_{i:06d}": "Design a plate 60 x 40 x 5 mm" for i in range(20)}
    designs.update({f"design_1{i:05d}": f"Design a spacer {i} mm tall" for i in range(20)})
    splits = build_splits(designs, seed=3)
    assert_disjoint(splits)

    homes = {
        name for name in SPLIT_NAMES
        for d in splits[name].design_ids if designs[d] == "Design a plate 60 x 40 x 5 mm"
    }
    assert len(homes) == 1, "one requirement text was spread across splits"


def test_contamination_is_detected_by_shared_id():
    splits = SplitSet(splits={
        "train": Split("train", ["design_000001"], ["aaa"]),
        "test": Split("test", ["design_000001"], ["bbb"]),
    })
    with pytest.raises(ContaminationError, match="share 1 designs"):
        assert_disjoint(splits)


def test_contamination_is_detected_by_shared_text():
    """Different designs, same requirement, still contamination."""
    splits = SplitSet(splits={
        "train": Split("train", ["design_000001"], ["same"]),
        "test": Split("test", ["design_000002"], ["same"]),
    })
    with pytest.raises(ContaminationError, match="share 1 requirement texts"):
        assert_disjoint(splits)


def test_text_hash_ignores_whitespace_and_case():
    assert text_hash("Design  a PLATE ") == text_hash("design a plate")
    assert text_hash("design a plate") != text_hash("design a spacer")


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        build_splits(_designs(), fractions=(0.5, 0.2, 0.2))


def test_split_is_deterministic_for_a_seed():
    a = build_splits(_designs(), seed=7)
    b = build_splits(_designs(), seed=7)
    assert a["test"].design_ids == b["test"].design_ids


def test_a_different_seed_gives_a_different_split():
    a = build_splits(_designs(), seed=1)
    b = build_splits(_designs(), seed=2)
    assert a["test"].design_ids != b["test"].design_ids


def test_split_does_not_depend_on_input_ordering():
    """Filesystem enumeration order must not change the partition."""
    designs = _designs(60)
    reversed_designs = dict(reversed(list(designs.items())))
    assert (
        build_splits(designs, seed=5)["test"].design_ids
        == build_splits(reversed_designs, seed=5)["test"].design_ids
    )


def test_split_round_trips_through_json(tmp_path):
    splits = build_splits(_designs(), seed=0)
    restored = SplitSet.load(splits.save(tmp_path / "splits.json"))
    for name in SPLIT_NAMES:
        assert restored[name].design_ids == splits[name].design_ids
        assert restored[name].text_hashes == splits[name].text_hashes


def test_saved_split_is_stable_on_disk(tmp_path):
    """The file is an artifact; re-saving must not churn it."""
    splits = build_splits(_designs(), seed=0)
    first = splits.save(tmp_path / "a.json").read_text()
    second = SplitSet.load(tmp_path / "a.json").save(tmp_path / "b.json").read_text()
    assert first == second


def test_requirements_for_returns_deduplicated_texts():
    designs = {"design_000001": "same text", "design_000002": "same text",
               "design_000003": "other text"}
    splits = SplitSet(splits={
        "test": Split("test", ["design_000001", "design_000002", "design_000003"],
                      [text_hash("same text"), text_hash("other text")]),
    })
    assert sorted(requirements_for(splits, "test", designs)) == ["other text", "same text"]


def test_loading_a_dataset_skips_unreadable_entries(tmp_path):
    good = tmp_path / "designs" / "design_000000"
    good.mkdir(parents=True)
    (good / "requirements.json").write_text(json.dumps({"text": "Design a plate"}))
    broken = tmp_path / "designs" / "design_000001"
    broken.mkdir(parents=True)
    (broken / "requirements.json").write_text("{not json")
    assert load_requirements_by_design(tmp_path) == {"design_000000": "Design a plate"}
