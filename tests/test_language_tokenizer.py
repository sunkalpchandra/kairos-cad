"""Tokenizer tests: stable ids, number handling, corpus coverage."""

import pytest

from kairos.language import tokenizer as tk


def test_specials_occupy_fixed_ids():
    """Checkpoints bake these in; they must never move."""
    assert (tk.PAD_ID, tk.UNK_ID, tk.NUM_ID, tk.BOS_ID, tk.EOS_ID) == (0, 1, 2, 3, 4)
    assert tk.VOCAB[: len(tk.SPECIALS)] == tk.SPECIALS
    assert len(set(tk.VOCAB)) == len(tk.VOCAB)  # no duplicate words


def test_numbers_become_num_tokens_with_scaled_values():
    ids, values, mask = tk.encode("4 holes of 6.5 mm diameter", max_length=16)
    num_positions = [i for i, t in enumerate(ids) if t == tk.NUM_ID]
    assert len(num_positions) == 2
    assert values[num_positions[0]] == pytest.approx(4.0 / tk.NUM_SCALE)
    assert values[num_positions[1]] == pytest.approx(6.5 / tk.NUM_SCALE)
    # Non-numeric positions carry no magnitude.
    assert all(values[i] == 0.0 for i in range(len(ids)) if i not in num_positions)
    assert sum(mask) == ids.index(tk.EOS_ID) + 1


def test_padding_and_truncation_keep_shapes():
    short = tk.encode("plate", max_length=12)
    long = tk.encode(" ".join(["plate"] * 100), max_length=12)
    for ids, values, mask in (short, long):
        assert len(ids) == len(values) == len(mask) == 12
    assert short[2].count(1) < 12  # padded
    assert long[2] == [1] * 12  # truncated, fully attended


def test_hyphenated_compounds_survive_tokenization():
    assert "cross-wall" in tk.tokenize("2 cross-wall holes")
    assert "through-holes" in tk.tokenize("4 corner through-holes")


def test_unknown_words_map_to_unk_without_raising():
    ids, _, _ = tk.encode("design a flurbulator with 2 holes")
    assert tk.UNK_ID in ids
    assert tk.unknown_rate("design a flurbulator") == pytest.approx(1 / 3)


def test_generated_corpus_is_fully_covered():
    """Every family's phrasing must tokenize without a single <unk>."""
    from kairos.data.families import family_names, get_family

    for name in family_names():
        family = get_family(name)
        text = family.requirements(family.params_cls())["text"]
        assert tk.unknown_rate(text) == 0.0, f"{name}: {text}"


def test_decode_round_trips_words():
    text = "design a rectangular plate with 4 holes"
    ids, _, _ = tk.encode(text)
    assert tk.decode(ids) == "design a rectangular plate with <num> holes"
