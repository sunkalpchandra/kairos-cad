"""Deterministic tokenizer for engineering requirement text.

The vocabulary is **frozen in this file**, not learned at load time: token ids
are baked into trained checkpoints, so they must not shift when the dataset is
regenerated or a new family is added. Unknown words map to ``<unk>``; adding a
family means appending to ``EXTRA_WORDS`` (never reordering ``CORPUS_WORDS``).

Numbers are the part of a CAD requirement that carries the most signal — "6 mm
holes" and "20 mm holes" are different designs — but treating every distinct
magnitude as its own token would fragment the vocabulary. Each numeric literal
therefore becomes a single ``<num>`` token *plus* its value in a parallel
array, so the encoder can embed magnitude continuously instead of
categorically.
"""

from __future__ import annotations

import re

PAD, UNK, NUM, BOS, EOS = "<pad>", "<unk>", "<num>", "<bos>", "<eos>"
SPECIALS: tuple[str, ...] = (PAD, UNK, NUM, BOS, EOS)

PAD_ID, UNK_ID, NUM_ID, BOS_ID, EOS_ID = range(5)

#: Every word appearing in the eight families' generated requirements
#: (1,080 designs). Frozen — append to EXTRA_WORDS instead of editing this.
CORPUS_WORDS: tuple[str, ...] = (
    "a", "all", "and", "at", "base", "bolt", "bore", "braced", "bracket", "by",
    "central", "chamfers", "circle", "circular", "corner", "cross-wall",
    "cylindrical", "deep", "degree", "design", "diameter", "flange",
    "full-length", "grid", "gusset", "height", "holes", "hub", "in", "inner",
    "l-bracket", "leg", "legs", "mass", "minimize", "mm", "mounting", "of",
    "on", "outer", "per", "plate", "provide", "raised", "rectangular",
    "reinforced", "rib", "ribs", "rims", "side", "spacer", "stiffened",
    "support", "tall", "the", "thick", "thickness", "through-bore",
    "through-holes", "triangular", "u-channel", "vertical", "wall", "walls",
    "wide", "width", "with", "x",
)

#: Benchmark and free-form phrasings the generated corpus does not contain but
#: the parser accepts (§ requirement suite): materials, objectives, tolerances,
#: envelope wording, thread designations.
EXTRA_WORDS: tuple[str, ...] = (
    "abs", "aluminum", "angle", "arm", "boss", "bracket-mounted", "clearance",
    "counterbored", "envelope", "fewest", "fit", "fillet", "hole", "least",
    "lightweight", "m10", "m3", "m4", "m5", "m6", "m8", "maximum", "min",
    "minimum", "must", "n", "nm", "operations", "pattern", "pla", "possible",
    "reduce", "remove", "same", "shell", "steel", "steps", "stiffness",
    "symmetric", "symmetry", "titanium", "tolerance", "under", "volume",
    "weight", "while", "within",
)

VOCAB: tuple[str, ...] = SPECIALS + CORPUS_WORDS + EXTRA_WORDS
VOCAB_SIZE = len(VOCAB)

_TOKEN_TO_ID: dict[str, int] = {word: i for i, word in enumerate(VOCAB)}

#: Words split on whitespace and punctuation, keeping hyphenated compounds
#: ("cross-wall") and decimal numbers ("6.5") intact.
_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[A-Za-z][A-Za-z-]*")

#: Values are scaled into roughly [0, 1] before embedding; 200 mm covers the
#: ±100 mm workspace envelope and every dimension the families sample.
NUM_SCALE = 200.0


def tokenize(text: str) -> list[str]:
    """Split text into vocabulary-shaped tokens (lowercased, numbers kept)."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def encode(
    text: str,
    max_length: int = 64,
    add_special: bool = True,
) -> tuple[list[int], list[float], list[int]]:
    """Encode text into (token ids, numeric values, attention mask).

    ``values[i]`` is the literal at position ``i`` divided by :data:`NUM_SCALE`
    when ``ids[i]`` is ``<num>``, and 0.0 everywhere else. All three lists are
    padded to ``max_length``; the mask is 1 for real tokens.
    """
    ids: list[int] = [BOS_ID] if add_special else []
    values: list[float] = [0.0] if add_special else []

    budget = max_length - (1 if add_special else 0)
    for token in tokenize(text):
        if len(ids) >= budget:
            break
        if token[0].isdigit():
            ids.append(NUM_ID)
            values.append(float(token) / NUM_SCALE)
        else:
            ids.append(_TOKEN_TO_ID.get(token, UNK_ID))
            values.append(0.0)

    if add_special:
        ids.append(EOS_ID)
        values.append(0.0)

    mask = [1] * len(ids)
    pad = max_length - len(ids)
    if pad > 0:
        ids += [PAD_ID] * pad
        values += [0.0] * pad
        mask += [0] * pad
    return ids[:max_length], values[:max_length], mask[:max_length]


def decode(ids: list[int]) -> str:
    """Inverse of :func:`encode` for debugging (numbers read back as <num>)."""
    return " ".join(
        VOCAB[i] for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID) and 0 <= i < VOCAB_SIZE
    )


def unknown_rate(text: str) -> float:
    """Fraction of word tokens that fall outside the vocabulary."""
    words = [t for t in tokenize(text) if not t[0].isdigit()]
    if not words:
        return 0.0
    return sum(w not in _TOKEN_TO_ID for w in words) / len(words)
