"""Feature-history encoder: ordered feature tree → token ids / one-hot.

The vocabulary covers the PartDesign feature types the engine can create.
Unknown types map to ``UNK`` rather than raising, so encodings stay stable
if the CAD layer grows new features before the encoder is retrained.
"""

from __future__ import annotations

import numpy as np

#: Token vocabulary; index 0 reserved for padding, 1 for unknown.
PAD, UNK = "PAD", "UNK"
FEATURE_VOCAB: tuple[str, ...] = (
    PAD,
    UNK,
    "SketchObject",
    "Pad",
    "Pocket",
    "Revolution",
    "Fillet",
    "Chamfer",
    "Thickness",
    "Mirrored",
    "LinearPattern",
    "PolarPattern",
)

_INDEX = {name: i for i, name in enumerate(FEATURE_VOCAB)}
VOCAB_SIZE = len(FEATURE_VOCAB)


def encode_history(
    history: list[str], max_length: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a feature-history list (e.g. from ``summary["feature_history"]``).

    Returns:
        (ids, mask): int64 ids padded/truncated to ``max_length`` (most recent
        features kept on truncation), and a float32 validity mask.
    """
    ids = [_INDEX.get(name, _INDEX[UNK]) for name in history]
    if len(ids) > max_length:
        ids = ids[-max_length:]
    mask = [1.0] * len(ids)
    pad = max_length - len(ids)
    ids = ids + [_INDEX[PAD]] * pad
    mask = mask + [0.0] * pad
    return np.asarray(ids, dtype=np.int64), np.asarray(mask, dtype=np.float32)


def one_hot_history(history: list[str], max_length: int = 32) -> np.ndarray:
    """One-hot [max_length, VOCAB_SIZE] encoding (padding rows all-zero)."""
    ids, mask = encode_history(history, max_length)
    out = np.zeros((max_length, VOCAB_SIZE), dtype=np.float32)
    rows = np.arange(max_length)[mask > 0]
    out[rows, ids[mask > 0]] = 1.0
    return out
