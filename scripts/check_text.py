#!/usr/bin/env python3
"""Fail on unicode punctuation and invisible characters in source and docs.

    python3 scripts/check_text.py

Two separate problems, one check.

Zero-width and invisible characters are always a bug. A U+200C inside a glob
pattern in docs/phase8.md made the path unmatchable and unreadable at once.

Typographic punctuation (curly quotes, em and en dashes, ellipses) is a style
choice this repo does not make, so a sweep once replaced it repo-wide, and that
sweep collapsed the regex class `[x*]` to `[xx]` in three places. The class
still matched ASCII `x`, so the whole suite passed while any requirement using
the other character silently lost its envelope constraint. Catching the
characters at the door is cheaper than auditing a sweep afterwards.

Symbols that carry meaning are allowed: degree, plus-minus, superscripts,
Greek used in formulae, and box-drawing for ASCII-art diagrams.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Characters that carry technical meaning and are worth their non-ASCII cost.
ALLOWED = set("°±§²³µΔΩπλσθ‖")
#: Box-drawing and arrows used in the architecture diagrams.
ALLOWED |= set("─│┌┐└┘├┤┬┴┼╭╮╯╰▲▼◄►▶◀→←↔↑↓")

#: Always wrong: invisible, or a look-alike for something ASCII.
BANNED = {
    "​": "zero-width space",
    "‌": "zero-width non-joiner",
    "‍": "zero-width joiner",
    "﻿": "byte-order mark",
    " ": "non-breaking space",
    "‘": "left single quote",
    "’": "right single quote",
    "“": "left double quote",
    "”": "right double quote",
    "–": "en dash",
    "—": "em dash",
    "…": "ellipsis",
    "×": "multiplication sign",
    "−": "minus sign",
    "≤": "less-than-or-equal",
    "≥": "greater-than-or-equal",
}

#: Files whose non-ASCII content is the point.
SKIP = {"scripts/check_text.py", "docs/dashboard.html"}


def offenders(path: Path) -> list[tuple[int, int, str, str]]:
    """(line, column, character, why) for every disallowed character."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    found = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for column, char in enumerate(line, 1):
            if char.isascii() or char in ALLOWED:
                continue
            why = BANNED.get(char)
            if why is None:
                # Unlisted non-ASCII: report it rather than guessing, but name
                # it so the decision to allow it can be made deliberately.
                why = unicodedata.name(char, "unnamed").lower()
            found.append((lineno, column, char, why))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patterns", nargs="+", default=["*.py", "*.md", "*.sh", "Makefile"],
        help="git-tracked globs to check",
    )
    args = parser.parse_args()

    listed = subprocess.run(
        ["git", "ls-files", *args.patterns],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()

    total = 0
    for name in listed:
        if name in SKIP:
            continue
        for lineno, column, char, why in offenders(ROOT / name):
            escape = f"\\u{ord(char):04x}"
            print(f"{name}:{lineno}:{column}: {why} ({escape})")
            total += 1

    if total:
        print(
            f"\n{total} disallowed characters. Use the ASCII equivalent, or add "
            f"the character to ALLOWED in {Path(__file__).name} if it carries "
            "technical meaning.",
            file=sys.stderr,
        )
        return 1
    print(f"clean: {len(listed)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
