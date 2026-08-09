#!/usr/bin/env python3
"""Fail when the app script reaches for an element the markup does not have.

    python3 scripts/check_wiring.py

`studio.js` addresses the page by id, forty-odd times. `el('coverage-stats')`
on markup that has no such element returns null, and the next property access
throws -- part way through `init`, so everything wired after that point is
silently never wired. The page still renders, because most of it was built
before the throw.

That happened while these very sections were being added: an insert into
`studio.html` did not match, the id was never created, and the only signal was
a heading missing from a screenshot.

This is a static check because the failure is static: both sides are text in
the repository, and neither needs a browser to compare.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "kairos/dashboard/assets"

#: script -> the markup it drives, from `build.LAYOUTS`. The report layout's
#: template is `index.html`; `dashboard.html` is what it is *built into*, which
#: is a different file and would have made this check quietly cover one page.
PAIRS: dict[str, str] = {
    "studio.js": "studio.html",
    "app.js": "index.html",
}

#: `el('x')` and `getElementById('x')`, single or double quoted.
_LOOKUP = re.compile(r"""(?:\bel|getElementById)\(\s*['"]([A-Za-z0-9_-]+)['"]\s*\)""")
_ID = re.compile(r"""\bid=["']([A-Za-z0-9_-]+)["']""")

#: Ids the script creates at runtime rather than finding in the template.
CREATED: set[str] = set()


def missing(script: Path, markup: Path) -> list[str]:
    """Ids the script looks up that the markup never defines."""
    if not script.exists():
        return [f"{script.name} is missing"]
    if not markup.exists():
        return [f"{markup.name} is missing"]

    wanted = set(_LOOKUP.findall(script.read_text()))
    have = set(_ID.findall(markup.read_text())) | CREATED
    return sorted(wanted - have)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    del args

    problems: list[str] = []
    checked = 0
    for script, markup in PAIRS.items():
        script_path, markup_path = ASSETS / script, ASSETS / markup
        # A pair that does not exist is a broken registry, not a page to skip:
        # skipping is how this check reported success over one page.
        if not script_path.exists() or not markup_path.exists():
            problems.append(f"{script} or {markup} is not in assets/")
            continue
        checked += 1
        for name in missing(script_path, markup_path):
            problems.append(f"{script}: el({name!r}) has no id in {markup}")

    if problems:
        print("The app reaches for elements the markup does not have:\n  "
              + "\n  ".join(problems), file=sys.stderr)
        print("\nEach one throws part way through init and silently unwires "
              "everything after it.", file=sys.stderr)
        return 1

    print(f"wiring: every id the app looks up exists, across {checked} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
