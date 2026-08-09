#!/usr/bin/env python3
"""Fail when the built station is older than the assets it inlines.

    python3 scripts/check_station.py

`docs/index.html` is assembled by inlining the stylesheet, the icon sprite and
three scripts. Editing one of those and forgetting to rebuild ships a page that
silently lags the source: it still renders, still passes every test, and simply
is not the thing in the repository.

This needs no dataset and no run artifacts, so unlike the doc-table gate it can
run in CI. It compares content rather than modification times, which survives a
fresh clone where every file has the same checkout timestamp.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "kairos/dashboard/assets"

#: Built page -> the assets it inlines, from `build.LAYOUTS`.
PAGES: dict[str, tuple[str, ...]] = {
    "docs/index.html": ("studio.css", "icons.html", "viewer.js", "charts.js", "studio.js"),
    "docs/dashboard.html": ("style.css", "viewer.js", "charts.js", "app.js"),
}

#: Lines that legitimately differ: the artifact variant appends a hosted-frame
#: override, and both carry a generated stamp.
_NOISE = re.compile(r"^\s*(/\*|\*|//|<!--)")


def _signature(text: str) -> list[str]:
    """Comment-free, blank-free lines, for a comparison that ignores wrapping."""
    return [line.strip() for line in text.splitlines()
            if line.strip() and not _NOISE.match(line)]


def stale(page: Path, assets: tuple[str, ...]) -> list[str]:
    """Assets whose current content is not present in the built page."""
    if not page.exists():
        return [f"{page.name} has not been built"]
    built = page.read_text()
    missing = []
    for name in assets:
        source = ASSETS / name
        if not source.exists():
            missing.append(f"{name} is missing from assets/")
            continue
        lines = _signature(source.read_text())
        if not lines:
            continue
        # A sample rather than every line: enough to catch an edit, cheap enough
        # to run on a 300 KB page. Distinctive lines only, so a shared `}` in
        # two stylesheets cannot make a stale page look current.
        distinctive = [line for line in lines if len(line) > 30]
        sample = distinctive[:: max(1, len(distinctive) // 12)][:12]
        absent = [line for line in sample if line not in built]
        if absent:
            missing.append(
                f"{name}: {len(absent)} of {len(sample)} sampled lines are not in "
                f"{page.name}, e.g. {absent[0][:70]!r}"
            )
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", nargs="*", default=list(PAGES))
    args = parser.parse_args()

    problems: list[str] = []
    for name in args.pages:
        assets = PAGES.get(name)
        if assets is None:
            print(f"error: {name} is not a known built page", file=sys.stderr)
            return 2
        for problem in stale(ROOT / name, assets):
            problems.append(f"{name}: {problem}")

    if problems:
        print("The built pages lag their assets:\n  " + "\n  ".join(problems),
              file=sys.stderr)
        print("\nRebuild with:  make dashboard-studio && make dashboard",
              file=sys.stderr)
        return 1

    print(f"up to date: {len(args.pages)} built pages match their assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
