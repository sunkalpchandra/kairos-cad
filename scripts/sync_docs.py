#!/usr/bin/env python3
"""Splice generated tables into markdown docs between markers.

    python3 scripts/sync_docs.py --check     # fail if a doc is stale
    python3 scripts/sync_docs.py --write     # rewrite the docs in place

A doc block looks like this, and everything between the markers is replaced:

    <!-- generated: benchmark-leaderboard -->
    ...anything...
    <!-- /generated -->

Why this exists: `docs/phase7.md` carried five tables of numbers copied out of a
run. Copies go stale silently --- the doc still renders, the numbers still look
like numbers, and nothing anywhere says they describe a run that no longer
exists. This repo has already shipped one stale figure that way.

`--check` is the useful mode. It makes a stale doc a *failure* rather than
something a reader has to notice.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: block id -> (doc, command producing the replacement, section headings to keep)
BLOCKS: dict[str, dict] = {
    "benchmark-tables": {
        "doc": "docs/phase7.md",
        "command": [sys.executable, "scripts/benchmark_report.py",
                    "--runs", "runs/benchmark_core"],
    },
}

_MARKER = "<!-- generated: {} -->"
_END = "<!-- /generated -->"


def _pattern(block_id: str) -> re.Pattern[str]:
    return re.compile(
        re.escape(_MARKER.format(block_id)) + r".*?" + re.escape(_END),
        re.DOTALL,
    )


def render(block_id: str, spec: dict) -> str:
    """Run the generator and wrap its output in the markers."""
    result = subprocess.run(
        spec["command"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"generator for {block_id!r} failed:\n{result.stderr.strip()}"
        )
    body = result.stdout.strip()
    return f"{_MARKER.format(block_id)}\n\n{body}\n\n{_END}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="fail if any doc is stale")
    mode.add_argument("--write", action="store_true", help="rewrite docs in place")
    args = parser.parse_args()

    stale: list[str] = []
    for block_id, spec in BLOCKS.items():
        path = ROOT / spec["doc"]
        if not path.exists():
            print(f"error: {spec['doc']} does not exist", file=sys.stderr)
            return 2

        text = path.read_text()
        pattern = _pattern(block_id)
        if not pattern.search(text):
            print(
                f"error: {spec['doc']} has no '{block_id}' block; add\n"
                f"  {_MARKER.format(block_id)}\n  {_END}",
                file=sys.stderr,
            )
            return 2

        replacement = render(block_id, spec)
        # A plain string would have backslashes and \g read as replacement
        # escapes; generated tables contain neither today, but a future one
        # would corrupt silently.
        updated = pattern.sub(lambda _, value=replacement: value, text)

        if updated == text:
            print(f"up to date: {spec['doc']} [{block_id}]")
            continue
        if args.check:
            stale.append(f"{spec['doc']} [{block_id}]")
            continue
        path.write_text(updated)
        print(f"updated: {spec['doc']} [{block_id}]")

    if stale:
        print(
            "\nSTALE: these docs no longer match the artifacts:\n  "
            + "\n  ".join(stale)
            + "\n\nRun: python3 scripts/sync_docs.py --write",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
