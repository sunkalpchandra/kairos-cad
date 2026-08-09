#!/usr/bin/env python3
"""Emit the review station as an artifact-hostable fragment.

    python3 scripts/build_artifact.py --out docs/kairos-artifact.html

An artifact host wraps the file in its own `<!doctype html><head></head><body>`
skeleton, so the page must not bring its own. This strips the document wrapper
and keeps the style and script blocks, which are valid inside a body.

Everything else is unchanged: the same bundle, the same assets, so a figure
cannot differ between the hosted page and `docs/kairos.html`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kairos.dashboard.build import render  # noqa: E402

#: Height the hosted frame settles at. Tall enough for the browser tree, the
#: viewport and the full inspector including its title block.
HOSTED_MIN_HEIGHT = 860
from kairos.dashboard.bundle import MAX_DESIGNS, build_bundle  # noqa: E402


def to_fragment(html: str) -> str:
    """Strip the document wrapper, keeping everything inside <body>."""
    style = "".join(re.findall(r"<style>.*?</style>", html, re.DOTALL))
    if not style:
        raise ValueError("no <style> block found; the template changed shape")

    match = re.search(r"<body[^>]*>(.*)</body>", html, re.DOTALL)
    if not match:
        raise ValueError("no <body> found; the template changed shape")

    # The standalone file owns the viewport: body is overflow:hidden and the
    # shell is exactly 100vh. An artifact host instead measures scrollHeight and
    # sizes its iframe to match, which makes both of those circular -- a hidden
    # overflow reports no height, and 100vh then resolves against whatever the
    # iframe defaulted to.
    #
    # min-height gives that loop a fixed point: the host measures at least
    # HOSTED_MIN_HEIGHT, sizes the frame to it, and 100vh settles there.
    hosted = f"""
<style>
/* Hosted variant: the frame sizes itself from content height. */
body {{ overflow: visible; height: auto; }}
html {{ height: auto; }}
.shell {{ height: 100vh; min-height: {HOSTED_MIN_HEIGHT}px; }}
</style>"""
    return f"{style}{hosted}\n{match.group(1).strip()}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--benchmark", default="runs/benchmark_core")
    parser.add_argument("--ablation", default="runs/ablation")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="docs/kairos-artifact.html")
    parser.add_argument("--limit", type=int, default=MAX_DESIGNS)
    parser.add_argument("--stamp", default="kairos-cad-v1")
    args = parser.parse_args()

    bundle = build_bundle(
        dataset=args.dataset,
        benchmark_runs=args.benchmark,
        ablation_runs=args.ablation,
        runs_root=args.runs,
        limit=args.limit,
    )
    bundle["generated_at"] = args.stamp

    fragment = to_fragment(render(bundle, layout="studio"))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fragment)

    size_kb = path.stat().st_size / 1024
    print(f"wrote {path} ({size_kb:.0f} KB)")
    print(f"  designs : {bundle['counts']['designs_embedded']}")
    print(f"  meshes  : {bundle['counts']['meshes_attached']}")
    # Match real tags, not substrings: `<header>` and `.rail-head` both contain
    # "<head" and neither is a document wrapper.
    import re as _re

    for pattern, name in (
        (r"<!doctype\b", "<!doctype>"),
        (r"<html\b", "<html>"),
        (r"<head\s*>", "<head>"),
        (r"<body\b", "<body>"),
    ):
        if _re.search(pattern, fragment, _re.IGNORECASE):
            print(f"  WARNING: fragment still contains {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
