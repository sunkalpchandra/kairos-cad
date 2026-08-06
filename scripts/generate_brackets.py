#!/usr/bin/env python
"""Generate a validated procedural CAD dataset.

Run under FreeCAD's bundled interpreter (see Makefile `generate-data`):

    PYTHONPATH=. /Applications/FreeCAD.app/Contents/Resources/bin/python \
        scripts/generate_brackets.py --count 25 --out dataset/designs
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=25, help="validated designs to write")
    parser.add_argument("--out", type=Path, default=Path("dataset/designs"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=["l_bracket", "plate"],
        choices=["l_bracket", "plate"],
    )
    args = parser.parse_args()

    from kairos.cad.backend import freecad_available, freecad_version

    if not freecad_available():
        print("error: FreeCAD unavailable in this interpreter; see docs/freecad-setup.md")
        return 1

    from kairos.data.generator import generate_dataset

    print(f"FreeCAD {freecad_version()} — generating {args.count} designs → {args.out}")
    start = time.time()
    stats = generate_dataset(args.out, args.count, seed=args.seed, kinds=tuple(args.kinds))
    elapsed = time.time() - start
    print(
        f"done in {elapsed:.1f}s: {stats.written} written / {stats.attempted} attempted "
        f"({stats.infeasible} infeasible draws, {stats.failed} build failures, "
        f"{stats.invalid} failed validation)"
    )
    for reason in stats.reasons[:10]:
        print(f"  - {reason}")
    return 0 if stats.written == args.count else 2


if __name__ == "__main__":
    raise SystemExit(main())
