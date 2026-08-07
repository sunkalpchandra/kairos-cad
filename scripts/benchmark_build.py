#!/usr/bin/env python
"""Freeze the KAIROS-CAD benchmark split as a committed artifact.

    python3 scripts/benchmark_build.py --root dataset --out benchmark/kairos-cad-v1

Writes `splits.json` plus a checksum. The split is **data, not code**: two
contaminated comparisons have shipped in this project because the split was a
function re-evaluated with different arguments, so the boundary moved. Once
frozen and committed, every consumer reads the same file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--out", type=Path, default=Path("benchmark/kairos-cad-v1"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fractions", type=float, nargs=3, default=(0.7, 0.15, 0.15),
        metavar=("TRAIN", "DEV", "TEST"),
    )
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite an existing frozen split (this invalidates published results)",
    )
    args = parser.parse_args()

    from kairos.benchmark import (
        SUITE_VERSION,
        assert_disjoint,
        build_splits,
        load_requirements_by_design,
    )

    designs = load_requirements_by_design(args.root)
    if not designs:
        print(f"error: no designs under {args.root}", file=sys.stderr)
        return 1

    target = args.out / "splits.json"
    if target.exists() and not args.force:
        print(
            f"error: {target} already exists. A frozen split must not move -- "
            "every published number is tied to it. Pass --force only if you "
            "intend to invalidate those results.",
            file=sys.stderr,
        )
        return 1

    splits = build_splits(designs, fractions=tuple(args.fractions), seed=args.seed)
    splits.dataset_root = str(args.root)
    assert_disjoint(splits)
    splits.save(target)

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (args.out / "SHA256SUMS").write_text(f"{digest}  splits.json\n")
    (args.out / "suite.json").write_text(
        json.dumps(
            {
                "suite_version": SUITE_VERSION,
                "dataset_root": str(args.root),
                "designs": len(designs),
                "seed": args.seed,
                "fractions": list(args.fractions),
                "splits_sha256": digest,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"froze {SUITE_VERSION} from {len(designs)} designs -> {args.out}")
    for name in ("train", "dev", "test"):
        split = splits[name]
        print(f"  {name:>5}: {len(split.design_ids):>4} designs, "
              f"{len(split.text_hashes):>4} unique requirements")
    print(f"  sha256: {digest[:16]}...")
    print("\ncommit this directory: results are only comparable against a fixed split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
