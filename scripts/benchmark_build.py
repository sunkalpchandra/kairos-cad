#!/usr/bin/env python
"""Freeze the KAIROS-CAD benchmark split as a committed artifact.

    python3 scripts/benchmark_build.py --root dataset --out benchmark/kairos-cad-v1

Writes `splits.json` plus a checksum. The split is data, not code: two
contaminated comparisons have shipped in this project because the split was a
function re-evaluated with different arguments, so the boundary moved. Once
frozen and committed, every consumer reads the same file.

Also writes `trajectories.sha256`, one digest per test design. The split pins
which designs are held out, but `build_tasks` re-reads each trajectory.json at
run time, so regenerating the dataset changes what the benchmark measures while
suite_version and splits_sha256 stay identical. The dataset was regenerated
three times during Phase 7 and the frozen suite silently followed it. Digests
rather than a copy of the trajectories: the point is to fail loudly on drift,
not to duplicate 28 MB of actions into git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _write_trajectory_digests(out: Path, root: Path, splits) -> str:
    """One digest per TEST design; returns the digest of the digest file.

    Only the test split: pinning train would fail every time the training data
    is legitimately extended, and test is what the benchmark replays.
    """
    digests = {}
    for design_id in sorted(splits["test"].design_ids):
        path = root / "designs" / design_id / "trajectory.json"
        if path.exists():
            digests[design_id] = hashlib.sha256(path.read_bytes()).hexdigest()
    target = out / "trajectories.sha256"
    target.write_text("".join(f"{d}  {design}\n" for design, d in sorted(digests.items())))
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _pin_trajectories(out: Path, root: Path, splits, splits_digest: str) -> int:
    """Add trajectory digests to an already-frozen suite."""
    digest = _write_trajectory_digests(out, root, splits)
    (out / "SHA256SUMS").write_text(
        f"{splits_digest}  splits.json\n{digest}  trajectories.sha256\n"
    )
    suite_file = out / "suite.json"
    if suite_file.exists():
        suite = json.loads(suite_file.read_text())
        suite["trajectories_sha256"] = digest
        suite["test_trajectories"] = len((out / "trajectories.sha256").read_text().splitlines())
        suite_file.write_text(json.dumps(suite, indent=2) + "\n")
    return len((out / "trajectories.sha256").read_text().splitlines())


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
    parser.add_argument(
        "--pin-trajectories", action="store_true",
        help="write trajectories.sha256 for the EXISTING split without re-splitting; "
             "use to add drift detection to a suite frozen before it existed",
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

    if args.pin_trajectories:
        if not target.exists():
            print(f"error: no frozen split at {target}", file=sys.stderr)
            return 1
        from kairos.benchmark import SplitSet

        splits = SplitSet.load(target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        count = _pin_trajectories(args.out, args.root, splits, digest)
        print(f"pinned {count} test trajectories; the split itself is untouched")
        return 0

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

    trajectories_digest = _write_trajectory_digests(args.out, args.root, splits)
    pinned = len((args.out / "trajectories.sha256").read_text().splitlines())
    (args.out / "SHA256SUMS").write_text(
        f"{digest}  splits.json\n"
        f"{trajectories_digest}  trajectories.sha256\n"
    )
    (args.out / "suite.json").write_text(
        json.dumps(
            {
                "suite_version": SUITE_VERSION,
                "dataset_root": str(args.root),
                "designs": len(designs),
                "seed": args.seed,
                "fractions": list(args.fractions),
                "splits_sha256": digest,
                "trajectories_sha256": trajectories_digest,
                "test_trajectories": pinned,
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
    print(f"  pinned {pinned} test trajectories")
    print("\ncommit this directory: results are only comparable against a fixed split.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
