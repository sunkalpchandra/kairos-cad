#!/usr/bin/env python
"""Evaluate a trained BC checkpoint on its held-out designs.

    python3 scripts/evaluate_bc.py --checkpoint runs/bc/checkpoint.pt

Re-derives the same design-level split the run used (from the checkpoint's own
train config), so the held-out set really is held out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/bc/checkpoint.pt"))
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--out", type=Path, default=None, help="write JSON here")
    parser.add_argument("--split", choices=("val", "train", "all"), default="val")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        import torch

        from kairos.training.bc_dataset import TrajectoryDataset, build_examples
        from kairos.training.bc_train import load_checkpoint, split_by_design
        from kairos.training.evaluate import evaluate, format_report
    except ImportError as err:  # pragma: no cover - depends on the environment
        print(f"error: the learning stack needs torch ({err}).", file=sys.stderr)
        return 2

    if not args.checkpoint.exists():
        print(f"error: no checkpoint at {args.checkpoint}", file=sys.stderr)
        return 1

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_config = payload.get("train_config", {})
    val_fraction = float(train_config.get("val_fraction", 0.15))
    seed = int(train_config.get("seed", 0))

    # build_examples(limit=...) changes design_index for every row, so a run
    # trained with --limit must be evaluated with the same one or the
    # "held-out" set silently includes designs it trained on.
    limit = None
    run_report = args.checkpoint.parent / "report.json"
    if run_report.exists():
        limit = json.loads(run_report.read_text()).get("limit")
    arrays, stats = build_examples(args.root, limit=limit)
    dataset = TrajectoryDataset(arrays)
    train_set, val_set = split_by_design(dataset, val_fraction, seed)
    subset = {"val": val_set, "train": train_set, "all": dataset}[args.split]

    model = load_checkpoint(args.checkpoint, device=args.device)
    report = evaluate(model, subset, device=args.device)
    report["split"] = args.split
    report["checkpoint"] = str(args.checkpoint)
    report["dataset"] = stats.to_dict()

    print(f"checkpoint: {args.checkpoint}  split: {args.split}")
    print(format_report(report))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
