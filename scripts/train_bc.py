#!/usr/bin/env python
"""Behavioral-cloning run over the recorded expert trajectories.

    python3 scripts/train_bc.py --root dataset --epochs 12 --out runs/bc

Writes ``<out>/checkpoint.pt`` and ``<out>/report.json`` (dataset coverage,
config, per-epoch metrics). Requires the optional torch extra.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config supplying model/behavioral_cloning defaults "
        "(default: configs/default.yaml). Explicit flags below override it.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto | cpu | mps | cuda")
    parser.add_argument("--limit", type=int, default=None, help="cap trajectories (smoke runs)")
    args = parser.parse_args()

    try:
        from kairos.config import load_config
        from kairos.models.vla import KairosVLA
        from kairos.training.bc_dataset import TrajectoryDataset, build_examples
        from kairos.training.bc_train import (
            BCTrainer,
            configs_from,
            resolve_device,
            split_by_design,
            write_report,
        )
    except ImportError as err:  # pragma: no cover - depends on the environment
        print(f"error: the learning stack needs torch ({err}).", file=sys.stderr)
        print('install it with: pip install -e ".[learn]"', file=sys.stderr)
        return 2

    model_config, train_config, out_dir = configs_from(load_config(args.config))
    # Explicit flags win over the config file; anything left None keeps it.
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "device": args.device,
    }
    for field, value in overrides.items():
        if value is not None:
            setattr(train_config, field, value)
    if args.embed_dim is not None:
        model_config.embed_dim = args.embed_dim
    out = args.out or Path(out_dir)

    print(f"loading trajectories from {args.root} ...")
    arrays, stats = build_examples(args.root, limit=args.limit)
    if stats.steps_kept == 0:
        print(f"error: no supervisable steps under {args.root}", file=sys.stderr)
        return 1
    print(json.dumps(stats.to_dict(), indent=2))

    dataset = TrajectoryDataset(arrays)
    train_set, val_set = split_by_design(
        dataset, train_config.val_fraction, train_config.seed
    )
    print(
        f"{len(dataset)} steps from {len(set(dataset.design_index.tolist()))} designs "
        f"-> {len(train_set)} train / {len(val_set)} val (split by design)"
    )

    # Seed BEFORE constructing the model: fit() seeds too, but by then the
    # weights have already been drawn from an unseeded global RNG, so two
    # runs recording the same seed diverged from step one.
    import torch

    torch.manual_seed(train_config.seed)
    model = KairosVLA(model_config)
    trainer = BCTrainer(model, train_config)
    print(
        f"training {model.parameter_count():,} parameters on "
        f"{resolve_device(train_config.device)} for {train_config.epochs} epochs"
    )

    def report(metrics) -> None:
        print(
            f"epoch {metrics.epoch:>3}  "
            f"train {metrics.train_loss:.4f}  val {metrics.val_loss:.4f}  "
            f"op-acc {metrics.operation_accuracy:.3f}  "
            f"top3 {metrics.operation_top3:.3f}  "
            f"param-mae {metrics.parameter_mae:.4f}  ({metrics.seconds:.1f}s)"
        )

    history = trainer.fit(train_set, val_set, on_epoch=report)

    checkpoint = trainer.save(out / "checkpoint.pt", extra={"dataset": stats.to_dict()})
    write_report(
        out / "report.json",
        {
            "dataset_root": str(args.root),
            "limit": args.limit,
            "dataset": stats.to_dict(),
            "steps": {"total": len(dataset), "train": len(train_set), "val": len(val_set)},
            "model": {
                "parameters": model.parameter_count(),
                "config": model.config.to_dict(),
            },
            "train_config": train_config.to_dict(),
            "device": str(resolve_device(train_config.device)),
            "history": [m.to_dict() for m in history],
        },
    )
    best = max(history, key=lambda m: m.operation_accuracy)
    print(
        f"\nbest val operation accuracy {best.operation_accuracy:.3f} (epoch {best.epoch}); "
        f"checkpoint: {checkpoint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
