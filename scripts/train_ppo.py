#!/usr/bin/env python
"""PPO fine-tuning of the BC policy against the live CAD environment.

    python3 scripts/train_ppo.py --bc runs/bc/checkpoint.pt --iterations 30

Runs under the **torch** interpreter; the environment is served out of
FreeCAD's interpreter over the bridge (`kairos.rl.env_server`). Writes
``best.pt``, ``last.pt``, ``history.json`` and ``report.json`` under ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc", type=Path, default=Path("runs/bc/checkpoint.pt"),
                        help="BC checkpoint to initialize from (strongly recommended)")
    parser.add_argument("--from-scratch", action="store_true",
                        help="skip BC initialization (baseline comparison only)")
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--steps-per-iteration", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--bc-kl-coef", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--requirements", type=int, default=64,
                        help="cap the requirement pool (smaller = faster episodes)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--freecad-python", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from last.pt in --out, keeping its history and best-so-far",
    )
    args = parser.parse_args()

    try:
        from kairos.config import load_config
        from kairos.models.actor_critic import ActorCritic
        from kairos.models.vla import KairosVLA, VLAConfig
        from kairos.rl.collect import RolloutCollector
        from kairos.rl.env_client import RemoteCADEnv
        from kairos.rl.ppo import PPOConfig, PPOTrainer
        from kairos.rl.requirements import requirement_pools
        from kairos.rl.train_loop import LoopConfig, PPOTrainingLoop
    except ImportError as err:  # pragma: no cover - depends on the environment
        print(f"error: the learning stack needs torch ({err}).", file=sys.stderr)
        print('install it with: pip install -e ".[learn]"', file=sys.stderr)
        return 2

    config = load_config(args.config)
    ppo_section = dict(config.get("ppo", {}) or {})
    loop_fields = set(LoopConfig.__dataclass_fields__)
    ppo_fields = set(PPOConfig.__dataclass_fields__)
    unknown = set(ppo_section) - loop_fields - ppo_fields
    if unknown:
        print(f"error: unknown ppo config keys: {sorted(unknown)}", file=sys.stderr)
        return 2

    ppo_config = PPOConfig(**{k: v for k, v in ppo_section.items() if k in ppo_fields})
    loop_config = LoopConfig(**{k: v for k, v in ppo_section.items() if k in loop_fields})

    for field, value in (
        ("learning_rate", args.learning_rate),
        ("bc_kl_coef", args.bc_kl_coef),
        ("entropy_coef", args.entropy_coef),
    ):
        if value is not None:
            setattr(ppo_config, field, value)
    for field, value in (
        ("iterations", args.iterations),
        ("steps_per_iteration", args.steps_per_iteration),
        ("max_episode_steps", args.max_episode_steps),
        ("eval_every", args.eval_every),
        ("eval_episodes", args.eval_episodes),
        ("seed", args.seed if args.seed is not None else config.get("seed")),
    ):
        if value is not None:
            setattr(loop_config, field, value)
    out_dir = Path(args.out) if args.out else Path(loop_config.out_dir)
    loop_config.out_dir = str(out_dir)

    # ------------------------------------------------------------ the policy
    resume_checkpoint = out_dir / "last.pt"
    if args.resume and resume_checkpoint.exists():
        from kairos.models.actor_critic import load_actor_critic

        model = load_actor_critic(resume_checkpoint, device=args.device)
        print(f"policy: resumed from {resume_checkpoint}")
    elif args.from_scratch:
        model_section = dict(config.get("model", {}) or {})
        model = ActorCritic(KairosVLA(VLAConfig.from_dict(model_section)))
        print("policy: randomly initialized (--from-scratch)")
    else:
        if not args.bc.exists():
            print(
                f"error: no BC checkpoint at {args.bc}. Run `make train-bc` first, "
                "or pass --from-scratch to train without it.",
                file=sys.stderr,
            )
            return 1
        model = ActorCritic.from_bc_checkpoint(args.bc, device=args.device)
        print(f"policy: initialized from {args.bc}")
    print(f"{model.parameter_count():,} trainable parameters on {args.device}")

    # ------------------------------------------------------- the environment
    train_pool, held_out_pool = requirement_pools(
        args.root, limit=args.requirements, seed=loop_config.seed
    )
    print(f"requirements: {len(train_pool)} train / {len(held_out_pool)} held out")

    try:
        env = RemoteCADEnv(
            max_steps=loop_config.max_episode_steps, python=args.freecad_python
        )
    except Exception as err:
        print(f"error: could not start the CAD environment: {err}", file=sys.stderr)
        print("check that FreeCAD is installed, or set KAIROS_FREECAD_PYTHON.", file=sys.stderr)
        return 1

    trainer = PPOTrainer(model, ppo_config, device=args.device)
    collector = RolloutCollector(
        env, model, train_pool,
        max_episode_steps=loop_config.max_episode_steps,
        device=args.device, seed=loop_config.seed,
    )
    loop = PPOTrainingLoop(trainer, collector, loop_config, eval_requirements=held_out_pool)
    start_iteration = loop.resume_from(out_dir) if args.resume else 0
    if start_iteration:
        print(
            f"resuming after iteration {start_iteration} "
            f"(best so far {loop.best_success_rate:.3f})"
        )

    def report(record) -> None:
        rollout, update = record.rollout, record.update
        line = (
            f"iter {record.iteration:>3}  "
            f"reward {rollout.get('reward_mean', 0.0):+.2f}  "
            f"len {rollout.get('episode_length_mean', 0.0):.1f}  "
            f"success {rollout.get('success_rate', 0.0):.2f}  "
            f"invalid {rollout.get('invalid_action_rate', 0.0):.2f}  "
            f"kl {update.get('approx_kl', 0.0):+.4f}  "
            f"bc-kl {update.get('bc_kl', 0.0):.3f}  "
            f"ev {update.get('explained_variance', 0.0):+.2f}  "
            f"({record.seconds:.0f}s)"
        )
        if record.evaluation:
            line += f"  | eval success {record.evaluation.get('success_rate', 0.0):.2f}"
        print(line, flush=True)

    print(
        f"training {loop_config.iterations} iterations x "
        f"{loop_config.steps_per_iteration} steps"
    )
    try:
        loop.run(on_iteration=report, start_iteration=start_iteration)
    except KeyboardInterrupt:
        print("\ninterrupted; keeping the checkpoints written so far", file=sys.stderr)
    finally:
        env.close()

    report_payload = loop.report()
    report_payload["bc_checkpoint"] = None if args.from_scratch else str(args.bc)
    report_payload["environment"] = {
        "restarts": getattr(env, "restarts", 0),
        "timeouts": getattr(env, "timeouts", 0),
    }
    (out_dir / "report.json").write_text(json.dumps(report_payload, indent=2) + "\n")

    print(
        f"\nbest held-out success rate {loop.best_success_rate:.3f} "
        f"(iteration {loop.best_iteration}); checkpoints in {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
