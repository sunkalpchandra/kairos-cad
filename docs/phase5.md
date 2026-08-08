# Phase 5: Reinforcement learning in the live CAD environment

Phase 4 produced a policy that predicts the expert's next action with 96%
accuracy. Phase 5 asks the question that actually matters: can it *drive a
build*, choosing every action itself, living with its own mistakes?

The answer, measured rather than assumed and then re-measured after an audit:
**no policy tested completes a held-out design.** BC scores 0.983 next-action
accuracy and 0.000 closed-loop success; PPO fine-tuning does not change that,
though it does eliminate invalid actions entirely.

## The interpreter problem, and the bridge

The CAD stack runs under FreeCAD's bundled Python 3.11. torch lives in the
system Python 3.12. Installing torch into FreeCAD's interpreter would need
~530 MB, and disk on this machine was already critical, so instead the
environment is **served out of** FreeCAD's interpreter and driven from the one
that has torch:

```text
torch python 3.12                     FreeCAD python 3.11
┌──────────────────────┐   JSON/stdio  ┌──────────────────────────┐
│ PPO, policy, GAE     │ ────────────▶ │ env_server → KairosCADEnv │
│ RemoteCADEnv (client)│ ◀──────────── │ FreeCAD recompute         │
└──────────────────────┘   one line    └──────────────────────────┘
                           per message
```

- `protocol.py`, versioned, newline-delimited JSON. JSON rather than pickle
  because the two ends run different Python versions, where pickle is a
  portability trap. The payloads are a 24-float state and two masks, so
  encoding cost is nothing next to a recompute.
- `env_server.py`, runs under FreeCAD. Its stdout is claimed for the protocol
  and every stray write (FreeCAD prints banners on import) is redirected to
  stderr, because one loose line would desynchronize the stream.
- `env_client.py`, presents `reset`/`step`/`close` to the trainer. A dead or
  hung server restarts and truncates the episode rather than ending the run:
  FreeCAD can segfault or stall on a pathological recompute, and over a
  multi-hour run that is an expected event, not an exceptional one.

Startup to first observation is ~1.2 s, and a 25-iteration run completed with
**0 restarts and 0 timeouts**.

## Algorithm

Standard clipped-surrogate PPO over an actor-critic sharing the VLA trunk, plus
the pieces this domain forces:

- **A KL anchor to the frozen BC policy.** The reasoning: a valid build is a
  long, precisely ordered action sequence with a sparse finish reward, so a
  policy that drifts off the demonstration manifold chasing shaping reward
  should not recover, random exploration does not rediscover a 12-step valid
  build. `bc_kl_coef` weights KL(current ‖ BC); 0 disables it. The ablation
  below tests this and only partly supports it: the anchor does not improve
  success rate, but it does keep invalid actions at zero.
- **Termination and truncation are distinguished.** `FINISH_DESIGN` has no
  future, so its next value is 0; a step-budget cutoff bootstraps from the
  critic. Conflating them would stamp a fabricated terminal penalty on every
  episode that hits the cap, which is most of them.
- **Illegal operations are removed from the distribution, not penalized.**
  Entropy is then measured over the surviving support, so a constrained state
  does not read as an uncertain policy.
- **Parameters use a squashed Gaussian.** Log-probabilities stay finite at 0
  and 1, which is exactly where BC-fitted parameters cluster.
- **Requirements are sampled per episode.** Training against one fixed
  requirement would teach one build and give the language encoder no signal.

### A bug worth recording

The first live run reported `approx_kl = 1.18` after a single update at a 1e-4
learning rate, implausible, and the signature of a real defect: collection ran
with dropout active while the update re-scored under a *different* dropout
sample. PPO's ratio then measures network noise instead of how far the policy
moved. Forcing dropout off in both paths dropped the same measurement to 0.11.
Exploration must come from the action distribution, never from network noise.

## Results


25 iterations x 200 steps against live FreeCAD, initialized from the Phase 4 BC
checkpoint, 40-requirement pool split into train and held-out.

Closed-loop evaluation, 14 episodes per policy on held-out requirements, every
policy facing the same requirements in the same order:

| policy | success | 95% CI | solid | mean reward | steps | invalid actions |
| --- | --- | --- | --- | --- | --- | --- |
| behavioral cloning | 0.000 | [0.00, 0.00] | 1.000 | +0.72 | 23.7 | 0.018 |
| PPO (best checkpoint) | 0.000 | [0.00, 0.00] | 1.000 | +0.72 | 19.4 | 0.000 |
| legal-random baseline | 0.000 | [0.00, 0.00] | 0.143 | -1.68 | 8.1 | 0.265 |

The gap between 0.983 next-action accuracy and 0.000 closed-loop success is
the whole finding. Teacher forcing hands the policy the expert's state at every
step, so per-step errors never compound; driving its own build, it reaches
states the demonstrations never visited. It still produces a valid solid in
every episode. It can pad and pocket. It simply never finishes a design that
satisfies its requirement.

PPO's measurable gain is narrower than previously reported: invalid actions
fall to **zero** and episodes shorten (23.7 → 19.4 steps) at identical mean
reward. It does not convert either into a completed design.

**A correction.** This table previously read 0.286 for PPO. That was an
artifact, not a result: `evaluate_ppo` re-derived its own held-out pool from a
different pool size than training used, and three of the six "held-out"
requirements had been trained on. Runs now record their exact pools and
evaluation refuses to score a contaminated split. During training the loop
still reported a best of 0.188 at iteration 15 on a 16-episode evaluation, which
the 14-episode comparison did not reproduce, at these success rates the
sampling noise is larger than the effect being measured.

The random baseline is legal-random, not noise, and it still never finishes and
produces a solid in only 14% of episodes. That is the floor these numbers sit
above.

### Ablation: does the BC anchor earn its place?

Run twice, identically, changing only `bc_kl_coef` (0.05 vs 0.0), then scored
under the same 14-episode held-out protocol:

| run | success | solid | mean reward | invalid actions |
| --- | --- | --- | --- | --- |
| anchored (`bc_kl_coef=0.05`) | 0.286 | **1.000** | **+1.88** | **0.000** |
| unanchored (`bc_kl_coef=0.0`) | 0.286 | 0.786 | -0.37 | 0.295 |

**The anchor did not change the success rate.** Both reach 0.286, so the
claim that it is "what makes RL viable here" is not supported by this
experiment. It is a hypothesis the data declined to confirm.

What it does buy is stability in everything around the headline number: the
anchored policy produces a valid solid in every episode and emits no invalid
actions at all, while the unanchored one degrades to 79% solids and a 30%
invalid-action rate, with mean reward going negative. That is consistent with
the drift story, just far weaker than "necessary".

There is a second lesson here, and it is about measurement rather than
algorithms. During training the unanchored run reported a held-out success rate
of **0.500**, apparently twice the anchored run, on the 6-episode evaluation
the loop runs every 5 iterations. At 14 episodes it scored 0.286, identical to
the anchored run. Six episodes over six requirements simply cannot separate
those hypotheses, and the best-checkpoint selection inside the loop is picking
on that noisy estimate. `eval_episodes` is now defaulted higher for this
reason, and the reported intervals (`[0.07, 0.50]` for a 0.286 point estimate)
show how much room remains.

### Baseline: does BC initialization actually matter?

Also tested rather than asserted, 12 iterations with `--from-scratch`,
everything else identical:

| iteration | 1 | 4 | 7 | 10 | 12 |
| --- | --- | --- | --- | --- | --- |
| mean reward | -0.19 | -0.26 | -0.29 | **-0.50** | -0.30 |
| episode length | 7.7 | 4.3 | 3.9 | **1.9** | 3.4 |

Held-out success stayed at **0.000** throughout, and the failure is more
interesting than a flat line: reward gets *worse* while episodes get *shorter*.
The policy is not slowly learning to build. It is learning to quit. With no
route to the sparse finish reward, the fastest way to stop losing points to
per-action costs is to terminate immediately, so it converges on ~2-step
episodes that do nothing.

That is the degenerate optimum BC initialization exists to avoid, and unlike
the anchor claim, this one held up.

### What this does not show

- **28.6% is not a solved task.** Roughly seven in ten held-out requirements
  still end without a satisfying design.
- **The held-out pool is 6 requirements over 14 episodes.** That is enough to
  separate 0.286 from 0.000, not enough for a tight confidence interval;
  `bootstrap_interval` is provided for exactly this reason.
- **The BC-KL anchor grew to ~1.8 over the run.** The policy drifts from the
  demonstrations even with the anchor: the coefficient shapes behavior but
  does not pin it, which the ablation above bears out.
- **Constraint satisfaction sat at 0.40 for both policies.** PPO learned to
  finish more often, not to satisfy more constraints; the gain is in
  completing valid builds, not in better engineering.

## Running it

```bash
make setup-learn          # torch, in the system interpreter only
make train-bc             # Phase 4 policy, PPO needs it to start from
make train-ppo            # PPO against the live environment
make eval-ppo             # BC vs PPO vs random, closed loop
```

`--from-scratch` skips BC initialization and `--resume` continues an
interrupted run from `last.pt`, keeping its history and best-so-far.

## Still out of scope

- A `bc_kl_coef` sweep and multiple seeds. The single-seed ablation above is
  suggestive, not conclusive.
- Parallel environments. Rollouts are serial because FreeCAD recomputes are,
  and several bridged servers would multiply throughput straightforwardly.
- Target-head supervision (Phase 4 could not supervise it; PPO trains it only
  indirectly through reward, and it is disabled by default).
- Curriculum over requirement difficulty, and per-family success breakdowns.
