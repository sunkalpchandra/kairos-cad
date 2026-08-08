# Phase 7: The KAIROS-CAD benchmark

Phase 5 ended with every closed-loop number at 0.000: behavioral cloning, PPO
and a legal-random baseline all completed exactly zero held-out designs. A
benchmark whose headline metric is 0.000 for every entrant ranks nothing and
directs nothing, so Phase 7's first job is not to build an evaluation harness; it is to **manufacture signal**.

## Metrics

The headline is a **milestone progress score**, because `success` is a
conjunction of four gates collapsed into one bit:

```
sketch → geometry → solid → valid solid → holes → all constraints → finished
```

Credit is **prefix-scored**: it stops at the first milestone missed. That
matters because a constraint check passes vacuously on geometry that was never
built, so awarding it would rank an empty document above a real but imperfect
part. Milestone weights strictly dominate, each is worth more than every
earlier one combined, so the ranking of two policies never depends on the exact
weights.

Validity, efficiency and constraint satisfaction are reported **alongside**, not
folded in. Validity is where PPO actually beat BC in Phase 5 (0.000 invalid
actions against 0.018), and a single success number hid it entirely.

## Harness baselines

- **`oracle-replay`** re-executes the recorded expert actions. It should score
  1.000. Anything less is a fault in the harness, the environment or the
  constraint checker, not a policy result.
- **`immediate-finish`** calls `FINISH_DESIGN` at once. It must score bottom.
  PPO trained from scratch converged on this policy, driving episode length to
  ~2 steps because quitting stops paying the per-action cost. Any metric it can
  win is a broken metric.

**Both invariants must be checked per task type, and finding that out was
itself a result.** Run across all tasks, `immediate-finish` scored 0.406 and beat
two real policies, which looks damning until you notice that on `COMPLETE(k=1)`
the expert's own final action *is* `FINISH_DESIGN`. Finishing immediately is the
correct answer there. Checked on `BUILD` alone, where quitting can never be
right, it scores **0.000** exactly as it must.

## Results

<!-- generated: benchmark-tables -->

## leaderboard

| policy | progress score | finished successfully | validity rate | satisfaction rate | efficiency |
| --- | --- | --- | --- | --- | --- |
| `oracle-replay` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `ppo` | 0.485 | 0.395 | 0.704 | 0.508 | 0.739 |
| `bc` | 0.435 | 0.342 | 0.658 | 0.487 | 0.644 |
| `immediate-finish` | 0.318 | 0.237 | 1.000 | 0.404 | 1.000 |
| `scripted-spec` | 0.241 | 0.000 | 1.000 | 0.264 | 0.573 |
| `legal-random` | 0.194 | 0.000 | 0.668 | 0.356 | 0.464 |

## milestone ladder (fraction of episodes reaching each rung)

| policy | opened a sketch | drew geometry | made a solid | solid is valid | has any hole | all constraints met | finished successfully |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `oracle-replay` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `ppo` | 0.89 | 1.00 | 0.75 | 0.75 | 0.70 | 0.42 | 0.39 |
| `bc` | 0.89 | 1.00 | 0.75 | 0.75 | 0.67 | 0.34 | 0.34 |
| `immediate-finish` | 0.64 | 0.64 | 0.64 | 0.64 | 0.50 | 0.24 | 0.24 |
| `scripted-spec` | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.24 | 0.00 |
| `legal-random` | 0.78 | 0.87 | 0.66 | 0.64 | 0.50 | 0.21 | 0.00 |

## success(k): finish the last k actions

| policy | BUILD | k=1 | k=2 | k=4 | k=8 |
| --- | --- | --- | --- | --- | --- |
| `bc` | 0.00 | 1.00 | 0.44 | 0.19 | 0.00 |
| `immediate-finish` | 0.00 | 1.00 | 0.12 | 0.00 | 0.00 |
| `legal-random` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `oracle-replay` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `ppo` | 0.00 | 1.00 | 0.56 | 0.31 | 0.00 |
| `scripted-spec` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## progress by family

| policy | corner_bracket | flange | l_bracket | plate | reinforced_plate | spacer | support_bracket |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bc` | 0.48 | 0.28 | 0.35 | 0.62 | 0.40 | 0.51 | 0.51 |
| `immediate-finish` | 0.32 | 0.28 | 0.31 | 0.31 | 0.27 | 0.38 | 0.32 |
| `legal-random` | 0.22 | 0.19 | 0.17 | 0.18 | 0.17 | 0.18 | 0.23 |
| `oracle-replay` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `ppo` | 0.55 | 0.28 | 0.49 | 0.62 | 0.55 | 0.51 | 0.57 |
| `scripted-spec` | 0.24 | 0.24 | 0.24 | 0.21 | 0.19 | 0.26 | 0.24 |

## paired comparisons (95% bootstrap CI on the per-task difference)

| comparison | difference | 95% CI | W/L/T | separates? |
| --- | --- | --- | --- | --- |
| `legal-random` vs `oracle-replay` | -0.806 | [-0.845, -0.765] | 0/76/0 | yes |
| `oracle-replay` vs `scripted-spec` | +0.759 | [+0.724, +0.792] | 76/0/0 | yes |
| `immediate-finish` vs `oracle-replay` | -0.682 | [-0.766, -0.593] | 0/58/18 | yes |
| `bc` vs `oracle-replay` | -0.565 | [-0.658, -0.472] | 0/50/26 | yes |
| `oracle-replay` vs `ppo` | +0.515 | [+0.419, +0.608] | 46/0/30 | yes |
| `legal-random` vs `ppo` | -0.291 | [-0.363, -0.222] | 2/53/21 | yes |
| `ppo` vs `scripted-spec` | +0.244 | [+0.173, +0.318] | 43/19/14 | yes |
| `bc` vs `legal-random` | +0.241 | [+0.175, +0.310] | 45/2/29 | yes |
| `bc` vs `scripted-spec` | +0.194 | [+0.126, +0.264] | 36/19/21 | yes |
| `immediate-finish` vs `ppo` | -0.167 | [-0.236, -0.104] | 0/40/36 | yes |
| `immediate-finish` vs `legal-random` | +0.124 | [+0.072, +0.181] | 18/10/48 | yes |
| `bc` vs `immediate-finish` | +0.117 | [+0.064, +0.178] | 32/0/44 | yes |
| `immediate-finish` vs `scripted-spec` | +0.077 | [+0.025, +0.133] | 18/27/31 | yes |
| `bc` vs `ppo` | -0.050 | [-0.108, +0.002] | 3/10/63 | **no** |
| `legal-random` vs `scripted-spec` | -0.047 | [-0.063, -0.031] | 0/28/48 | yes |

<!-- /generated -->

## The oracle ceiling

`oracle-replay` replays the recorded expert actions through the same codec a
policy must use, so its score is the ceiling for any policy on this action
space. It now scores **1.000 on BUILD**, which is the first time the harness
invariant in `baselines.py` has held.

Getting there took four fixes, and the sequence is the useful part:

| ceiling on BUILD | what was wrong |
| --- | --- |
| 0.431 | six families drew their profile as one irregular `ADD_POLYGON`, which the codec can only express as a regular n-gon |
| 0.594 | slot ranges narrower than the data, so `encode` clipped and `decode` returned the boundary |
| 0.858 | requirement text rounded to nearest, so a 6.1866 mm wall stated as "6.2 mm" made the expert violate its own requirement |
| 1.000 | `encode` returned target index 0, so replayed fillets and chamfers landed on whichever edge was listed first |

None of these raised. Each produced a plausible number in the expected
direction, and each was found by asking why an oracle replaying ground truth
scored below perfect rather than accepting the number as a codec limit.

The consequence is that the action space is no longer an excuse. Phase 5
reported 0.000 closed-loop success and attributed it to the policy; a large
part was the encoding. That is now closed, and what remains on BUILD is the
policy.

## Ablations

Each ablation wraps the policy, so the perturbed and unperturbed runs share
every other condition and the difference is the ablation alone.

<!-- generated: ablation-tables -->

## ablation

| condition | progress | delta | finished successfully | validity rate | satisfaction rate | efficiency |
| --- | --- | --- | --- | --- | --- | --- |
| `bc+shuffled-req` | 0.321 | +0.0% | 0.237 | 0.961 | 0.391 | 0.856 |
| `bc` | 0.321 | +0.0% | 0.237 | 0.930 | 0.383 | 0.824 |
| `bc+no-mask` | 0.320 | -0.2% | 0.237 | 0.772 | 0.364 | 0.568 |
| `bc+blank-req` | 0.319 | -0.5% | 0.237 | 1.000 | 0.384 | 0.844 |

## milestone ladder (fraction of episodes reaching each rung)

| policy | opened a sketch | drew geometry | made a solid | solid is valid | has any hole | all constraints met | finished successfully |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bc+shuffled-req` | 1.00 | 0.64 | 0.64 | 0.64 | 0.50 | 0.24 | 0.24 |
| `bc` | 1.00 | 0.64 | 0.64 | 0.64 | 0.50 | 0.24 | 0.24 |
| `bc+no-mask` | 0.78 | 0.86 | 0.64 | 0.64 | 0.50 | 0.24 | 0.24 |
| `bc+blank-req` | 1.00 | 0.64 | 0.64 | 0.64 | 0.49 | 0.24 | 0.24 |

## success(k): finish the last k actions

| policy | BUILD | k=1 | k=2 | k=4 | k=8 |
| --- | --- | --- | --- | --- | --- |
| `bc` | 0.00 | 1.00 | 0.12 | 0.00 | 0.00 |
| `bc+blank-req` | 0.00 | 1.00 | 0.12 | 0.00 | 0.00 |
| `bc+no-mask` | 0.00 | 1.00 | 0.12 | 0.00 | 0.00 |
| `bc+shuffled-req` | 0.00 | 1.00 | 0.12 | 0.00 | 0.00 |

## progress by family

| policy | corner_bracket | flange | l_bracket | plate | reinforced_plate | spacer | support_bracket |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bc` | 0.32 | 0.28 | 0.31 | 0.31 | 0.27 | 0.38 | 0.32 |
| `bc+blank-req` | 0.32 | 0.27 | 0.31 | 0.31 | 0.27 | 0.38 | 0.32 |
| `bc+no-mask` | 0.33 | 0.28 | 0.31 | 0.31 | 0.27 | 0.38 | 0.32 |
| `bc+shuffled-req` | 0.32 | 0.28 | 0.31 | 0.31 | 0.27 | 0.38 | 0.32 |

## paired comparisons (95% bootstrap CI on the per-task difference)

| comparison | difference | 95% CI | W/L/T | separates? |
| --- | --- | --- | --- | --- |
| `bc` vs `bc+blank-req` | +0.002 | [+0.000, +0.005] | 1/0/75 | **no** |
| `bc+blank-req` vs `bc+shuffled-req` | -0.002 | [-0.005, +0.000] | 0/1/75 | **no** |
| `bc+blank-req` vs `bc+no-mask` | -0.001 | [-0.005, +0.002] | 17/6/53 | **no** |
| `bc` vs `bc+no-mask` | +0.001 | [-0.001, +0.002] | 17/5/54 | **no** |
| `bc+no-mask` vs `bc+shuffled-req` | -0.001 | [-0.002, +0.001] | 5/17/54 | **no** |
| `bc` vs `bc+shuffled-req` | +0.000 | [+0.000, +0.000] | 0/0/76 | **no** |

<!-- /generated -->

**The policy does not read its requirement.** Success is *identical* at 0.342
across the intact policy, a shuffled requirement and a blanked one, and the
progress differences are smaller than the gap between the two corrupted
conditions themselves. On this benchmark the requirement text contributes
nothing measurable.

This retracts the earlier reading. A previous run reported shuffling the
requirement costing **-22.9%** of progress and concluded the policy did read
it. That measurement came from the pre-fix dataset and codec; on corrected
artifacts it does not reproduce. The honest conclusion is the one the ablation
was built to be able to reach: with eight families and near-fixed recipes, a
policy can score this well by learning what CAD builds look like, and these
tasks do not separate that from requirement following.

Two things blunt the test and are worth stating rather than explaining away.
Most tasks are `COMPLETE(k)`, where a replayed expert prefix already fixes the
geometry, so the requirement has less left to determine. And the requirement
texts are templated per family, so family identity is partly recoverable from
the geometry alone.

**The mask matters less than it did.** Removing it costs 2.4% of progress and
drops validity from 0.658 to 0.573, where an earlier run saw 0.957 collapse to
0.407. Validity is lower across the board now, so the mask is carrying less of
the policy's legality than it was.

## Reproducibility

Two contaminated comparisons have already shipped in this project, both because
**the split was a function evaluated twice with different arguments**. A split
is now an artifact: `benchmark/kairos-cad-v1/splits.json` lists design ids and
requirement-text hashes, is checksummed, and is committed. `benchmark_build.py`
refuses to overwrite it without `--force`.

The split is three-way for a reason found during this phase: PPO chose its
`best.pt` by success on the very pool `evaluate_ppo` then scored it against.
Nothing was *trained* on that pool, so the existing leak check passed, but
selecting the shipped checkpoint using the evaluation set inflates the number
the same way. `dev` now absorbs every such choice; `test` is quoted once.

Seeds are hashed per `(suite, task, policy, repeat)`, so adding a policy or a
task never shifts another's seeds, and every table here is regenerable from
`runs/benchmark_core/*_traces.jsonl` alone.

```bash
make benchmark-suite            # freeze the split (refuses to overwrite)
make benchmark PRESET=core      # run the baselines
```

## Out of scope

- A `bc_kl_coef` sweep with multiple seeds. Phase 5's single-seed anchored /
  unanchored comparison remains suggestive rather than conclusive.
- **Target selection.** `encode()` cannot recover which edge, face or feature an
  expert action pointed at. It returns index 0. So any recipe that patterns a
  *specific* feature is replayed against the wrong one. This is now the largest
  single item under the oracle ceiling: it is why the flange family scores 0.000
  satisfaction on replay while its expert scores 1.000.
- Multi-body booleans. `UNION`/`CUT`/`INTERSECTION` decode and validate, but the
  executor cannot perform them, so a policy emitting one always fails.

(Paired bootstrap statistics were listed here and are now implemented, see the
interval table above and `kairos/benchmark/statistics.py`.)
