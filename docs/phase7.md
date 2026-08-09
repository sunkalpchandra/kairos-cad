# Phase 7: The KAIROS-CAD benchmark

Phase 5 ended with every closed-loop number at 0.000: behavioral cloning, PPO
and a legal-random baseline all completed exactly zero held-out designs. A
benchmark whose headline metric is 0.000 for every entrant ranks nothing, so
the first job here was to produce a metric that discriminates before any policy
succeeds.

## Metrics

The headline is a milestone progress score, because `success` is a conjunction
of four gates collapsed into one bit:

```
sketch → geometry → solid → valid solid → holes → all constraints → finished
```

Credit is prefix-scored: it stops at the first milestone missed. A constraint
check passes vacuously on geometry that was never built, so awarding it would
rank an empty document above a real but imperfect part. Milestone weights
strictly dominate (each is worth more than every earlier one combined), so the
ranking of two policies never depends on the exact weights.

Validity, efficiency and constraint satisfaction are reported alongside rather
than folded in. Validity is where PPO beat BC in Phase 5 (0.000 invalid actions
against 0.018), and a single success number hid it.

A caution the metric itself teaches: validity rises when a policy stops
emitting actions at all. Cutting episodes short once took BC's validity from
0.658 to 0.930 while progress fell from 0.435 to 0.321, because the failing
steps stopped being counted. Read validity next to progress, never alone.

## Harness baselines

- **`oracle-replay`** re-executes the recorded expert actions. It should score
  1.000. Anything less is a fault in the harness, the environment or the
  constraint checker, not a policy result.
- **`immediate-finish`** calls `FINISH_DESIGN` at once. It must score bottom.
  PPO trained from scratch converged on this policy, driving episode length to
  ~2 steps because quitting stops paying the per-action cost. Any metric it can
  win is a broken metric.

Both invariants have to be checked per task type. Run across all tasks,
`immediate-finish` scored 0.406 and beat two real policies, which looks damning
until you notice that on `COMPLETE(k=1)` the expert's own final action *is*
`FINISH_DESIGN`, so finishing immediately is the correct answer there. Checked
on `BUILD` alone, where quitting can never be right, it scores 0.000 as it must.

The check covers success, satisfaction and efficiency as well as progress.
Checking progress alone let `immediate-finish` hold a perfect 1.000 efficiency
for quitting in one step, on a baseline whose whole contract is that it must
lose every column.

## Results

<!-- generated: benchmark-tables -->

## leaderboard

| policy | progress score | finished successfully | validity rate | satisfaction rate | efficiency |
| --- | --- | --- | --- | --- | --- |
| `oracle-replay` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `ppo` | 0.472 | 0.382 | 0.705 | 0.486 | 0.613 |
| `bc` | 0.458 | 0.368 | 0.689 | 0.464 | 0.624 |
| `immediate-finish` | 0.318 | 0.237 | 1.000 | 0.404 | 1.000 |
| `scripted-spec` | 0.241 | 0.000 | 1.000 | 0.264 | 0.573 |
| `legal-random` | 0.194 | 0.000 | 0.668 | 0.356 | 0.464 |

## milestone ladder (fraction of episodes reaching each rung)

| policy | opened a sketch | drew geometry | made a solid | solid is valid | has any hole | all constraints met | finished successfully |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `oracle-replay` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `ppo` | 0.89 | 1.00 | 0.75 | 0.75 | 0.67 | 0.41 | 0.38 |
| `bc` | 0.89 | 1.00 | 0.75 | 0.75 | 0.67 | 0.38 | 0.37 |
| `immediate-finish` | 0.64 | 0.64 | 0.64 | 0.64 | 0.50 | 0.24 | 0.24 |
| `scripted-spec` | 1.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.24 | 0.00 |
| `legal-random` | 0.78 | 0.87 | 0.66 | 0.64 | 0.50 | 0.21 | 0.00 |

## success(k): finish the last k actions

| policy | BUILD | k=1 | k=2 | k=4 | k=8 |
| --- | --- | --- | --- | --- | --- |
| `bc` | 0.00 | 1.00 | 0.50 | 0.25 | 0.00 |
| `immediate-finish` | 0.00 | 1.00 | 0.12 | 0.00 | 0.00 |
| `legal-random` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `oracle-replay` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `ppo` | 0.00 | 1.00 | 0.50 | 0.19 | 0.17 |
| `scripted-spec` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## progress by family

| policy | corner_bracket | flange | l_bracket | plate | reinforced_plate | spacer | support_bracket |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bc` | 0.50 | 0.28 | 0.58 | 0.62 | 0.40 | 0.51 | 0.46 |
| `immediate-finish` | 0.32 | 0.28 | 0.31 | 0.31 | 0.27 | 0.38 | 0.32 |
| `legal-random` | 0.22 | 0.19 | 0.17 | 0.18 | 0.17 | 0.18 | 0.23 |
| `oracle-replay` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `ppo` | 0.50 | 0.28 | 0.50 | 0.62 | 0.40 | 0.51 | 0.58 |
| `scripted-spec` | 0.24 | 0.24 | 0.24 | 0.21 | 0.19 | 0.26 | 0.24 |

## paired comparisons (95% bootstrap CI on the per-task difference)

| comparison | difference | 95% CI | W/L/T | separates? |
| --- | --- | --- | --- | --- |
| `legal-random` vs `oracle-replay` | -0.806 | [-0.845, -0.765] | 0/76/0 | yes |
| `oracle-replay` vs `scripted-spec` | +0.759 | [+0.724, +0.792] | 76/0/0 | yes |
| `immediate-finish` vs `oracle-replay` | -0.682 | [-0.766, -0.593] | 0/58/18 | yes |
| `bc` vs `oracle-replay` | -0.542 | [-0.635, -0.449] | 0/48/28 | yes |
| `oracle-replay` vs `ppo` | +0.528 | [+0.433, +0.620] | 47/0/29 | yes |
| `legal-random` vs `ppo` | -0.278 | [-0.351, -0.208] | 2/48/26 | yes |
| `bc` vs `legal-random` | +0.264 | [+0.196, +0.335] | 48/2/26 | yes |
| `ppo` vs `scripted-spec` | +0.231 | [+0.161, +0.307] | 39/19/18 | yes |
| `bc` vs `scripted-spec` | +0.218 | [+0.148, +0.289] | 39/19/18 | yes |
| `immediate-finish` vs `ppo` | -0.153 | [-0.222, -0.092] | 0/35/41 | yes |
| `bc` vs `immediate-finish` | +0.140 | [+0.082, +0.206] | 35/0/41 | yes |
| `immediate-finish` vs `legal-random` | +0.124 | [+0.072, +0.181] | 18/10/48 | yes |
| `immediate-finish` vs `scripted-spec` | +0.077 | [+0.025, +0.133] | 18/27/31 | yes |
| `legal-random` vs `scripted-spec` | -0.047 | [-0.063, -0.031] | 0/28/48 | yes |
| `bc` vs `ppo` | -0.013 | [-0.060, +0.030] | 2/4/70 | **no** |

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
| `bc+blank-req` | 0.512 | +11.7% | 0.395 | 0.895 | 0.565 | 0.612 |
| `bc` | 0.458 | +0.0% | 0.368 | 0.689 | 0.464 | 0.624 |
| `bc+no-mask` | 0.449 | -2.0% | 0.368 | 0.588 | 0.473 | 0.631 |
| `bc+shuffled-req` | 0.423 | -7.8% | 0.303 | 0.714 | 0.488 | 0.593 |

## milestone ladder (fraction of episodes reaching each rung)

| policy | opened a sketch | drew geometry | made a solid | solid is valid | has any hole | all constraints met | finished successfully |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bc+blank-req` | 0.89 | 0.89 | 0.89 | 0.89 | 0.80 | 0.42 | 0.39 |
| `bc` | 0.89 | 1.00 | 0.75 | 0.75 | 0.67 | 0.38 | 0.37 |
| `bc+no-mask` | 0.89 | 0.95 | 0.70 | 0.70 | 0.64 | 0.38 | 0.37 |
| `bc+shuffled-req` | 0.89 | 1.00 | 0.83 | 0.83 | 0.67 | 0.34 | 0.30 |

## success(k): finish the last k actions

| policy | BUILD | k=1 | k=2 | k=4 | k=8 |
| --- | --- | --- | --- | --- | --- |
| `bc` | 0.00 | 1.00 | 0.50 | 0.25 | 0.00 |
| `bc+blank-req` | 0.06 | 1.00 | 0.44 | 0.31 | 0.08 |
| `bc+no-mask` | 0.00 | 1.00 | 0.50 | 0.25 | 0.00 |
| `bc+shuffled-req` | 0.00 | 0.94 | 0.38 | 0.12 | 0.00 |

## progress by family

| policy | corner_bracket | flange | l_bracket | plate | reinforced_plate | spacer | support_bracket |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `bc` | 0.50 | 0.28 | 0.58 | 0.62 | 0.40 | 0.51 | 0.46 |
| `bc+blank-req` | 0.58 | 0.32 | 0.50 | 0.81 | 0.55 | 0.45 | 0.65 |
| `bc+no-mask` | 0.50 | 0.28 | 0.58 | 0.56 | 0.37 | 0.51 | 0.44 |
| `bc+shuffled-req` | 0.55 | 0.30 | 0.45 | 0.54 | 0.48 | 0.42 | 0.41 |

## paired comparisons (95% bootstrap CI on the per-task difference)

| comparison | difference | 95% CI | W/L/T | separates? |
| --- | --- | --- | --- | --- |
| `bc+blank-req` vs `bc+shuffled-req` | +0.089 | [+0.021, +0.158] | 23/6/47 | yes |
| `bc+blank-req` vs `bc+no-mask` | +0.063 | [-0.012, +0.135] | 25/7/44 | **no** |
| `bc` vs `bc+blank-req` | -0.053 | [-0.125, +0.021] | 7/23/46 | **no** |
| `bc` vs `bc+shuffled-req` | +0.036 | [-0.014, +0.091] | 14/12/50 | **no** |
| `bc+no-mask` vs `bc+shuffled-req` | +0.027 | [-0.022, +0.082] | 11/15/50 | **no** |
| `bc` vs `bc+no-mask` | +0.009 | [+0.001, +0.018] | 5/0/71 | yes |

<!-- /generated -->

**The requirement ablations do not reproduce across seeds, and that is the
result.** This suite has now been run twice against checkpoints that differ
only in initialization noise (the second dropped 115,089 vision parameters that
never received a gradient, which shifts the RNG draw but nothing else). The
mask condition agrees to within half a point. Both requirement conditions
change sign:

| condition | run A | run B |
| --- | --- | --- |
| `bc+shuffled-req` | **+3.7%** | **-7.8%** |
| `bc+blank-req` | +1.8% | +11.7% |
| `bc+no-mask` | -2.3% | -2.0% |

An effect that flips sign between two seeds is smaller than the seed variance,
so no claim about requirement conditioning survives from this data. Earlier
versions of this document asserted first that shuffling costs 22.9% and the
policy therefore reads its requirement, then that success was identical and it
therefore does not. Both were single-run readings of a measurement too noisy to
support either.

What the suite can currently support:

- **The mask does a small, reproducible amount of work.** Removing it costs
  about 2% of progress in both runs and drops validity by 10 to 17 points. That
  is the one condition with a stable sign.
- **Nothing about the requirement**, in either direction, until this is run
  across several seeds with intervals. `paired_bootstrap` already exists in
  `kairos/benchmark/statistics.py` and is applied to the leaderboard; extending
  it over ablation seeds is the obvious next step and is not done.

Two structural caveats compound the noise and are worth stating rather than
explaining away. Most tasks are `COMPLETE(k)`, where a replayed expert prefix
already fixes the geometry, so the requirement has less left to determine. And
requirement texts are templated per family, so family identity is partly
recoverable from geometry alone.

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
