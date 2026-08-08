# Phase 7 — The KAIROS-CAD benchmark

Phase 5 ended with every closed-loop number at 0.000: behavioral cloning, PPO
and a legal-random baseline all completed exactly zero held-out designs. A
benchmark whose headline metric is 0.000 for every entrant ranks nothing and
directs nothing, so Phase 7's first job is not to build an evaluation harness —
it is to **manufacture signal**.

## Metrics that discriminate when nothing succeeds

The headline is a **milestone progress score**, because `success` is a
conjunction of four gates collapsed into one bit:

```
sketch → geometry → solid → valid solid → holes → all constraints → finished
```

Credit is **prefix-scored**: it stops at the first milestone missed. That
matters because a constraint check passes vacuously on geometry that was never
built, so awarding it would rank an empty document above a real but imperfect
part. Milestone weights strictly dominate — each is worth more than every
earlier one combined — so the ranking of two policies never depends on the exact
weights.

Validity, efficiency and constraint satisfaction are reported **alongside**, not
folded in. Validity is where PPO actually beat BC in Phase 5 (0.000 invalid
actions against 0.018), and a single success number hid it entirely.

## Two baselines that audit the benchmark, not a policy

- **`oracle-replay`** re-executes the recorded expert actions. It should score
  1.000. Anything less is a fault in the harness, the environment or the
  constraint checker — not a policy result.
- **`immediate-finish`** calls `FINISH_DESIGN` at once. It must score bottom.
  This is not hypothetical: PPO trained from scratch converged on exactly this
  policy, driving episode length to ~2 steps because quitting stops paying the
  per-action cost. Any metric it can win is a broken metric.

**Both invariants must be checked per task type, and finding that out was
itself a result.** Run across all tasks, `immediate-finish` scored 0.406 and beat
two real policies — which looks damning until you notice that on `COMPLETE(k=1)`
the expert's own final action *is* `FINISH_DESIGN`. Finishing immediately is the
correct answer there. Checked on `BUILD` alone, where quitting can never be
right, it scores **0.000** exactly as it must.

## Results

32 tasks from the frozen held-out test split (`benchmark/kairos-cad-v1`, 756
train / 162 dev / 162 test), every policy facing identical tasks in identical
order:

| policy | progress | success | valid | constraints | efficiency |
| --- | --- | --- | --- | --- | --- |
| `oracle-replay` (ceiling) | **0.594** | 0.500 | 0.924 | 0.656 | 1.000 |
| `bc` | **0.445** | 0.281 | 0.957 | 0.453 | 0.620 |
| `ppo` | 0.413 | 0.250 | 0.951 | 0.477 | 0.675 |
| `scripted-spec` | 0.224 | 0.000 | 1.000 | 0.211 | 0.640 |
| `immediate-finish` | 0.217 | 0.156 | 1.000 | 0.266 | 1.000 |
| `legal-random` | 0.146 | 0.000 | 0.676 | 0.268 | 0.604 |

Milestone attainment shows where each dies:

| policy | sketch | geometry | solid | valid | holes | constraints | finished |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `oracle-replay` | 1.00 | 1.00 | 0.78 | 0.78 | 0.75 | 0.59 | 0.50 |
| `bc` | 1.00 | 1.00 | **1.00** | **1.00** | 0.84 | 0.31 | 0.28 |
| `ppo` | 1.00 | 1.00 | **1.00** | **1.00** | 0.72 | 0.31 | 0.25 |
| `scripted-spec` | 1.00 | 1.00 | **1.00** | **1.00** | 0.47 | 0.19 | 0.00 |
| `immediate-finish` | 0.44 | 0.44 | 0.44 | 0.44 | 0.38 | 0.16 | 0.16 |
| `legal-random` | 0.66 | 0.69 | 0.47 | 0.47 | 0.38 | 0.16 | 0.00 |

**BC and PPO succeed on 28% and 25% of tasks — not 0.000.** Phase 5 measured
only full builds from an empty document and reported zero for everything. Given
a partially built part to finish, both complete a quarter of the work. The
capability was there; the Phase 5 task simply could not see it.

Both also reach a valid solid on **every** task, beating the oracle's 0.78 —
because the oracle's codec-degraded polygon builds sometimes fail outright
while a learned policy reaches *some* valid solid regardless. They then die at
the same rung the scripted baseline does: holes to constraints (0.84 → 0.31).
Getting the right holes in the right places is the binding failure for every
policy tested, which is a far more actionable diagnosis than "success 0.000".

BC appears to edge PPO (0.445 vs 0.413), **but the paired test says the
benchmark cannot separate them**: the per-task difference is +0.031 with a 95%
interval of [−0.008, +0.091] over 32 paired tasks, 5 wins to 1 loss with 26
ties. Reporting "BC beats PPO" from those point estimates would have been
exactly the error Phase 5 made when a 6-episode evaluation produced 0.500 for a
policy that scored 0.286.

What *does* separate, with intervals excluding zero:

| comparison | difference | 95% CI | W/L/T |
| --- | --- | --- | --- |
| `bc` vs `legal-random` | +0.298 | [+0.204, +0.402] | 27/2/3 |
| `bc` vs `scripted-spec` | +0.220 | [+0.126, +0.323] | 21/2/9 |
| `ppo` vs `scripted-spec` | +0.189 | [+0.098, +0.291] | 18/3/11 |
| `bc` vs `oracle-replay` | −0.149 | [−0.276, −0.035] | 7/10/15 |
| `bc` vs `ppo` | +0.031 | [−0.008, +0.091] | 5/1/26 |

Both learned policies clear the scripted null hypothesis and the random floor,
and BC's remaining gap to the oracle is small (−0.149) — most of the headroom on
these tasks is the codec ceiling, not the policy.

## The compounding-error curve

The `COMPLETE(k)` tasks exist to measure one thing: how fast a policy degrades
as it must supply more of its own actions. It does, cleanly:

| policy | BUILD | k=1 | k=2 | k=4 | k=8 |
| --- | --- | --- | --- | --- | --- |
| `oracle-replay` | 0.38 | 0.83 | 0.60 | 0.33 | 0.50 |
| `bc` | 0.00 | **0.83** | **0.60** | **0.33** | 0.00 |
| `ppo` | 0.00 | 0.83 | 0.60 | 0.00 | 0.00 |
| `immediate-finish` | 0.00 | 0.83 | 0.00 | 0.00 | 0.00 |
| `scripted-spec` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `legal-random` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

BC decays 0.83 → 0.60 → 0.33 → 0.00. **This is the number Phase 4's 0.983
teacher-forced accuracy and Phase 5's 0.000 closed-loop success were the two
endpoints of** — and it took a task type that hands the policy a partially built
part to see the middle of it at all.

BC matches the oracle exactly at k ≤ 4, which says the remaining gap on those
tasks is the action codec, not the policy. PPO tracks BC to k=2 and then falls
away faster, so its Phase 5 advantage (zero invalid actions) does not extend to
multi-step planning.

`immediate-finish` scoring 0.83 at k=1 is the control working: the expert's last
action usually *is* FINISH_DESIGN, so a policy that only knows how to quit gets
that one right. It collapses to 0.00 the moment k=2 asks for anything else.

Per-family progress locates the difficulty: the oracle reaches 1.00 on `plate`,
`reinforced_plate` and `support_bracket` and only 0.35–0.37 on `flange` and
`corner_bracket` — the families whose profiles the codec cannot express.

## The finding that recontextualizes Phase 5

**`oracle-replay` scores 0.431 on BUILD tasks, not 1.000.**

The oracle replays the recorded expert actions through the same action codec a
policy must use. It cannot reproduce the expert's own build, because six of the
eight families draw their profile with an irregular `ADD_POLYGON` that the
codec can only express as a regular n-gon.

So **0.431 is the ceiling for any policy on BUILD tasks**. Phase 5 reported
0.000 closed-loop success and attributed it to the policy. A substantial part
of it is the action space: no policy, however good, can build those parts
through this codec. That does not excuse 0.000 — the ceiling is 0.431, not
0.000 — but any future claim about policy quality on BUILD has to be read
against it rather than against 1.0.

This is exactly what an auditing baseline is for. It was cheap, it ran first,
and it changed the interpretation of every number downstream.

## Ablations

Each ablation wraps the policy, so the perturbed and unperturbed runs share
every other condition and the difference is the ablation alone. Run on BC over
the same 32 tasks:

| run | progress | Δ | success | validity | constraints |
| --- | --- | --- | --- | --- | --- |
| `bc` | 0.445 | — | 0.281 | 0.957 | 0.453 |
| `bc+blank-req` | 0.374 | −15.9% | 0.188 | 1.000 | 0.487 |
| `bc+shuffled-req` | 0.343 | **−22.9%** | 0.156 | 0.955 | 0.372 |
| `bc+no-mask` | 0.339 | −23.9% | 0.281 | **0.407** | 0.385 |

**The policy does read the requirement.** Handed another task's requirement it
loses 23% of its progress and nearly half its success (0.281 → 0.156). I
expected the opposite — that a policy trained on eight families with near-fixed
recipes would have learned a build prior and ignored the text — and the
experiment refuted it. Blanking the requirement costs less (−15.9%) than
swapping in a wrong one, which is the right ordering: absent information is
less damaging than misleading information.

**But the low invalid-action rate is mostly the mask, not the policy.** Strip
the legality mask and BC's validity collapses from 0.957 to **0.407** — while
its success rate does not move at all (0.281 either way). So the mask is doing
the work of keeping actions legal, and the policy is doing the work of choosing
*which* legal action. That directly qualifies Phase 5's headline PPO gain
(invalid actions 0.018 → 0.000): a large part of that number belongs to the
environment's masking, not to what PPO learned.

## Reproducibility

Two contaminated comparisons have already shipped in this project, both because
**the split was a function evaluated twice with different arguments**. A split
is now an artifact: `benchmark/kairos-cad-v1/splits.json` lists design ids and
requirement-text hashes, is checksummed, and is committed. `benchmark_build.py`
refuses to overwrite it without `--force`.

The split is three-way for a reason found during this phase: PPO chose its
`best.pt` by success on the very pool `evaluate_ppo` then scored it against.
Nothing was *trained* on that pool, so the existing leak check passed — but
selecting the shipped checkpoint using the evaluation set inflates the number
the same way. `dev` now absorbs every such choice; `test` is quoted once.

Seeds are hashed per `(suite, task, policy, repeat)`, so adding a policy or a
task never shifts another's seeds — and every table here is regenerable from
`runs/benchmark_core/*_traces.jsonl` alone.

```bash
make benchmark-suite            # freeze the split (refuses to overwrite)
make benchmark PRESET=core      # run the baselines
```

## Still out of scope

- A `bc_kl_coef` sweep with multiple seeds. Phase 5's single-seed anchored /
  unanchored comparison remains suggestive rather than conclusive.
- **Target selection.** `encode()` cannot recover which edge, face or feature an
  expert action pointed at — it returns index 0 — so any recipe that patterns a
  *specific* feature is replayed against the wrong one. This is now the largest
  single item under the oracle ceiling: it is why the flange family scores 0.000
  satisfaction on replay while its expert scores 1.000.
- Multi-body booleans. `UNION`/`CUT`/`INTERSECTION` decode and validate, but the
  executor cannot perform them, so a policy emitting one always fails.

(Paired bootstrap statistics were listed here and are now implemented — see the
interval table above and `kairos/benchmark/statistics.py`.)
