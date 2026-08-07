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
| `oracle-replay` | **0.594** | 0.500 | 0.924 | 0.656 | 1.000 |
| `scripted-spec` | 0.224 | 0.000 | 1.000 | 0.211 | 0.640 |
| `immediate-finish` | 0.217 | 0.156 | 1.000 | 0.266 | 1.000 |
| `legal-random` | 0.146 | 0.000 | 0.676 | 0.268 | 0.604 |

Milestone attainment shows where each dies:

| policy | sketch | geometry | solid | valid | holes | constraints | finished |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `oracle-replay` | 1.00 | 1.00 | 0.78 | 0.78 | 0.75 | 0.59 | 0.50 |
| `scripted-spec` | 1.00 | 1.00 | **1.00** | **1.00** | 0.47 | 0.19 | 0.00 |
| `immediate-finish` | 0.44 | 0.44 | 0.44 | 0.44 | 0.38 | 0.16 | 0.16 |
| `legal-random` | 0.66 | 0.69 | 0.47 | 0.47 | 0.38 | 0.16 | 0.00 |

The scripted baseline builds a valid solid **every single time** and then dies
at holes — it places them, but rarely the right count in the right places. That
is a far more actionable diagnosis than "success 0.000", which is all the
Phase 5 metric could say about it.

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

- **BC and PPO are not yet in the table.** They need a policy adapter that
  encodes observations into tensors; the baselines run without one.
- The `success(k)` curve over `COMPLETE` suffix lengths — the tasks exist and
  are scored, but the curve is not yet plotted or reported as a headline.
- Ablations: requirement blanking (does the policy read the requirement at
  all?), masking on/off, and a `bc_kl_coef` sweep with multiple seeds.
- Paired statistics. Every policy faces identical tasks, which makes paired
  bootstrap CIs the correct test; the current tables report point estimates.
