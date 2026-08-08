# Phases 2-3, Dataset scale-up and CAD representation

Phase 2 delivers the procedural dataset at benchmark scale; Phase 3 delivers
the state representation the learning stack consumes. Alongside them this
milestone landed the requirement parser, constraint checker, shaped reward,
and the Gymnasium environment (the non-learning half of Phase 5).

## Design families (`kairos/data/families/`)

Eight self-registering parametric families, each exposing `sample(rng)`,
`is_feasible()`, an action-sequence builder, requirement metadata, and
`expected_holes` for validation:

| family | geometry | exercises |
| --- | --- | --- |
| `l_bracket` | 90° L + per-leg holes + corner fillet | polygon, pad, pocket, fillet, edge search |
| `plate` | rectangle + hole grid | rectangle, through-all pockets |
| `u_bracket` | U-channel + floor & cross-wall holes | coaxial-bore hole grouping |
| `corner_bracket` | L + triangular gusset | multi-pad fusion, rib-clearance feasibility |
| `support_bracket` | base + wall + rib (Task C) | three fused pads, two hole axes |
| `reinforced_plate` | plate + stiffening ribs | offset-sketch pad direction checks |
| `spacer` | revolved annulus + rim chamfers | revolve, circular-edge search, chamfer |
| `flange` | disk + hub + bore + bolt circle | revolve, CIRCULAR_PATTERN of a pocket |

Every family test asserts a **closed-form analytic volume** (rel=1e-6), not
snapshots, plus hole-count validation and seeded feasibility property tests.

## Dataset

`make generate-data` / `scripts/generate_brackets.py` cycles the registry,
rejects infeasible draws and invalid geometry (with recorded reasons), and
writes per design: FCStd/STEP/STL, four rendered views, `state.json`,
`requirements.json`, and `trajectory.json`. Generation shards by
`--start-id`/`--seed` for parallel runs; per-shard stats merge from
`generation_stats_*.json`.

Each design also emits `dataset/trajectories/trajectory_NNNNNN.json`:

```json
{"requirement": ..., "spec": ..., "states": [[24-dim vectors]],
 "actions": [...], "rewards": [...], "final_metrics": {...}}
```

recorded **single-pass** during the build (executor callback → observation →
constraint report → shaped reward), so expert trajectories carry exactly the
reward signal the RL agent will see.

## Representation (`kairos/representation/`)

- `observation.py`, one JSON snapshot per step (summary, holes, faces,
  sketch status). Every downstream consumer takes these dicts, never live
  FreeCAD objects: constraint checking, rewards, and encoders are all
  pure-python testable.
- `numerical_encoder.py`, frozen 24-dim state vector (layout documented in
  `FEATURE_NAMES`; BC data and policies must agree on it).
- `feature_encoder.py`, feature-history token ids / one-hot with PAD/UNK.
- `geometry_graph.py`, typed attributed graph: body/solid/face/edge/vertex/
  sketch/feature/constraint nodes; contains/adjacent_to/created_by/
  constrained_by/depends_on/modified_by relations; numpy arrays ready for a
  PyG wrapper in Phase 4.

## Requirement understanding and evaluation

- `kairos/language/`, deterministic rule-based parser → `EngineeringSpec`
  (typed constraints + objectives). Never invents values; covers all
  benchmark phrasings (M-thread sizes, wall thickness, envelopes, symmetry,
  cylindrical interfaces, mass-reduction %).
- `kairos/evaluation/constraints.py`, checkers resolve each constraint to
  satisfied / violated / **unmeasured**. Unmeasured kinds (min wall
  thickness until Phase 6, symmetry) are excluded from satisfaction rates
  and can never earn reward credit. `hole_diameter` checks the holes the
  requirement names, not every bore in the part: a flange legitimately
  carries a central bore alongside its bolt holes, and wrong *totals* are
  `hole_count`'s job.

Two parser rules exist because the alternative is inventing requirements the
part is meant to violate: a bare `A x B x C mm` triple is only read as the
part envelope when the text does not stack ribs or walls onto it (otherwise
it sizes a sub-component), and the digit in a thread designation (`M4`) is a
diameter, never a quantity.

## Reward and environment (`kairos/rl/`)

- `rewards.py`, episode-scoped shaped reward per the project spec
  (§16-17): one-shot shaping bonuses, per-constraint bonuses, validity
  regression penalty, complexity/action costs, and a mass-progress term
  that only activates while all measured constraints hold (no farming mass
  reduction by skipping requirements). Mass progress is **potential-based**:
  it pays against the lightest constraint-satisfying design so far, so an
  episode's total telescopes to the real improvement and a pad-then-pocket
  cycle earns nothing the second time. A requirement that parses to zero
  constraints is winnable, success is `all_measured_satisfied`, which
  already distinguishes "nothing to check" from "nothing checkable".
- `action_space.py`, codec between policy outputs
  (operation index + [0,1]⁶ params + target index) and validated structured
  actions; documented denormalization ranges; `encode` inverts expert
  actions for BC. `ADD_POLYGON` decodes to a regular 3-8-gon; irregular
  expert profiles (the L and U recipes) are not representable in the fixed
  slots, so `encode` raises `UnrepresentableAction` rather than emit a
  target that decodes into a different shape, BC expands those into
  `ADD_LINE` sequences.
- `environment.py`, `KairosCADEnv` (Gymnasium): Dict observation
  (numeric vector + legality mask), Dict action, per-step info carrying the
  action result, reward breakdown, and constraint report.

## Verification

Full suite under FreeCAD 1.1.3: **251 tests** (`make test-all`), including
27 family tests authored and verified in parallel worktrees. The Phase 4-5
learning tests skip there (no torch) and run under the system interpreter
(**330 tests**), where the CAD tests skip instead. Lint: `ruff` clean.

## Still deliberately out of scope here

- Multimodal fusion / VLA training (Phase 4) and PPO training (Phase 5).
- Minimum-wall-thickness measurement (Phase 6).
- Benchmark runs, baselines, ablations (Phase 7); UI (8); paper (9).
