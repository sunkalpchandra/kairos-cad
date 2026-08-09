# Phase 8: Interactive dashboard

One HTML file. Open it from disk, no server, no network, no build step:

```bash
make dashboard          # -> docs/dashboard.html
open docs/dashboard.html
```

It reads only committed artifacts, `dataset/designs/*/`, `runs/*/report.json`,
`runs/*/*_traces.jsonl`, `runs/*/leaderboard.json`. Nothing is computed from a
live model at page-build time, so a number on screen can always be traced to a
file on disk. This is the same discipline `benchmark_report.py` follows, for the
same reason: a dashboard that can disagree with the repo is worse than no
dashboard.

## What it shows

**Designs.** Every embedded part as real tessellated geometry, orbitable, beside
its requirement text, its measured properties, its per-constraint pass/fail with
the measured value, and the expert's action sequence.

**Benchmark.** The leaderboard, the success-against-difficulty curve, and paired
bootstrap intervals. `oracle-replay` is marked as what it is, a ceiling, not a
competitor.

**Training.** BC accuracy and loss per epoch (train and held-out), PPO reward
and rates per iteration.

**Ablations.** Each corrupted-input condition as a delta against the intact
policy, since the absolute score of `bc+shuffled-req` means little alone.

## Deviations from the plan

**No Three.js.** The plan named it; this ships a ~250-line WebGL renderer
instead. Three.js minified is ~600 KB, six times the size of every part,
metric and curve in the bundle put together, and the one constraint that
actually decides this dashboard's shape is that it must be a single file that
opens offline. What the viewer needs is to orbit a static mesh and shade it.

**No charting library**, for the same arithmetic. The charts are SVG strings,
which keeps them crisp at any zoom.

**Meshes come from the STLs**, not from re-meshing FCStd files. Every generated
design already ships `model.stl`, so the whole dashboard builds under the torch
interpreter with no FreeCAD subprocess.

## Mesh budget

| step | effect |
| --- | --- |
| binary STL → welded vertices | 744 raw corners → 122 on a typical bracket |
| float64 → integers quantized to 0.01 mm | ~20 chars per coordinate → ~4 |
| result | the JSON mesh is *smaller than the STL it came from* |

The 0.01 mm quantum sits an order of magnitude below the 0.1 mm tolerance every
constraint check in this repo uses, so it cannot change what the viewer shows
about a part that passed or failed.

## Bugs the screenshots caught

Everything below typechecked, linted, and produced a page that looked finished.

**1. Smooth normals melted the machined edges.** Welding is what makes the mesh
compact, but averaging normals across a welded 90° edge shades every corner as
if it were filleted, and the tessellation's diagonals appear as creases across
faces that are dead flat. Fixed with per-fragment face normals from screen-space
derivatives. The derivative cross product's sign follows screen orientation
rather than winding, so `gl_FrontFacing` cannot correct it, every visible
fragment of a closed solid faces the camera, so the normal is oriented against
the view ray instead.

**2. The whole success(k) curve collapsed to one point.** Task ids look like
`complete-k4-design_000000`. Splitting on `-` and testing `isdigit()` on the
middle field reads `k4` as non-numeric, so every COMPLETE task was filed under
`build`. No error anywhere; the chart simply drew a single dot. With the `k`
prefix stripped, the curve reproduces the documented BC decay exactly:
0.833 → 0.600 → 0.333 → 0.000.

**3. The held-out BC curve was missing from its own chart.** The report names
that series `operation_accuracy`. Nothing in it is called `dev_accuracy` or
`val_accuracy`, so those lookups returned null for every epoch and the series
was filtered out, leaving a chart of *training* accuracy alone, which is the
one curve that proves nothing.

**4. Bounding box and wall thickness showed a dash.** The state stores span per
axis as `x_len`/`y_len`/`z_len`, and wall thickness is never stored at all, the
constraint checker measures it, so it has to be lifted back out of the result.

The pattern: **this pipeline never raises on a missing
field.** It renders a dash, drops a series, or flattens a curve, and the page
still looks complete. That is why `tests/test_dashboard_bundle.py` pins every
field name the bundle reads, and why the headless capture is part of the
workflow rather than a nicety.

## The viewport

The renderer is hand-written WebGL, ~900 lines, no library. What it draws, and
why each piece is there rather than being decoration:

**Model edges.** A shaded solid with no edges reads as a blob. On a tessellated
mesh the model edges are the ones where two faces meet at an angle; the
triangulation's own diagonals lie flat between coplanar faces and must not be
drawn. `buildEdges` separates them with a 22-degree test, and open boundaries
are always kept. Welding is what makes this possible: two triangles only share
an edge if they share vertex indices. The fill is pushed back with a polygon
offset while the edges draw, or a line coplanar with the surface it bounds
z-fights and half of every edge drops out as the camera moves.

**Section cap.** Clipping alone deletes fragments and leaves the far wall
showing through an open mouth, so a solid part reads as a shell. The stencil
counts the faces the cut removed and fills where the plane passes through
material. The cut face takes the part's brightness pulled warm -- drawn in the
part's own tone first, and readback showed 1667 capped pixels while the
screenshot showed nothing, which is a correct cap that looks like more surface.

**Ground shadow.** The part is flattened onto the floor plane down the light
ray, one matrix and a second pass over the same buffers. Stencilled, or a part
many triangles deep blends into a silhouette of its own tessellation. Drawn
after the grid, because on a dark ground the shadow has no room against the
background and what actually reads is the grid dimming inside the silhouette.

**Measurement.** Ray-triangle against every triangle, once per click; a part
here is a few thousand of them, so an acceleration structure would be code with
no measurement behind it. The ray has to come back out of the renderer's
normalized space into millimetres, which is exact because both transforms are
translation plus uniform scale. Hits snap to the nearest corner of their
triangle when close, since measuring a machined part means measuring between
its corners.

Verified over a 19x19 sweep of the viewport: 97 hits, every one lying on a
triangle plane to 0.0000 mm, none outside the bounds, all reprojecting to
within the snap radius of the pixel they were picked from.

## Scrubbing the timeline

`scripts/build_steps.py` replays each recorded trajectory one action at a time
under FreeCAD's interpreter and exports an STL wherever the solid changed.
`bundle.py` attaches them, and clicking a timeline node loads the part as it
stood at that feature -- which is what a parametric timeline does, and what
this one could not do while it had only the action list.

The step meshes are attached for **one design per family, 8 of 24**, and that
is a size decision: they are 189 KB of a 412 KB bundle as it is. A node with no
exported solid falls back to the last one that has it, so clicking a sketch
action shows the solid it was drawn on rather than doing nothing.

The framing deliberately does not change between steps. An early step is
genuinely smaller than the finished part, and re-normalizing each one would
hide that by blowing every intermediate up to fill the viewport.

## Rollouts: what the policies actually did

The leaderboard says bc scores 0.458. It does not say what bc *did*, and on
these tasks the answer is not subtle once it is visible: on
`build-design_000000` it opens a sketch, draws five lines, and then emits PAD
for the remaining thirty-four steps while the environment rejects every one.

That needed a change to the runner. The trace recorded operation names, so it
could say the policy emitted PAD thirty-four times; it could not say the
environment refused thirty-four of them, which is the difference between a
policy building badly and a policy not building at all. `TaskResult` now
carries `accepted` and `rejections`, one entry per operation.

The workspace draws each episode as a strip, one cell per action, accepted or
rejected, with the milestones reached beneath. One BUILD task per family: BUILD
starts from nothing, so the whole sequence belongs to the policy, where a
COMPLETE task is mostly replayed expert prefix and would credit the policy with
actions it never chose.

Rejection is encoded in cell height as well as colour, so the strip still reads
without separating red from green.

### What the policy built

The strip says what a policy did. It does not say what it *made*, and for a
project about building CAD that is the thing. `scripts/build_rollout_meshes.py`
replays the actions the environment accepted and exports the solid the episode
left, and the workspace shows it beside the expert's.

That needed one more change. The environment already built the resolved action
and put it in `info`; the bridge kept only its name, which is what made a
policy's geometry unreproducible from its trace. It now passes the whole
action through -- a few hundred bytes a step over a local pipe.

Rejected actions are skipped on replay. One changed nothing when it was
refused, so replaying it would stop the rebuild at the first rejection rather
than at the end of what the policy actually built.

**20 of 42 episodes leave a solid** on the core preset. `oracle-replay` and
`scripted-spec` always do; `bc` and `ppo` on three of the seven tasks;
`immediate-finish` and `legal-random` never. The export reports the count that
did not, because a policy that never made a solid is a result rather than a
gap in the export.

The rebuild happens outside the environment that scored the episode, so a
replay that diverged would show as a wrong part rather than an error. The
trace records the mass the runner measured, so the export checks every rebuilt
solid against it and exits non-zero on a mismatch. Worst drift across the 20
solids is **0.00%**: the replay reproduces the episodes exactly. The number is
printed rather than a bare pass, since a threshold with nothing behind it hides
a check drifting toward its own limit.

Each strip also carries its first refusal. After it these policies repeat the
same action and collect the same message for the rest of the episode, so the
later ones are echoes and the first is the diagnosis.

On `build-design_000006` the expert builds a base plate, a wall, a triangular
rib and two holes. bc builds the plate and the wall and nothing else, then
jams for the remaining 31 steps on `Pad failed to build: ["Pad002:
state=['Touched', 'Invalid']"]`. ppo fails identically. Both viewers share a camera and the expert's
framing, or the comparison would be between two framings rather than two
parts.

A trace written before this change has no per-step record. The page says so
rather than drawing the cells as accepted -- an empty list rendered as a clean
run would show a policy jamming as a policy succeeding, which is the same
class of silent failure the rest of this document is about.

## What the requirement asked for, and what was checked

The requirement is prose. The spec is what the parser made of it. The
constraint report is what was checked against the built solid. Those are three
different things, and the page showed the first and the last, so a value the
requirement asked for and nothing verified was invisible.

Only `spec.kind` was ever read out of the bundle, which left the first phase of
this project with no representation on the page at all.

It surfaces a real gap. On `design_000001` the flange parses six measurable
values and **two** are verified: `bore_diameter`, `bolt_count`,
`bolt_circle_diameter` and `min_wall_thickness` are asked for and never
checked, while the part reports CONSTRAINTS MET 100%. Across the 24 embedded
designs, nine parsed fields have no constraint of that name anywhere, and
`min_wall_thickness` appears in 21 specs against 6 constraints.

`kind` and `objective` are marked directives rather than unchecked. They pick
the family and the optimization target; neither is a measurable property, so
neither is missing a constraint.

## Where the suite is won

The leaderboard is a mean over 76 tasks, and two policies with the same mean
can be solving disjoint halves of the suite. The matrix is that number
un-averaged: one column per task, shaded by milestones reached.

Milestones rather than success, because success is 0.000 for three of the six
policies and a success matrix would be one colour. Bands are named, because
the strongest pattern in the grid is otherwise a shape with nothing naming it:
`immediate-finish` is dark across all of BUILD and bright across every
COMPLETE band, which is the auditing baseline doing exactly what it exists to
do. A task a policy never attempted leaves a hole rather than a zero.

## How the actions were refused

The rollout strips show seven tasks. The taxonomy is every refused action
across all 76, reduced to the failure it describes, which is the only way to
tell a failure of this task from a failure of the policy.

The contrast is the finding. `bc` and `ppo` each fail in **4** distinct ways --
Pad and Revolution are 97% of bc's 616 refusals -- while `legal-random` fails
in **77**. A learned policy repeats one mistake; a random one makes many.

Two things had to be normalized away: FreeCAD's instance name (`Pad002`), and
constraint index tuples. Leaving the indices in split `sketch rejected
Coincident constraint` across 80 rows of count one, and the tail read as 157
unrelated failures rather than 77. What the top eight leave out is reported,
so the listed counts do not read as the whole of it.

## Addressable views

The hash carries the workspace and the selection: `#model/design_000017`,
`#rollouts/build-design_000010/ppo`. It is written on every state change and
read only on load and on an explicit back or forward -- a listener reacting to
its own writes would fight the UI. A hash naming a part this bundle does not
carry leaves the selection alone rather than clearing it, since the bundle is
capped at 24 designs and a link to the 500th names something this page never
had.

## Reading a mean apart

Four views exist because one number was hiding four different things.

**Building versus finishing.** A BUILD task is a sentence; a COMPLETE task
hands the policy an expert prefix and asks for the last k actions. bc and ppo
both score **0.069 on BUILD with 0.000 success** and around 0.57 on COMPLETE.
Nearly all of the 0.458 headline comes from tasks where the prefix had already
built most of the part.

**Which parts are hard.** bc and ppo span **0.339** between their best family
and their worst -- 0.622 on plate, 0.283 on flange -- where every baseline
spans under 0.11. They are not uniformly better than the baselines; they are
much better at some shapes and barely better at others.

**Where each policy stops.** Progress is prefix-scored, so the rung a policy
fails at zeroes everything after it. scripted-spec loses half its episodes at
`has_any_hole`; immediate-finish 36% at `opened_a_sketch`; bc and ppo theirs at
`all_constraints_met`.

**Mistakes, or a stop.** bc and ppo hit their first refusal a quarter of the
way into an episode and then spend three quarters of it being refused,
recovering in **2 of 31** and 2 of 30. legal-random is refused more often and
recovers in 42 of 53. The learned policies are not making scattered mistakes;
they reach a state they cannot act from and stay there.

The codec audit is stated beside the ceiling it explains: all 16,452 expert
actions survive the round trip through the action space, the worst by 5e-10
mm. If it did not, the oracle would sit below 1.000 and it would be the codec
rather than any policy -- which it has been, twice.

## The corpus

The browser carries a capped 24 and used to say only that. The Dataset
workspace says what the 24 are a sample of: **1,080 designs across 8
families**, 114 to 144 each, masses 2.0 to 278.1 g, expert trajectories 7 to
22 steps with a mean of 15.2.

The families are near-even by construction and not exactly so, because
generation rejects parts that fail their own constraints and some families
fail more often than others.

## The surrogate, on the page

Phase 6 had no representation here at all, and its headline is the least
interesting thing in it. The Optimize workspace shows the fit (R2 0.923 on
mass, 0.972 on thickness, 8 held-out rows), the search descending toward zero
predicted mass over 1,803 evaluations, and the number that matters:

**0.32 g predicted, 29.16 g built. The surrogate was 98.9% wrong at the
parameters it recommended.** The answer is still right -- 54.1% lighter than
the baseline, still manufacturable -- because the reported number is the built
one. A search that skipped the verification build would have reported a 0.3 g
plate.

This was already in [phase6.md](phase6.md). What was missing was that anyone
reading the station would ever see it.

## Two gates the station needed

**`check_station.py --staged`** asks the built-page gate about the commit
rather than the working tree. `make lint` reads the tree, which passes after a
rebuild whether or not the page was staged; two commits went red in CI that
way, and neither reproduced locally because locally there was nothing to
reproduce.

**`check_wiring.py`** fails when the app looks up an element id the markup does
not define. `el('x')` returning null throws part way through `init`, so
everything wired after that point is silently never wired while the page still
renders -- most of it was built before the throw. That happened while these
sections were being written, and the only signal was a heading missing from a
screenshot.

## Verifying the render

Lint cannot see any of the above. Screenshot it:

```bash
make dashboard
cd docs && python3 -m http.server 8912 &
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new \
  --disable-gpu --enable-unsafe-swiftshader --use-gl=angle --use-angle=swiftshader \
  --hide-scrollbars --window-size=1600,1000 --virtual-time-budget=9000 \
  --screenshot=/tmp/shot.png http://127.0.0.1:8912/dashboard.html
```

Software rendering (swiftshader) is enough to judge shading, layout and colour.

## Limits

- **24 designs, not 1,080.** Beyond that the file stops being openable. The cap
  is `--limit`; the selection round-robins across families, so every family
  appears before any family repeats. It is not a random sample, within a
  family it takes the lowest ids.
- **Static.** It shows the run that built it. Re-run `make dashboard` after new
  results; there is no live reload and no server.
- **WebGL required.** The metrics, tables and charts render without it; only the
  3D panel degrades, and it says so rather than showing an empty box.
- **The section cap needs a stencil buffer** and a consistently wound mesh.
  Without one it is skipped and the section still works, uncapped, as it did
  before. Nothing checks the winding; a mesh that fails it would cap with
  holes rather than error.
- **Measurement is point to point.** No edge, arc or face snapping, no angle,
  no radius. It snaps to triangle corners, which on these parts are the model
  corners, but on a curved face that is a tessellation vertex and not a
  feature.
- **16 of 24 designs have no step meshes.** Their timelines highlight a node
  and nothing else, exactly as before. The build report prints the scrubbable
  count so a page built without `build_steps.py` says so.
