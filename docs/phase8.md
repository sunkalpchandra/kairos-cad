# Phase 8: Interactive dashboard

One HTML file. Open it from disk, no server, no network, no build step:

```bash
make dashboard          # -> docs/dashboard.html
open docs/dashboard.html
```

It reads only committed artifacts, `dataset/designs/*/`, `runs/*/report.json`,
`runs/*/‌*_traces.jsonl`, `runs/*/leaderboard.json`. Nothing is computed from a
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

## Three deliberate deviations from the plan

**No Three.js.** The plan named it; this ships a ~250-line WebGL renderer
instead. Three.js minified is ~600 KB, six times the size of every part,
metric and curve in the bundle put together, and the one constraint that
actually decides this dashboard's shape is that it must be a single file that
opens offline. What the viewer needs is to orbit a static mesh and shade it.

**No charting library**, for the same arithmetic. The charts are SVG strings,
which also means they lift straight into the paper figures.

**Meshes come from the STLs**, not from re-meshing FCStd files. Every generated
design already ships `model.stl`, so the whole dashboard builds under the torch
interpreter with no FreeCAD subprocess.

## Making 24 parts fit in one file

| step | effect |
| --- | --- |
| binary STL → welded vertices | 744 raw corners → 122 on a typical bracket |
| float64 → integers quantized to 0.01 mm | ~20 chars per coordinate → ~4 |
| result | the JSON mesh is *smaller than the STL it came from* |

The 0.01 mm quantum sits an order of magnitude below the 0.1 mm tolerance every
constraint check in this repo uses, so it cannot change what the viewer shows
about a part that passed or failed.

## Four bugs the screenshots caught that lint could not

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

There is a pattern here worth naming: **this pipeline never raises on a missing
field.** It renders a dash, drops a series, or flattens a curve, and the page
still looks complete. That is why `tests/test_dashboard_bundle.py` pins every
field name the bundle reads, and why the headless capture is part of the
workflow rather than a nicety.

## Verifying it renders

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
