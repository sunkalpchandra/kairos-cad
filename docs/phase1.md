# Phase 1: CAD backend and structured action API

Phase 1 delivers the foundation everything else builds on: a controlled,
transactional, fully headless CAD execution layer, plus the structured action
interface the RL agent will use.

## What exists

```text
kairos/
├── cad/
│   ├── backend.py       FreeCAD discovery/bootstrap (no pip package exists)
│   ├── errors.py        typed failure taxonomy (validation vs feature vs export)
│   ├── document.py      one document + one PartDesign body, origin refs, tree
│   ├── sketches.py      lines/rects/circles/arcs/polygons on origin planes
│   ├── constraints.py   typed Sketcher constraints + DoF readout
│   ├── features.py      pad/pocket/revolve/fillet/chamfer/shell/patterns
│   ├── boolean.py       shape-level union/cut/intersection
│   ├── measurements.py  volume/mass/bbox/topology/edge+face inventory/holes
│   ├── validation.py    structured reports (kernel validity, shells, volume)
│   ├── rendering.py     software rasterizer → PNG (iso/front/top/right)
│   ├── export.py        STEP / STL / FCStd
│   └── engine.py        CADEngine facade; all targets are name strings
├── actions/
│   ├── schema.py        Operation vocabulary + Action/ActionResult dataclasses
│   ├── parameters.py    per-operation typed parameter specs (the gate)
│   ├── executor.py      validated no-raise dispatch + trajectory recording
│   └── masking.py       state flags → legal operation set (pure logic)
└── data/
    ├── procedural.py    L-bracket / plate recipes as expert action sequences
    └── generator.py     validated dataset writer (renders, STEP/STL, JSON)
```

## Design decisions that matter later

**Transactional features.** Every feature builder validates the recompute; on
failure the feature is removed and the previous body tip restored before a
typed `FeatureError` propagates. The executor converts this into
`ActionResult(ok=False)`. Consequence: an RL agent can attempt aggressive
actions without corrupting the episode's document.

**No silent no-ops.** FreeCAD quirks discovered and handled during Phase 1:

- `body.newObject` does not advance the tip for transform features
  (Mirrored, patterns), the engine now advances it explicitly, otherwise
  measurements keep reading the pre-pattern solid.
- Pattern features that leave the volume unchanged are rejected as failures
  rather than reported as successes.
- Compound tip shapes have no `CenterOfMass`; it is aggregated
  volume-weighted over solids.

**Hole detection is semantic.** "4 x M5 holes" is checked by grouping
*concave* cylindrical faces on a shared axis and requiring (nearly) full
360° angular wrap. This excludes convex rounds *and* concave corner-fillet
coves, both of which are cylinders too.

**Serializable everything.** Engine targets are name strings
(`Sketch001`, `Edge7`); actions and results round-trip through JSON; recorded
trajectories replay through the same executor the agent will use.

## Verification

- 285 tests under FreeCAD's interpreter, and 364 under the system one
  (there the CAD tests skip and the Phase 4-6 learning tests run), `make test-all`.
- Volume assertions are analytic (e.g. pad = w-h-l, fillet removes
  `(r² - πr²/4)-L`), not snapshots.
- `make generate-data` writes only designs that pass kernel validation and
  requirement checks; rejects are counted with reasons in
  `generation_stats.json`. First 30-design run: 30 written / 33 attempted
  (2 infeasible parameter draws, 1 hole-count validation reject).

## Known limitations (deliberate for Phase 1)

- Minimum-wall-thickness measurement is not yet implemented (Phase 6 per the
  project plan); requirements record the parametric wall thickness instead.
- Boolean ops operate on shapes, not yet wired into multi-body documents or
  the executor (single-body PartDesign covers Phases 2-5 tasks).
- Edge/face names (`Edge7`) are FreeCAD topological names and can shift
  across feature insertions, recipes locate edges geometrically
  (`find_edges(direction=..., near=...)`) immediately before use, which is also
  the pattern the agent will learn.

## Next (Phase 2)

Scale the generator to 1,000+ validated designs across more families
(U-bracket, corner bracket, flanges), then build the geometry-graph
representation on top of `document.feature_tree()` + face/edge inventories.
