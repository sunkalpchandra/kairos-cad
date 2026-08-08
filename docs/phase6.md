# Phase 6: Engineering optimization and a learned surrogate

Phase 6 does two things: it makes the last unmeasured constraint measurable, and
it uses that measurement to optimize designs under a manufacturing floor.

## Minimum wall thickness is now measured

`min_wall_thickness` was the project's standing example of an **unmeasured**
constraint. 940 requirement specs declared one, 283 designs stated it in their
text, and the checker could only answer "unmeasured", earning no reward credit
in either direction. It is now measured.

**Method: inward ray casting.** Sample points across each planar face, shoot a
ray inward along the face normal, intersect it with the solid, and take the
length of the intersection as the material thickness there. The minimum over all
samples is the wall thickness. Tube walls have no planar face to probe, so
coaxial cylinder radii cover them, for a spacer, that gap *is* the wall.

The textbook answer is a medial-axis transform or a distance field. FreeCAD
exposes neither, and both cost far more than an entire episode. Ray casting is
exact along each ray and its only error is one of *sampling*: it can miss a thin
spot between samples, so it can **only ever over-estimate**. That is the safe
direction for a manufacturing check to err in, provided you say so, the ray
count travels with every measurement and `sampling_error_bound()` states the
limitation in words.

| family | measured | declared |
| --- | --- | --- |
| corner_bracket | 5.00 mm | 5.0 |
| flange | 6.00 mm | 6.0 |
| l_bracket | 5.00 mm | 5.0 |
| plate | 6.00 mm | 6.0 |
| reinforced_plate | 6.00 mm | 6.0 |
| spacer | 6.00 mm | (radial wall) |
| support_bracket | 8.00 mm | 8.0 |
| u_bracket | 6.00 mm | 6.0 |

All eight families measure exactly what they declare, and the measurement is
accurate to 0.02 mm against boxes and tubes whose thickness is known
analytically.

The measurement is **off by default** (`observe(wall_thickness=True)`) because
it ray-casts against the solid. When it is absent the constraint reads
`unmeasured`, never `satisfied`, a missing measurement must never become a pass.

### Two FreeCAD traps

Both cost real debugging and are recorded in the code:

- **`Vector.multiply()` mutates in place and returns self.** The rays were
  therefore scaled three times and reached 1/1000th of their intended length,
  so a 5 mm wall measured 0.054 mm. Use `vector * scalar`, which returns a new
  vector.
- **`Face.normalAt()` already accounts for face orientation.** Flipping again
  for `Reversed` faces aimed those rays *out* of the solid. Boxes built by
  `Part.makeBox` have all-`Forward` faces and measured fine, so this hid until
  an L-bracket, whose faces are `Reversed`, measured nothing at all.

## Surrogate-driven optimization

Optimizing means evaluating thousands of candidates. A FreeCAD build costs
~0.3 s and a thickness measurement ~0.5 s more, so a ten-thousand-candidate
search is hours. The loop is therefore **propose cheaply, verify exactly**:

1. Sample the family in FreeCAD (tens of builds) to fit a surrogate.
2. Search parameter space with a cross-entropy method, scoring candidates with
   the surrogate in microseconds and rejecting infeasible draws using the
   family's own `is_feasible` before the surrogate ever sees them.
3. **Build the winner for real** and report the verified numbers.

The surrogate is a closed-form ridge fit on polynomial features, not a neural
network: with a few hundred rows, each one a real build, a closed-form fit has
no optimizer, no seed, and no training curve to misread, and it needs no torch,
so the search runs in the same interpreter as the verification build.

### Result

A plate run against a 5.0 mm manufacturing floor:

| | mass | wall thickness | manufacturable |
| --- | --- | --- | --- |
| default design | 63.53 g |, |, |
| optimized | **29.16 g** | 6.06 mm | yes |

A **54% mass saving**, verified by building it, with the wall clearing the floor.

### Three bugs the tests could not have found alone

The synthetic ground-truth test caught two:

- **Mass is a three-way product of dimensions** (width x height x thickness). A
  degree-2 surrogate scores R² 0.997 overall while getting the *thickness*
  direction wrong, so the optimizer walked the wrong way on exactly the
  parameter the manufacturing constraint governs. Degree 3 fixes it.
- **An additive feasibility penalty** large enough to forbid a real violation
  also makes a 0.01 mm prediction error at the boundary cost more than every
  mass difference in the search, so the optimum, which sits *on* the boundary,
  is precisely where the search refused to go. The penalty is multiplicative.

The real FreeCAD run caught the third, and it is the one that matters most:

- **The surrogate extrapolated to a negative mass (-85 g)**, and because the
  penalty is multiplicative, scaling a negative objective by a violation made it
  *better*. The search drove straight into that region and returned a part whose
  3.10 mm wall violated the 5.0 mm floor. Non-physical predictions are now
  refused outright, because a surrogate outside the region it was fitted on has
  nothing to say.

Note what this cost: on the same run the surrogate's mass error **at the
optimum was 98.9%** (0.32 g predicted against 29.16 g built). The result is
still correct, because the winner is always built and measured rather than
reported from prediction. That is the entire argument for verifying.

## Still out of scope

- The surrogate is fitted per run rather than shipped; a persisted per-family
  surrogate would remove the sampling cost from every optimization.
- Trust-region bounds. The search is boxed to the observed parameter range, but
  a box is not the convex hull of the samples, and the corners are exactly where
  extrapolation bites.
- Structural performance. Mass and wall thickness are geometric; stiffness or
  stress would need an FEA backend, which KAIROS does not have.
- Regenerating the dataset now that thickness is measurable would convert 283
  `unmeasured` constraints into real checks.
