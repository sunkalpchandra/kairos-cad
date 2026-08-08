# Phase 9 — Paper

```bash
make paper          # -> paper/main.pdf
```

Source is [paper/main.tex](../paper/main.tex). Six pages plus references, in
NeurIPS layout.

## No number in it is typed by hand

`scripts/build_paper_tables.py` reads the committed run artifacts and writes
`paper/generated/`:

| file | contents |
| --- | --- |
| `leaderboard.tex` | main results table |
| `success_curve.tex` | success against COMPLETE(k) difficulty |
| `comparisons.tex` | paired bootstrap intervals |
| `ablations.tex` | corrupted-input conditions |
| `facts.tex` | `\newcommand` macros for every inline figure |

Prose says `\OracleCeiling`, not `0.698`. That is the whole point: this project
has already had to retract a published "100% of designs satisfy their
constraints" that covered 266 designs where nothing was measured, and a "0.286
closed-loop success" measured on a split the policy had trained on. A table
generated from artifacts cannot drift from them without the build failing;
a table typed into LaTeX drifts the moment anything is re-run.

If `make paper` produces a table you did not expect, the artifacts changed. That
is the feature.

## The argument

The paper is organised around the finding this project kept rediscovering: **on
a task with this many layers, the headline score usually measures the harness
rather than the policy.**

Section 6 is the load-bearing one. It lists four defects, each of which produced
a plausible, publishable, wrong number:

1. **An action space that cannot express its own demonstrations.** Six of eight
   families sketched their outline as one polygon with an arbitrary vertex list;
   the codec can only express a *regular* n-gon. 7.81% of expert steps were
   silently dropped, across 829 of 1,080 designs.
2. **Normalisation that clips instead of failing.** A sketch offset of 89.9 mm
   against a ±50 mm range decoded back as 50 mm — the oracle built the feature
   40 mm from where the expert put it, reporting success at every step. 260
   steps across 19.1% of designs.
3. **A requirement the ground truth cannot satisfy.** A 6.1866 mm wall stated as
   "6.2 mm" makes the expert violate its own requirement while having built
   exactly the right part. 139 designs (12.9%).
4. **Model selection on the evaluation set.** An early PPO run picked its best
   checkpoint on the pool the evaluation scored.

None of these raised an exception. Each produced a number in the expected
direction of the expected magnitude. What caught them was not testing that the
code runs, but replaying ground truth through the agent's own machinery and
refusing to accept a score below perfect without an explanation.

That is why `oracle-replay` is in the baseline set at all, and the paper argues
it belongs in any agent benchmark whose action space is a lossy encoding of its
demonstrations.

## The style file is not the official one

[paper/neurips_2026.sty](../paper/neurips_2026.sty) is a reimplementation
written for this repository, and its header says so. It reproduces the
measurements a reader recognises — 5.5in column, 9in height, Times 10pt, ruled
abstract, numbered lines in preprint mode — so the draft reviews at the right
length.

Before submitting anywhere, drop in that venue's official file and rebuild. The
body needs no changes: it uses only `\author`/`\And`/`\AND` and the
`preprint`/`final` options the real style provides.

## Building

`tectonic` is the only dependency; it fetches packages on demand.

```bash
brew install tectonic
make paper
```

`make paper` regenerates the tables first, so the PDF always matches the current
artifacts. To rebuild only the tables:

```bash
python3 scripts/build_paper_tables.py --out paper/generated
```

## What the paper does not claim

The learned results are weak in absolute terms and the paper says so in
Section 7 rather than burying it. Full builds succeed rarely. The benchmark
separates BC and PPO from the null baselines but **not from each other** — the
paired interval straddles zero, and the paper reports that as the result rather
than quoting the point estimates. Target selection is not recoverable by the
current encoder, which caps the families whose recipes pattern a specific
feature. The RL stage is single-seed. Requirement language is templated, because
the families are procedural.
