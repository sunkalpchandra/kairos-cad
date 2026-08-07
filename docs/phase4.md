# Phase 4 — Multimodal VLA and behavioral cloning

Phase 4 turns the recorded expert trajectories into a policy: a
vision-language-action model that reads a natural-language requirement and the
current geometry, and emits the next **structured CAD action** — never code.

```text
requirement tokens ──→ LanguageEncoder ──┐
rendered views ──────→ VisionEncoder ────┼──→ FusionEncoder ──→ PolicyHeads ──→ Action
numeric state + build history → StateEncoder ┘
```

The learning stack is an **optional** install (`make setup-learn`). Nothing
under `kairos/models` or `kairos/training` may be imported by the CAD, dataset,
or RL-environment layers, because those must keep running under FreeCAD's
bundled interpreter, which has no torch and cannot get one.

## Encoders

| module | input | notes |
| --- | --- | --- |
| `language_encoder.py` | token ids + magnitudes | pre-norm transformer over a frozen vocabulary |
| `vision_encoder.py` | `[B, 4, 3, H, W]` renders | one shared trunk, learned view identities, attention pooling |
| `state_encoder.py` | 24-dim numerics + history | MLP + GRU read at the last real feature |
| `fusion.py` | three modality embeddings | self-attention over a 3-token sequence |
| `policy.py` | fused embedding | operation / parameter / target heads |

Three choices are worth stating because the obvious alternative is wrong:

- **Numbers are embedded, not tokenized.** `6 mm` and `60 mm` describe
  different parts. Each numeric literal becomes one `<num>` token plus its
  scaled value, which a learned projection adds to the token embedding —
  otherwise every distinct dimension would fragment the vocabulary.
- **The vision trunk is shared across views.** An orthographic silhouette
  means the same thing whichever axis produced it; a per-view trunk would
  quadruple parameters to relearn edges four times. *Which* view saw a feature
  still matters, so each view adds a learned identity vector before pooling.
- **Parameter and target heads are conditioned on the operation.** `PAD`'s
  length and `FILLET`'s radius occupy the same slot but mean different things;
  a flat head would learn one distribution averaged over every operation.

Illegal operations receive `-1e9` logits from `kairos.actions.masking`, so they
cannot be sampled and contribute no gradient — the policy is never trained to
pick an action the document forbids. The value is finite rather than `-inf` so
a fully masked row softmaxes to uniform instead of NaN.

## What the trajectories can and cannot supervise

The dataset builder is where the honest limits live.

- **States are shifted by one.** `TrajectoryRecorder` observes in the
  post-action callback, so `states[i]` already contains the effect of
  `actions[i]`. Step `i` is supervised from `states[i-1]` — pairing them as
  recorded would let the model read "a pad exists" and predict `PAD`. Step 0
  reads the encoding of an empty document.
- **8% of steps are dropped.** The codec expresses `ADD_POLYGON` only as a
  regular n-gon, while six families sketch irregular profiles. Those steps are
  dropped and counted rather than fitted to the nearest hexagon, which would
  train the policy toward a shape the expert never drew. Coverage on the
  1,080-design dataset is **92.2%** (12,454 steps seen, 11,480 kept, 974
  dropped).
- **Targets are not supervised.** `encode` cannot recover a target *index*
  without the live edge/face list, which trajectories do not record.
  Supervising the recorded `0` would teach "always pick the first edge", so the
  target head is left to RL in Phase 5.
- **Vision is not yet exercised in training.** Only final-design renders exist,
  not per-step ones, so BC currently trains on language + state. The visual
  path is implemented and tested, and a learned placeholder stands in when
  views are absent, so one checkpoint stays valid either way.

Legality masks are rebuilt from the frozen numeric vector rather than a live
engine. On the full dataset the expert's own action is legal under that
reconstruction in **11,480 of 11,480 steps**, which is what makes the
reconstruction trustworthy.

## Training

```bash
make setup-learn
make train-bc                                   # or: python3 scripts/train_bc.py
python3 scripts/evaluate_bc.py --checkpoint runs/bc/checkpoint.pt
```

Loss is cross-entropy on the operation plus MSE on its parameters in the
codec's normalized [0, 1] space. The parameter term is **masked to the slots
the operation actually decodes**, probed from the codec rather than read off
the action schema — the two disagree (`ADD_POLYGON` takes one schema parameter
but consumes five slots). Averaging over all six would teach every operation to
emit 0.5 into slots it ignores.

Splits hold out **whole designs, never individual steps**. Steps within one
design are far too correlated: a step-level split would put a design's step 3
in training and step 4 in validation and report memorization as generalization.

## Results

1.14M parameters, 40 epochs on the 1,080-design dataset (Apple M-series GPU
via Metal, ~10 s/epoch), 9,723 training steps and 1,757 held-out steps from
162 unseen designs. Both runs are in `runs/`:

| metric | unweighted | inverse-sqrt weighted |
| --- | --- | --- |
| operation accuracy (held-out) | 0.955 | **0.960** |
| top-3 operation accuracy | 1.000 | 1.000 |
| majority-class baseline | 0.277 | 0.277 |
| parameter MAE (used slots) | 0.026 | **0.023** |

Reporting the majority baseline alongside accuracy is deliberate: a quarter of
all steps are `ADD_CIRCLE`, so "96%" needs the 27.7% floor next to it to mean
anything. Per-family accuracy runs from 0.868 (`u_bracket`) to 1.000
(`corner_bracket`, `flange`, `plate`, `spacer`), so no single easy family is
carrying the number.

### Class weighting fixed a real failure

The per-operation breakdown is what made the problem visible. The unweighted
run **never once predicted `FILLET`** — recall 0.000 across its 14 held-out
steps — because fillets are under 1% of the expert action mix. Aggregate
accuracy hid it completely: 95.5% looks healthy while an entire operation is
dead.

Inverse-sqrt class weighting (`class_weighting: inverse_sqrt`) recovered it,
and the two operations that were being confused with their neighbours as well:

| operation | support | unweighted recall | weighted recall |
| --- | --- | --- | --- |
| `FILLET` | 14 | 0.000 | **1.000** |
| `POCKET` | 231 | 0.809 | **0.935** |
| `ADD_CIRCLE` | 486 | 0.990 | 0.940 |
| `FINISH_DESIGN` | 162 | 1.000 | 0.932 |

The last two rows are the cost: weighting trades a little recall on the common
operations for the rare ones. It is a good trade here — overall accuracy and
parameter error both improved — but `FILLET` precision is only 0.560, so the
policy now over-predicts fillets. Weighting rebalanced the skew rather than
eliminating it.

### Replay against recorded designs

`scripts/replay_policy.py` walks a recorded design's states and lines the
policy's choices up against the expert's. On a plate it reproduces the whole
12-step build exactly; across 12 randomly sampled designs it agrees on 121 of
139 steps (0.871).

This is **teacher forced** — each step is scored from the expert's recorded
state, not from what the policy's own previous action would have produced — so
it measures per-step agreement, not the compounding error a closed-loop rollout
would expose. The gap between 0.96 next-action accuracy and whatever
closed-loop success turns out to be is exactly what Phase 5 has to close.

## Still out of scope here

- **Closed-loop rollout in `KairosCADEnv`.** It needs torch and FreeCAD in one
  interpreter; the environment runs under FreeCAD's python, which has no torch.
  Phase 5 resolves this (a policy server, or numpy-exported weights).
- PPO training on top of the BC initialization (Phase 5).
- Per-step rendering to activate the vision path, and re-emitting irregular
  profiles as `ADD_LINE` sequences to recover the dropped 8%.
