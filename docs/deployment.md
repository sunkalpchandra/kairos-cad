# Deployment

KAIROS is two things with different hosting requirements, and conflating them
is why "just deploy it" has no single answer.

| tier | what it is | needs | where it can run |
| --- | --- | --- | --- |
| Review station | one HTML file, data inlined | a browser | any static host, or a file:// open |
| Live sandbox | FreeCAD + torch + the JSON bridge | ~4 GB image, 2 CPU | a container host |

The review station is what most people want: it answers what the agent built,
how it scored, and where it fails. The sandbox is for running the agent.

## Review station (static)

```bash
make dashboard-studio          # -> docs/index.html
open docs/index.html
```

One file, ~280 KB, no server and no network. It carries 24 tessellated parts,
the leaderboard, the success(k) curve, paired intervals, training histories and
the ablations. Everything is read from `dataset/` and `runs/` at build time, so
a number on screen can always be traced to a file on disk.

Host it anywhere that serves a file:

- **GitHub Pages**: served from `docs/` on `main`. Free for public repositories
  only; a private repo needs GitHub Pro or above. `docs/.nojekyll` is required,
  or Jekyll strips the underscore-prefixed assets and reprocesses the markdown.
- **Object storage**: upload the single file to S3, R2, or GCS with public read.
- **Netlify / Vercel / Cloudflare Pages**: point at `docs/`, no build step.
- **`python3 -m http.server --directory docs`** for a local share.

There is no build pipeline to configure because there is nothing to build: the
file is already assembled.

## Live sandbox (container)

The awkward part is that the two halves cannot share an interpreter. FreeCAD
ships its own Python and cannot import torch; the learning stack needs torch
and cannot import FreeCAD. They communicate over the newline-delimited JSON
bridge in `kairos/rl/`, with the CAD side running as `env_server` under
FreeCAD's interpreter.

`deploy/Dockerfile` puts both in one image on `freecad-python3`, which provides
the headless `FreeCADCmd` binary. No X server: nothing in the agent path opens a
window, and the rendered views are opt-in and produced by the repo's own
software rasterizer.

```bash
docker build -f deploy/Dockerfile -t kairos:latest .
docker compose -f deploy/docker-compose.yml run --rm benchmark
```

The build ends with an import check on both interpreters, so an image that
would start and then fail to import FreeCAD fails at build time instead.

**Not yet built.** The Dockerfile is written against the two-interpreter
constraint verified on the development machine and Debian's `freecad-python3`
packaging, but no image has been produced from it: the Docker daemon was not
running when it was authored. Treat the first build as the test. The most
likely thing to need adjusting is the `freecadcmd` binary name, which differs
between Debian's `freecad-python3` and upstream AppImage builds.

### Sizing

| resource | requirement | why |
| --- | --- | --- |
| image | ~4 GB | FreeCAD is ~1.5 GB, CPU torch ~2 GB |
| CPU | 2 cores minimum, 8 for generation | generation shards 8 ways |
| memory | 4 GB | one FreeCAD document at a time |
| disk | 1 GB plus dataset | 1,080 designs are ~138 MB |
| GPU | optional | BC trains in ~13 min on CPU |

The dataset is a volume mount, not a layer. Baking 138 MB of geometry into an
image that is rebuilt on every source change is wasted every time.

### Where this fits

Anything that runs a container: Fly.io, Railway, Render, ECS, Cloud Run (with
the CPU and memory raised), or a plain VM. It does **not** fit anywhere
serverless-by-function: FreeCAD's import alone takes seconds, and the bridge is
a long-lived process holding an open document.

## What is not deployable, and why

**A hosted "design something" demo.** The agent's full-build success rate is
low; see [phase7.md](phase7.md). A public demo would mostly show failed builds.
The COMPLETE(k) tasks are where it demonstrably works, and those need an expert
prefix, which means they are a replay rather than a live request.

**Browser-side CAD.** The geometry kernel is OCCT via FreeCAD. There is a WASM
build of OCCT, but the action executor, the constraint checkers and the
measurement code all target FreeCAD's Python API, so this would be a rewrite
rather than a port.

## Reproducing published numbers

```bash
docker compose -f deploy/docker-compose.yml run --rm generate
docker compose -f deploy/docker-compose.yml run --rm audit
docker compose -f deploy/docker-compose.yml run --rm benchmark
```

`audit` is a gate, not a report: it exits non-zero if any expert step fails to
round-trip through the action codec.

The benchmark refuses to run if the dataset has drifted from the frozen suite.
That check exists because the dataset was regenerated three times during
Phase 7 while `suite_version` stayed identical, so the "frozen" benchmark
silently followed the data. Digests now pin the test trajectories; see
`benchmark/kairos-cad-v1/trajectories.sha256`.
