#!/usr/bin/env bash
# Canonical KAIROS Phase-2 dataset run: 1,080 validated designs in 8 sharded
# passes, four at a time, then an integrity audit.
#
# Shard seeds and id offsets are fixed here so the dataset is reproducible;
# they are also recorded in <root>/manifest.json next to the designs.
#
#   scripts/generate_dataset.sh [root] [per_shard]
#
# Requires the FreeCAD bundled interpreter (see CLAUDE.md).
set -euo pipefail

ROOT="${1:-dataset}"
PER_SHARD="${2:-135}"
CONCURRENCY=4
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${KAIROS_PYTHON:-/Applications/FreeCAD.app/Contents/Resources/bin/python}"
SEEDS=(1001 1002 1003 1004 1005 1006 1007 1008)

cd "$REPO"
mkdir -p "$ROOT/designs"

# Waves of CONCURRENCY shards: macOS ships bash 3.2, which has no `wait -n`.
# Shards whose designs are already on disk are skipped, so an interrupted run
# (a full disk, a reboot) resumes instead of regenerating everything.
pids=""
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  start_id=$(( i * 10000 ))
  have=$(find "$ROOT/designs" -maxdepth 1 -type d -name 'design_*' \
    -exec basename {} \; 2>/dev/null \
    | awk -F_ -v lo="$start_id" -v hi="$(( start_id + 10000 ))" \
        '$2 + 0 >= lo && $2 + 0 < hi' | wc -l | tr -d ' ')
  if (( have >= PER_SHARD )); then
    echo "shard $((i + 1))/${#SEEDS[@]}: already has $have designs, skipping"
    continue
  fi
  echo "shard $((i + 1))/${#SEEDS[@]}: seed=$seed start_id=$start_id count=$PER_SHARD"
  PYTHONPATH="$REPO" "$PY" scripts/generate_brackets.py \
    --count "$PER_SHARD" --out "$ROOT/designs" --seed "$seed" --start-id "$start_id" &
  pids="$pids $!"
  if (( $(echo $pids | wc -w) >= CONCURRENCY )); then
    for pid in $pids; do wait "$pid"; done
    pids=""
  fi
done
for pid in $pids; do wait "$pid"; done

COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

COMMIT="$COMMIT" python3 - "$ROOT" "$PER_SHARD" "${SEEDS[@]}" <<'PY'
import json, os, sys
from pathlib import Path

root, per_shard, *seeds = sys.argv[1], int(sys.argv[2]), *map(int, sys.argv[3:])
manifest = {
    "generator": "scripts/generate_brackets.py",
    # Seeds alone do not pin the dataset: the families' feasibility rules and
    # the reward/constraint semantics recorded in each trajectory live in the
    # code, so reproducing a run means checking out this commit too.
    "commit": os.environ.get("COMMIT", "unknown"),
    "per_shard": per_shard,
    "shards": [
        {"seed": s, "start_id": i * 10000, "count": per_shard} for i, s in enumerate(seeds)
    ],
    "designs": len(list((Path(root) / "designs").glob("design_*"))),
}
(Path(root) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"wrote {root}/manifest.json ({manifest['designs']} designs)")
PY

python3 scripts/audit_dataset.py --root "$ROOT"
