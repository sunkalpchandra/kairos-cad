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

running=0
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  start_id=$(( i * 10000 ))
  echo "shard $((i + 1))/${#SEEDS[@]}: seed=$seed start_id=$start_id count=$PER_SHARD"
  PYTHONPATH="$REPO" "$PY" scripts/generate_brackets.py \
    --count "$PER_SHARD" --out "$ROOT/designs" --seed "$seed" --start-id "$start_id" &
  running=$(( running + 1 ))
  if (( running >= CONCURRENCY )); then
    wait -n
    running=$(( running - 1 ))
  fi
done
wait

python3 - "$ROOT" "$PER_SHARD" "${SEEDS[@]}" <<'PY'
import json, sys
from pathlib import Path

root, per_shard, *seeds = sys.argv[1], int(sys.argv[2]), *map(int, sys.argv[3:])
manifest = {
    "generator": "scripts/generate_brackets.py",
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
