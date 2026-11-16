#!/usr/bin/env bash
# Assemble the large index, then validate every preserved asset against its source.
set -euo pipefail
export WORK_ROOT=${WORK_ROOT:-/workspace}
PREP=$WORK_ROOT/preparation
VRA=$WORK_ROOT/code/visual_rag_agent_new
PY=$WORK_ROOT/envs/rl/bin/python
rm -f "$PREP/migration_verify.exit"
trap 'rc=$?; printf "%s\n" "$rc" > "$PREP/migration_verify.exit"' EXIT
if [ ! -f "$PREP/train_index_assembly.json" ]; then
  "$PY" "$VRA/scripts/assemble_runpod_index.py" > "$PREP/index_assembly.log" 2>&1
fi
"$PY" "$VRA/scripts/runpod_asset_manifest.py" \
  --root "$WORK_ROOT" --verify "$PREP/source_assets.json" \
  --output "$PREP/assets_verified.json" > "$PREP/asset_verification.log" 2>&1
printf 'MIGRATION_ASSETS_VERIFIED\n'
