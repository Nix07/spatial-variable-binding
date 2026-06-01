#!/bin/bash
# Submit the four per-direction jobs for one model. After all four
# succeed an afterok merge job stitches the per-direction artifacts into
# the canonical `squares_object_patching_under_ablation.npy`.
#
# Usage: scripts/submit_all.sh <model>
#   model ∈ {qwen2vl, gemma3, pixtral}

set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 <model>" >&2
  exit 2
fi

MODEL=$1
HERE=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$HERE/../.." && pwd)
LOG_DIR=${SLURM_LOGS_DIR:-/mnt/home/nikhil/slurm_logs}
mkdir -p "$LOG_DIR"

JOB_IDS=()
for DIR in left right above below; do
  JOB_NAME="ccua-${MODEL}-${DIR}"
  JID=$(sbatch --parsable \
    --job-name="$JOB_NAME" \
    --gpus=1 \
    --time=02:00:00 \
    --output="$LOG_DIR/${JOB_NAME}_%j.out" \
    --error="$LOG_DIR/${JOB_NAME}_%j.err" \
    --wrap="
      set -eu
      cd '$REPO_ROOT'
      . ~/.venvs/vlm-spatial-reasoning/bin/activate
      if [ -f .env ]; then set -a; . ./.env; set +a; fi
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      echo \"host: \$(hostname); HF_TOKEN: \${HF_TOKEN:+set}\"
      nvidia-smi -L
      PYTHONUNBUFFERED=1 python -m criss_cross_wo_ve_oi.run_experiment \
          --model '$MODEL' --direction '$DIR' \
          --output-name squares_object_patching_under_ablation_${DIR}
    ")
  JOB_IDS+=("$JID")
  echo "submitted $JOB_NAME: $JID"
done

DEP=$(IFS=:; echo "${JOB_IDS[*]}")
MERGE=$(sbatch --parsable \
  --job-name="ccua-${MODEL}-merge" \
  --cpus-per-task=1 \
  --mem=4G \
  --time=00:10:00 \
  --dependency=afterok:$DEP \
  --output="$LOG_DIR/ccua-${MODEL}-merge_%j.out" \
  --error="$LOG_DIR/ccua-${MODEL}-merge_%j.err" \
  --wrap="
    set -eu
    cd '$REPO_ROOT'
    . ~/.venvs/vlm-spatial-reasoning/bin/activate
    python - <<'PYEOF'
import numpy as np
from pathlib import Path
d = Path('criss_cross_wo_ve_oi/results/$MODEL')
merged = {}
for direction in ('left', 'right', 'above', 'below'):
    p = d / f'squares_object_patching_under_ablation_{direction}.npy'
    merged.update(np.load(p, allow_pickle=True).item())
out = d / 'squares_object_patching_under_ablation.npy'
np.save(out, merged)
print(f'merged {sorted(merged)} into {out}')
for direction in ('left', 'right', 'above', 'below'):
    (d / f'squares_object_patching_under_ablation_{direction}.npy').unlink()
PYEOF
  ")
echo "merge job: $MERGE (depends on $DEP)"
