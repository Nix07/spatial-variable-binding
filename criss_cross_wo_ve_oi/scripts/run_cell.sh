#!/bin/bash
# Submit one (model, direction) cell of the §5.3.2 experiment as a SLURM job.
# Usage: scripts/run_cell.sh <model> <direction>
#   model     ∈ {qwen2vl, gemma3, pixtral}
#   direction ∈ {left, right, above, below}

set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 <model> <direction>" >&2
  exit 2
fi

MODEL=$1
DIR=$2
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
JOB_NAME="ccua-${MODEL}-${DIR}"
LOG_DIR=${SLURM_LOGS_DIR:-/mnt/home/nikhil/slurm_logs}
mkdir -p "$LOG_DIR"

sbatch \
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
  "
