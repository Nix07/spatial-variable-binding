#!/bin/bash
# Submit one (model, setting, orientation) probe-training cell as a SLURM job.
# Usage: scripts/run_cell.sh <model> <setting> <orientation>
#   model       ∈ {qwen2vl, gemma3, pixtral}
#   setting     ∈ {squares, shapes, objects, whatsup, grid2x2, grid3x3}
#   orientation ∈ {horizontal, vertical}

set -euo pipefail

if [ $# -ne 3 ]; then
  echo "usage: $0 <model> <setting> <orientation>" >&2
  exit 2
fi

MODEL=$1
SETTING=$2
ORIENT=$3
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
JOB_NAME="probe-${MODEL}-${SETTING}-${ORIENT}"
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
    echo '==== probing --model $MODEL --setting $SETTING --orientation $ORIENT ===='
    python -m probing.run_experiment \
        --model '$MODEL' --setting '$SETTING' --orientation '$ORIENT'
  "
