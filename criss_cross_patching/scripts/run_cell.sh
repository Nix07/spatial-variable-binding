#!/bin/bash
# Submit one (model, kind, direction) criss-cross patching cell as a SLURM job.
# Usage: scripts/run_cell.sh <model> <kind> <direction>
#   model     ∈ {qwen2vl, gemma3, pixtral}
#   kind      ∈ {object, strip}      # object = Fig 6, strip = Fig 7
#   direction ∈ {left, right, above, below}

set -eu

if [ $# -ne 3 ]; then
  echo "usage: $0 <model> <kind> <direction>" >&2
  exit 2
fi

MODEL=$1
KIND=$2
DIRECTION=$3
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
JOB_NAME="cross-${MODEL}-${KIND}-${DIRECTION}"
LOG_DIR=${SLURM_LOGS_DIR:-/mnt/home/nikhil/slurm_logs}
mkdir -p "$LOG_DIR"

sbatch \
  --job-name="$JOB_NAME" \
  --gpus=1 \
  --time=06:00:00 \
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
    echo '==== criss-cross --model $MODEL --kind $KIND --direction $DIRECTION ===='
    python -m criss_cross_patching.run_experiment \
        --model '$MODEL' --kind '$KIND' --direction '$DIRECTION'
  "
