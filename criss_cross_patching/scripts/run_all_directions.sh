#!/bin/bash
# One job per (model, kind) — runs all 4 directions in a single Python
# process so the VLM only loads once.
# Usage: scripts/run_all_directions.sh <model> <kind>
#   model ∈ {qwen2vl, gemma3, pixtral}
#   kind  ∈ {object, strip}

set -eu

if [ $# -ne 2 ]; then
  echo "usage: $0 <model> <kind>" >&2
  exit 2
fi

MODEL=$1
KIND=$2
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
JOB_NAME="cross-${MODEL}-${KIND}"
LOG_DIR=${SLURM_LOGS_DIR:-/mnt/home/nikhil/slurm_logs}
mkdir -p "$LOG_DIR"

sbatch \
  --job-name="$JOB_NAME" \
  --gpus=1 \
  --time=08:00:00 \
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
    echo '==== criss-cross --model $MODEL --kind $KIND --all-directions ===='
    python -m criss_cross_patching.run_experiment \
        --model '$MODEL' --kind '$KIND' --all-directions
  "
