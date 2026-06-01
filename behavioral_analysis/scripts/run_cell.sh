#!/bin/bash
# Submit one Table-1 cell as a SLURM job.
# Usage: scripts/run_cell.sh <model> <task>
#   model ∈ {qwen2vl, gemma3, pixtral}
#   task  ∈ {squares, shapes, objects, whatsup}

set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: $0 <model> <task>" >&2
  exit 2
fi

MODEL=$1
TASK=$2
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
JOB_NAME="table1-${MODEL}-${TASK}"
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
    echo \"host: \$(hostname); HF_TOKEN: \${HF_TOKEN:+set}\"
    nvidia-smi -L
    PYTHONUNBUFFERED=1 python -m behavioral_analysis.src.run --model '$MODEL' --task '$TASK'
  "
