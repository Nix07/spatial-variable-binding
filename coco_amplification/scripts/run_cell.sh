#!/bin/bash
# Submit one COCO amplification run as a SLURM job.
# Usage: scripts/run_cell.sh <model> <dtype>
#   model in {qwen,gemma,pixtral}
#   dtype in {fp32,bf16}

set -eu

if [ $# -ne 2 ]; then
  echo "usage: $0 <model> <dtype>" >&2
  exit 2
fi

MODEL=$1
DTYPE=$2
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
JOB_NAME="coco-amp-${MODEL}-${DTYPE}"
LOG_DIR=${SLURM_LOGS_DIR:-/mnt/home/nikhil/slurm_logs}
mkdir -p "$LOG_DIR"

sbatch \
  --job-name="$JOB_NAME" \
  --gpus=1 \
  --time=12:00:00 \
  --output="$LOG_DIR/${JOB_NAME}_%j.out" \
  --error="$LOG_DIR/${JOB_NAME}_%j.err" \
  --wrap="
    set -eu
    cd '$REPO_ROOT'
    . ~/.venvs/vlm-spatial-reasoning/bin/activate
    if [ -f .env ]; then set -a; . ./.env; set +a; fi
    echo \"host: \$(hostname); HF_TOKEN: \${HF_TOKEN:+set}\"
    nvidia-smi -L
    python -m coco_amplification.run_experiment \
      --model '$MODEL' --torch-dtype '$DTYPE'
  "
