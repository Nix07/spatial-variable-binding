#!/bin/bash
# Submit COCO amplification runs.

set -eu

DIR=$(cd "$(dirname "$0")" && pwd)
MODEL=${1:-qwen}
"$DIR/run_cell.sh" "$MODEL" fp32
"$DIR/run_cell.sh" "$MODEL" bf16
