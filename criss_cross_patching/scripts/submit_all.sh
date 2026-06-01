#!/bin/bash
# Submit the full Fig 6 + Fig 7 sweep: 3 models × 2 patching kinds = 6 jobs,
# each covering all 4 directions in one process.

set -eu
HERE=$(cd "$(dirname "$0")" && pwd)

for model in qwen2vl gemma3 pixtral; do
  for kind in object strip; do
    "$HERE/run_all_directions.sh" "$model" "$kind"
  done
done
