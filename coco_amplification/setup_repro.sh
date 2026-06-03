#!/usr/bin/env bash
#
# COCO amplification reproduction setup.
#
# From the repo root:
#     bash coco_amplification/setup_repro.sh --model qwen
#     bash coco_amplification/setup_repro.sh --model gemma
#     bash coco_amplification/setup_repro.sh --model pixtral
#
# In Colab/current Python environments:
#     bash coco_amplification/setup_repro.sh --system
set -euo pipefail

TARGET="venv"
FRESH="false"
MODEL="qwen"

usage() {
  cat <<'EOF'
Usage:
  bash coco_amplification/setup_repro.sh --model qwen
      Create .venv and install the Qwen COCO reproduction deps.

  bash coco_amplification/setup_repro.sh --model gemma
      Create .venv and install the Gemma COCO reproduction deps.

  bash coco_amplification/setup_repro.sh --model pixtral
      Create .venv and install the Pixtral COCO reproduction deps.

  bash coco_amplification/setup_repro.sh --model pixtral --system
      Install into the current Python environment. Use this in Colab.

  bash coco_amplification/setup_repro.sh --model qwen --fresh
      Remove the existing .venv first, then create a clean reproduction env.

Options:
  --model NAME   Reproduction profile: qwen, gemma, or pixtral. Default: qwen.
  --system       Install into the current Python environment instead of .venv.
  --fresh        Remove .venv before creating the reproduction environment.
  --help         Show this message.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model)
      if [ $# -lt 2 ]; then
        echo "--model requires qwen, gemma, or pixtral" >&2
        exit 2
      fi
      MODEL="$2"
      shift
      ;;
    --system)
      TARGET="system"
      ;;
    --fresh)
      FRESH="true"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

COMMON_DEPS=(
  numpy
  tokenizers accelerate
  transformer-lens==2.11.0
  einops fancy-einsum jaxtyping typeguard better-abc
  torch torchvision
  datasets
  pandas rich wandb
  matplotlib pillow tqdm ipykernel
  huggingface_hub
)

case "$MODEL" in
  qwen)
    DEPS=("transformers==4.46.3" qwen-vl-utils "${COMMON_DEPS[@]}")
    ;;
  gemma)
    DEPS=("transformers==4.50.0" "${COMMON_DEPS[@]}")
    ;;
  pixtral)
    DEPS=("transformers==4.50.0" "${COMMON_DEPS[@]}")
    ;;
  *)
    echo "Unknown --model value: $MODEL (expected qwen, gemma, or pixtral)" >&2
    usage >&2
    exit 2
    ;;
esac

if [ "$TARGET" = "system" ]; then
  python -m pip install "${DEPS[@]}"
  echo
  echo "COCO $MODEL reproduction dependencies installed into the current Python environment."
else
  if [ "$FRESH" = "true" ] && [ -d .venv ]; then
    rm -rf .venv
  fi
  uv venv --python 3.11
  uv pip install --python .venv/bin/python "${DEPS[@]}"
  echo
  echo "COCO $MODEL reproduction dependencies installed into .venv - activate with: source .venv/bin/activate"
fi
