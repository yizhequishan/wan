#!/usr/bin/env bash
set -euo pipefail

# Run from the Wan repository regardless of the caller's current directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CKPT_DIR="${CKPT_DIR:-/data/ychu/models/Wan2.1-T2V-1.3B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/row_probe}"
SEED="${SEED:-20250308}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/two_objects_seed_${SEED}.mp4}"

PROMPT='A minimal 3D animation in a single continuous fixed wide shot. On a plain light-gray tabletop, a glossy red sphere starts on the left and moves steadily to the right. A matte blue cube starts on the right and moves steadily to the left. Around halfway through the video, they collide exactly once near the center, briefly touch without overlapping, then rebound and move apart. Both objects remain fully visible and preserve their color, shape, and size throughout the video. Strong object-background contrast, soft shadows, static camera, no zoom, no cuts, no extra objects.'

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "Checkpoint directory does not exist: ${CKPT_DIR}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c 'from PIL import Image' >/dev/null 2>&1; then
  echo "Pillow is missing from ${PYTHON_BIN}." >&2
  echo "Install it with: ${PYTHON_BIN} -m pip install Pillow" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Repository : ${REPO_ROOT}"
echo "GPU        : physical GPU ${GPU_ID}"
echo "Checkpoint : ${CKPT_DIR}"
echo "Seed       : ${SEED}"
echo "Output     : ${OUTPUT_FILE}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" generate.py \
  --task t2v-1.3B \
  --size '832*480' \
  --frame_num 81 \
  --ckpt_dir "${CKPT_DIR}" \
  --offload_model True \
  --t5_cpu \
  --sample_solver unipc \
  --sample_steps 50 \
  --sample_shift 8 \
  --sample_guide_scale 6 \
  --base_seed "${SEED}" \
  --prompt "${PROMPT}" \
  --save_file "${OUTPUT_FILE}"

echo "Finished: ${OUTPUT_FILE}"
