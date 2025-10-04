#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -euo pipefail

# ==============================================================================
# 1. CONFIGURATION - ALL PATHS AND PARAMETERS ARE DEFINED HERE
# ==============================================================================
echo "--- Setting up configuration ---"

# --- Model & Expert Paths ---
# Path to the GroundingDINO MoE model config file
readonly MODEL_CONFIG="groundingdino/config/GroundingDINO_SwinB_cfg.py"
# Path to the trained MoE model weights
readonly MODEL_WEIGHTS="weights/smoke_fdr_moe_best.pth"
# Path to the JSON file describing experts and their routing vectors
readonly EXPERTS_CONFIG="configs/experts.json"

# --- Dataset Paths ---
# NOTE: This script assumes you have ALREADY prepared a COCO-style .json annotation file.
# The original CSV-to-COCO conversion is a separate pre-processing step.
# Path to the COCO-style annotation file for the test set
readonly ANNO_JSON_PATH="dataset/smoke_dataset/test/annotations.json"
# Path to the directory containing the test images
readonly IMAGE_DIR="dataset/smoke_dataset/test"

# --- Evaluation Parameters ---
# The classes to detect and evaluate, separated by commas.
# This MUST match the classes defined in your ANNO_JSON_PATH.
readonly EVAL_CLASSES="smoke"

# --- Technical Parameters ---
readonly DEVICE="cuda"
readonly NUM_WORKERS=4

echo "Model Config:       ${MODEL_CONFIG}"
echo "Model Weights:      ${MODEL_WEIGHTS}"
echo "Experts Config:     ${EXPERTS_CONFIG}"
echo "Annotation JSON:    ${ANNO_JSON_PATH}"
echo "Image Directory:    ${IMAGE_DIR}"
echo "Evaluation Classes: '${EVAL_CLASSES}'"
echo "Device:             ${DEVICE}"
echo "-----------------------------------"
echo ""

# ==============================================================================
# 2. RUN EVALUATION
# ==============================================================================
echo "--- Starting FDR-MoE Model Evaluation ---"

python evaluate_fdr_moe.py \
    --config_file "${MODEL_CONFIG}" \
    --checkpoint_path "${MODEL_WEIGHTS}" \
    --experts_config "${EXPERTS_CONFIG}" \
    --anno_path "${ANNO_JSON_PATH}" \
    --image_dir "${IMAGE_DIR}" \
    --eval_classes "${EVAL_CLASSES}" \
    --device "${DEVICE}" \
    --num_workers "${NUM_WORKERS}"

echo ""
echo "--- Evaluation finished successfully! ---"