#!/bin/bash


set -euo pipefail

readonly IMAGE_PATH="data/smoke_task/test_images/000021.jpg"

readonly TEXT_PROMPT="smoke"

readonly OUTPUT_IMAGE_PATH="output/inference_result.jpg"

readonly MODEL_CONFIG="configs/GroundingDINO_SwinB_MoE.py"

readonly EXPERTS_CONFIG="configs/experts.json"

readonly MODEL_WEIGHTS="weights/latest_model.pth"

readonly BOX_THRESHOLD=0.35
readonly TEXT_THRESHOLD=0.25
readonly DEVICE="cuda"


echo "======================================================"
echo "  Starting Incre-MoE Inference Workflow"
echo "======================================================"
echo "  > Input Image: ${IMAGE_PATH}"
echo "  > Text Prompt: '${TEXT_PROMPT}'"
echo "------------------------------------------------------"


echo "[Step 1/2] Running FDR to select the best expert..."


expert_id=$(python fdr_inference.py \
    --image_path "${IMAGE_PATH}" \
    --experts_config "${EXPERTS_CONFIG}")

if ! [[ "${expert_id}" =~ ^[0-9]+$ ]]; then
    echo "Error: FDR script did not return a valid expert ID. Output was: '${expert_id}'. Aborting."
    exit 1
fi

echo "  > FDR selected Expert ID: ${expert_id}"
echo "------------------------------------------------------"



echo "[Step 2/2] Running main inference with selected expert..."

python inference.py \
    --model_config "${MODEL_CONFIG}" \
    --weights_path "${MODEL_WEIGHTS}" \
    --experts_config "${EXPERTS_CONFIG}" \
    --image_path "${IMAGE_PATH}" \
    --text_prompt "${TEXT_PROMPT}" \
    --expert_id "${expert_id}" \
    --output_path "${OUTPUT_IMAGE_PATH}" \
    --box_threshold "${BOX_THRESHOLD}" \
    --text_threshold "${TEXT_THRESHOLD}" \
    --device "${DEVICE}"

echo "------------------------------------------------------"
echo "  > Inference complete."
echo "  > Annotated image saved to: ${OUTPUT_IMAGE_PATH}"
echo ""
echo "======================================================"
echo "  Inference workflow finished successfully!"
echo "======================================================"