#!/bin/bash

set -euo pipefail

readonly TASK_NAME="forest_fire_smoke_detection"


readonly TRAIN_ANN_FILE="data/smoke_task/train.csv"
readonly TRAIN_IMAGE_DIR="data/smoke_task/train_images"
readonly TEST_ANN_FILE="data/smoke_task/test.csv"
readonly TEST_IMAGE_DIR="data/smoke_task/test_images"


readonly MODEL_CONFIG="configs/GroundingDINO_SwinB_MoE.py"

readonly EXPERTS_CONFIG="configs/experts.json"

readonly CURRENT_MODEL_WEIGHTS="weights/latest_model.pth"

readonly BASE_OUTPUT_DIR="output"

readonly TASK_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TASK_NAME}"

# --- 训练超参数 ---
readonly EPOCHS=100
readonly LEARNING_RATE=2e-5
readonly DEVICE="cuda"

readonly TEST_CATEGORIES="smoke"

readonly SGVF_GAMMA=0.6


echo "======================================================"
echo "  Starting Incre-MoE Training Workflow for Task: ${TASK_NAME}"
echo "======================================================"

echo "[Step 1/4] Running FDR to determine task type..."

if [ ! -f "${EXPERTS_CONFIG}" ]; then
    echo "Expert config not found. Creating a default one at ${EXPERTS_CONFIG}"
    mkdir -p "$(dirname "routing_vectors/expert_0.npy")"
    echo '{"total_experts": 1, "experts": [{"id": 0, "type": "base", "routing_vector_path": "routing_vectors/expert_0.npy"}]}' > "${EXPERTS_CONFIG}"
fi


if [ ! -f "${CURRENT_MODEL_WEIGHTS}" ]; then
    echo "Error: Base model weights not found at ${CURRENT_MODEL_WEIGHTS}. Please provide an initial model."
    exit 1
fi

fdr_output=$(python fdr_train.py \
    --new_task_data_dir "${TRAIN_IMAGE_DIR}" \
    --experts_config "${EXPERTS_CONFIG}" \
    --output_dir "routing_vectors")

# Extract the part after the prefix
decision_vars=${fdr_output#*FDR_OUTPUT:}

read -r task_type teacher_idx student_idx should_fuse <<< "$decision_vars"


if [ -z "${task_type}" ]; then
    echo "Error: FDR script did not return a valid decision. Full output: ${fdr_output}. Aborting."
    exit 1
fi

echo "  > FDR Decision:"
echo "    - Task Type: ${task_type}"
echo "    - Teacher Expert Index: ${teacher_idx}"
echo "    - Student Expert Index: ${student_idx}"
echo "    - Fusion Required: ${should_fuse}"
echo "------------------------------------------------------"


echo "[Step 2/4] Launching main training process (train_main.py)..."


python train_main.py \
    --task_type "${task_type}" \
    --student_expert_idx "${student_idx}" \
    --teacher_expert_idx "${teacher_idx}" \
    --model_config "${MODEL_CONFIG}" \
    --experts_config "${EXPERTS_CONFIG}" \
    --base_weights "${CURRENT_MODEL_WEIGHTS}" \
    --output_dir "${TASK_OUTPUT_DIR}" \
    --train_ann_file "${TRAIN_ANN_FILE}" \
    --train_image_dir "${TRAIN_IMAGE_DIR}" \
    --test_ann_file "${TEST_ANN_FILE}" \
    --test_image_dir "${TEST_IMAGE_DIR}" \
    --test_categories "${TEST_CATEGORIES}" \
    --epochs "${EPOCHS}" \
    --lr "${LEARNING_RATE}" \
    --device "${DEVICE}"


readonly TRAINED_WEIGHTS_PATH="${TASK_OUTPUT_DIR}/best_model.pth"

if [ ! -f "${TRAINED_WEIGHTS_PATH}" ]; then
    echo "Error: Training finished, but the expected output weight file was not found at ${TRAINED_WEIGHTS_PATH}. Aborting."
    exit 1
fi

echo "  > Training complete. Trained weights saved to ${TRAINED_WEIGHTS_PATH}"
echo "------------------------------------------------------"



echo "[Step 3/4] Launching SGVF module for post-processing..."


readonly NEW_MODEL_WEIGHTS_PATH="weights/model_after_${TASK_NAME}.pth"

if [ "${should_fuse}" == "fuse" ]; then
    echo "  > Mode: Fusing knowledge..."
    python sgvf.py \
        --mode "fuse" \
        --experts_config "${EXPERTS_CONFIG}" \
        --base_weights "${CURRENT_MODEL_WEIGHTS}" \
        --trained_weights "${TRAINED_WEIGHTS_PATH}" \
        --output_weights "${NEW_MODEL_WEIGHTS_PATH}" \
        --teacher_expert_id "${teacher_idx}" \
        --student_expert_id "${student_idx}" \
        --gamma "${SGVF_GAMMA}"

elif [ "${should_fuse}" == "no_fuse" ]; then

    echo "  > Mode: Updating configuration and adding new expert..."
    python sgvf.py \
        --mode "update_config" \
        --experts_config "${EXPERTS_CONFIG}" \
        --base_weights "${CURRENT_MODEL_WEIGHTS}" \
        --trained_weights "${TRAINED_WEIGHTS_PATH}" \
        --output_weights "${NEW_MODEL_WEIGHTS_PATH}" \
        --new_expert_id "${student_idx}"
else
    echo "Error: Invalid 'should_fuse' value from FDR: ${should_fuse}. Expected 'fuse' or 'no_fuse'."
    exit 1
fi

echo "  > SGVF process complete. New model saved to ${NEW_MODEL_WEIGHTS_PATH}"
echo "------------------------------------------------------"



echo "[Step 4/4] Finalizing and updating state..."


mv "${CURRENT_MODEL_WEIGHTS}" "weights/backup_before_${TASK_NAME}.pth"
mv "${NEW_MODEL_WEIGHTS_PATH}" "${CURRENT_MODEL_WEIGHTS}"

echo "  > Updated ${CURRENT_MODEL_WEIGHTS} to the latest version."
echo "  > Backup of previous model saved."
echo ""
echo "======================================================"
echo "  Workflow for task '${TASK_NAME}' completed successfully!"
echo "======================================================"