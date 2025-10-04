# fdr_inference.py
import argparse
import os
import json
import numpy as np
from PIL import Image
import warnings

# ==============================================================================
# 1. CORE FDR LOGIC (Reused from fdr_train.py)
# ==============================================================================

def calculate_frequency_feature(image_path: str, resize_dim: tuple = (256, 256)) -> np.ndarray:
    """Calculates the complex frequency-domain feature for a single image."""
    try:
        with Image.open(image_path).convert('L') as img:
            img_resized = img.resize(resize_dim, Image.LANCZOS)
        img_array = np.array(img_resized)
        fft_matrix = np.fft.fft2(img_array)
        fft_shifted = np.fft.fftshift(fft_matrix)
        feature_vector = fft_shifted.flatten()
        return feature_vector
    except Exception as e:
        warnings.warn(f"Could not process image {image_path}: {e}")
        return None

def calculate_similarity(vec1: np.ndarray, vec2: np.ndarray, alpha: float = 0.6) -> float:
    """Calculates the weighted cosine similarity between two complex vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    vec1_norm = vec1 / norm1
    vec2_norm = vec2 / norm2
    dot_product = np.vdot(vec1_norm, vec2_norm)
    real_part_sim = np.real(dot_product)
    imag_part_sim = np.imag(dot_product)
    similarity = alpha * real_part_sim + (1 - alpha) * imag_part_sim
    return similarity

# ==============================================================================
# 2. INFERENCE WORKFLOW
# ==============================================================================

def main(args):
    """
    Determines the best expert for a single input image and prints its ID.
    """
    # --- Step 1: Load existing expert configuration ---
    with open(args.experts_config, 'r') as f:
        experts_data = json.load(f)
    
    # --- Step 2: Calculate routing feature for the input image ---
    image_feature = calculate_frequency_feature(args.image_path)
    if image_feature is None:
        raise ValueError(f"Failed to calculate feature for image: {args.image_path}")

    # --- Step 3: Find the most similar expert ---
    expert_similarities = []
    for expert in experts_data['experts']:
        expert_id = expert['id']
        routing_vector_path = expert['routing_vector_path']
        
        if not os.path.exists(routing_vector_path):
             warnings.warn(f"Routing vector for expert {expert_id} not found at {routing_vector_path}. Skipping.")
             continue
        
        existing_vector = np.load(routing_vector_path)
        similarity = calculate_similarity(image_feature, existing_vector)
        expert_similarities.append((similarity, expert_id))

    if not expert_similarities:
        raise RuntimeError("No valid experts found or routing vectors are missing.")

    # --- Step 4: Determine the best expert and output its ID ---
    # Find the expert with the maximum similarity score
    _, best_expert_id = max(expert_similarities, key=lambda item: item[0])
    
    # This print is the output captured by the shell script.
    print(best_expert_id)

# ==============================================================================
# 3. SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FDR Inference: Selects the best expert for a single image.")
    
    parser.add_argument('--image_path', type=str, required=True,
                        help="Path to the input image file for inference.")
    parser.add_argument('--experts_config', type=str, required=True,
                        help="Path to the JSON file describing the current experts and their routing vectors.")
    
    args = parser.parse_args()
    main(args)