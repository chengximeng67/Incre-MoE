# fdr_train.py
import argparse
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import warnings

# ==============================================================================
# 1. CORE FDR LOGIC (Based on the paper)
# ==============================================================================

def calculate_frequency_feature(image_path: str, resize_dim: tuple = (256, 256)) -> np.ndarray:
    """
    Calculates the complex frequency-domain feature for a single image.
    Corresponds to equations (2) and (3) in the paper.

    Args:
        image_path (str): Path to the image file.
        resize_dim (tuple): The fixed resolution to resize images to for comparability.

    Returns:
        np.ndarray: A flattened 1D complex vector representing the frequency feature.
    """
    try:
        # Open image, convert to grayscale (simplifies feature extraction) and resize
        with Image.open(image_path).convert('L') as img:
            img_resized = img.resize(resize_dim, Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img_resized)
        
        # Apply 2D Discrete Fourier Transform (DFT) using FFT algorithm
        # This gives us a complex-valued matrix F_n
        fft_matrix = np.fft.fft2(img_array)
        
        # Shift the zero-frequency component to the center for consistency
        fft_shifted = np.fft.fftshift(fft_matrix)
        
        # Flatten the complex matrix into a 1D vector (f_n)
        feature_vector = fft_shifted.flatten()
        
        return feature_vector
    except Exception as e:
        warnings.warn(f"Could not process image {image_path}: {e}")
        return None


def get_task_routing_feature(image_dir: str, sample_limit: int = 200) -> np.ndarray:
    """
    Calculates the average routing feature for an entire task (dataset).
    This corresponds to equation (4) in the paper.

    Args:
        image_dir (str): Directory containing the images for the new task.
        sample_limit (int): Maximum number of images to sample for efficiency.

    Returns:
        np.ndarray: The centroid (mean vector) of the frequency features.
    """
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    if not image_files:
        raise FileNotFoundError(f"No images found in directory: {image_dir}")
        
    # Limit the number of samples to speed up the process
    if len(image_files) > sample_limit:
        image_files = np.random.choice(image_files, sample_limit, replace=False)
        
    feature_vectors = []
    print(f"Calculating routing feature from {len(image_files)} sample images...")
    for img_path in tqdm(image_files, desc="Processing Images for FDR"):
        feature = calculate_frequency_feature(img_path)
        if feature is not None:
            feature_vectors.append(feature)
    
    if not feature_vectors:
        raise ValueError("Could not extract any valid features from the provided image directory.")

    # Calculate the centroid (mean) of all feature vectors
    task_feature = np.mean(feature_vectors, axis=0)
    return task_feature


def calculate_similarity(vec1: np.ndarray, vec2: np.ndarray, alpha: float = 0.6) -> float:
    """
    Calculates the weighted cosine similarity between two complex vectors.
    This corresponds to equation (5) in the paper.

    Args:
        vec1 (np.ndarray): First complex vector.
        vec2 (np.ndarray): Second complex vector.
        alpha (float): Hyperparameter to balance real and imaginary parts.

    Returns:
        float: The similarity score.
    """
    # Normalize vectors to prevent magnitude bias
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    vec1_norm = vec1 / norm1
    vec2_norm = vec2 / norm2
    
    # Complex dot product
    dot_product = np.vdot(vec1_norm, vec2_norm)
    
    # Real part corresponds to magnitude similarity, imaginary part to phase similarity
    real_part_sim = np.real(dot_product)
    imag_part_sim = np.imag(dot_product)
    
    # Weighted sum as per the formula
    similarity = alpha * real_part_sim + (1 - alpha) * imag_part_sim
    return similarity


# ==============================================================================
# 2. WORKFLOW AND DECISION LOGIC
# ==============================================================================

def main(args):
    """
    Main function to orchestrate the FDR process and make a decision.
    """
    # --- Step 1: Load existing expert configuration ---
    print(f"Loading expert configuration from: {args.experts_config}")
    with open(args.experts_config, 'r') as f:
        experts_data = json.load(f)
    
    total_experts = experts_data['total_experts']
    
    # --- Step 2: Calculate routing feature for the new task ---
    print(f"\nCalculating routing feature for the new task in: {args.new_task_data_dir}")
    new_task_feature = get_task_routing_feature(args.new_task_data_dir)
    
    # --- Step 3: Load existing routing vectors and find the most similar expert ---
    print("\nComparing new task with existing experts...")
    expert_similarities = []
    for expert in experts_data['experts']:
        expert_id = expert['id']
        routing_vector_path = expert['routing_vector_path']
        
        if not os.path.exists(routing_vector_path):
            # This should only happen for the base expert on the very first run.
            if expert_id == 0:
                 warnings.warn(f"Routing vector for base expert not found at {routing_vector_path}. Assuming it's the first run and assigning zero similarity.")
                 # Treat the base vector as the same as the first task's vector to force a domain-inc decision.
                 # This is a practical bootstrap: the first task *must* be domain-incremental.
                 if total_experts == 1:
                     similarity = 1.0
                 else:
                     similarity = 0.0
            else:
                raise FileNotFoundError(f"Routing vector for expert {expert_id} not found at {routing_vector_path}")
        else:
            existing_vector = np.load(routing_vector_path)
            similarity = calculate_similarity(new_task_feature, existing_vector)

        expert_similarities.append((similarity, expert_id))
        print(f"  - Similarity with Expert {expert_id}: {similarity:.4f}")

    # Determine the best expert based on its ID, not its list index, to match model definition
    _, best_expert_id = max(expert_similarities, key=lambda item: item[0])
    
    # --- Step 4: Make the decision based on the best match ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    if best_expert_id == 0:
        # --- Decision: Domain-Incremental Task ---
        task_type = "domain_incremental"
        teacher_idx = 0  # Teacher is always the base expert for creating new domains
        student_idx = total_experts # The new expert will have the next available index
        should_fuse = "no_fuse"
        
        print(f"\nDecision: Task is DOMAIN-INCREMENTAL. Most similar to Base Expert ({best_expert_id}).")
        print(f"A new expert will be created at index {student_idx}.")
        
        # Create and save the routing vector for the new expert. Corresponds to Eq. (6)
        new_vector_path = os.path.join(os.path.dirname(experts_data['experts'][0]['routing_vector_path']), f"expert_{student_idx}.npy")
        np.save(new_vector_path, new_task_feature)
        print(f"  - Saved new routing vector to: {new_vector_path}")
        
    else:
        # --- Decision: Class-Incremental Task ---
        task_type = "class_incremental"
        teacher_idx = best_expert_id
        student_idx = 99  # Use a temporary index for the student expert during training
        should_fuse = "fuse"
        
        print(f"\nDecision: Task is CLASS-INCREMENTAL. Most similar to Domain Expert {teacher_idx}.")
        
        # Update the existing expert's routing vector via exponential moving average
        # Corresponds to the update rule below Eq. (7)
        beta = 0.5  # This can be a hyperparameter, e.g., n/(n+1) from the paper
        
        # Find the expert's config by its ID to get the correct path
        target_expert_config = next((exp for exp in experts_data['experts'] if exp['id'] == teacher_idx), None)
        if target_expert_config is None:
            raise ValueError(f"Could not find expert with ID {teacher_idx} in the config file.")
        existing_vector_path = target_expert_config['routing_vector_path']

        existing_vector = np.load(existing_vector_path)
        updated_vector = beta * existing_vector + (1 - beta) * new_task_feature
        np.save(existing_vector_path, updated_vector)
        print(f"  - Updated routing vector for expert {teacher_idx} at: {existing_vector_path}")
        
    # --- Step 5: Output the decision for the shell script ---
    # The format is critical for the `read` command in the shell script
    print(f"\nFDR_OUTPUT:{task_type} {teacher_idx} {student_idx} {should_fuse}")


# ==============================================================================
# 3. SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frequency-Domain Router (FDR) for Incre-MoE Training")
    
    parser.add_argument('--new_task_data_dir', type=str, required=True,
                        help="Path to the directory containing images for the new task.")
    parser.add_argument('--experts_config', type=str, required=True,
                        help="Path to the JSON file describing the current experts and their routing vectors.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save any new or updated routing vectors.")
    
    args = parser.parse_args()
    
    main(args)