# sgvf.py
import argparse
import os
import json
import torch
import numpy as np
from collections import OrderedDict
import warnings

# ==============================================================================
# 1. CORE SGVF & WEIGHT MANIPULATION LOGIC
# ==============================================================================

def get_expert_state_dict(model_state_dict: OrderedDict, expert_idx: int) -> OrderedDict:
    """
    Extracts the state_dict for a specific expert from a full model state_dict.
    
    Args:
        model_state_dict (OrderedDict): The state_dict of the entire model.
        expert_idx (int): The index of the expert to extract.

    Returns:
        OrderedDict: A new state_dict containing only the specified expert's parameters.
    """
    expert_dict = OrderedDict()
    expert_name_pattern = f'.ffn_expert_{expert_idx}.'
    
    for key, value in model_state_dict.items():
        if expert_name_pattern in key:
            expert_dict[key] = value
            
    if not expert_dict:
        warnings.warn(f"No parameters found for expert index {expert_idx} using pattern '{expert_name_pattern}'. Check model's naming convention.")
        
    return expert_dict

# MODIFIED: Helper function consistent with the provided example
def _cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculates cosine similarity between two flattened tensors."""
    # Ensure tensors are float for dot product
    x_flat = x.flatten().float()
    y_flat = y.flatten().float()
    
    dot_product = torch.dot(x_flat, y_flat)
    norm_x = torch.norm(x_flat)
    norm_y = torch.norm(y_flat)
    
    # Avoid division by zero
    if norm_x == 0 or norm_y == 0:
        return torch.tensor(0.0, device=x.device)
        
    return dot_product / (norm_x * norm_y)

# MODIFIED: SLERP function implementation is now identical to the provided example
def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float) -> torch.Tensor:
    """
    Performs Spherical Linear Interpolation (SLERP) between two vectors.
    This version operates on the original vectors, preserving magnitude in the interpolation,
    matching the provided reference code.
    """
    # Calculate the angle between the vectors
    omega = torch.acos(
        torch.clamp(_cosine_similarity(v0, v1), -1.0, 1.0)  # Clamp for numerical stability
    )
    sin_omega = torch.sin(omega)

    # If vectors are nearly collinear, sin_omega is close to 0.
    # In this case, linear interpolation is a stable fallback.
    if sin_omega.abs() < 1e-8:
        return (1.0 - t) * v0 + t * v1

    # Standard SLERP formula applied to original vectors
    a = torch.sin((1.0 - t) * omega) / sin_omega
    b = torch.sin(t * omega) / sin_omega
    
    return a * v0 + b * v1


def fuse_experts_with_sgvf(base_expert_sd: OrderedDict, teacher_expert_sd: OrderedDict, student_expert_sd: OrderedDict, gamma: float, teacher_expert_id: int, student_expert_id: int) -> OrderedDict:
    """
    Fuses teacher and student expert weights using SGVF on their variation vectors.
    Corresponds to equations (8), (11), and (13) in the paper.

    Args:
        base_expert_sd (OrderedDict): State dict of the base expert (E0).
        teacher_expert_sd (OrderedDict): State dict of the teacher expert (Em*).
        student_expert_sd (OrderedDict): State dict of the student expert (E'm*).
        gamma (float): Interpolation ratio.
        teacher_expert_id (int): The ID of the teacher expert.
        student_expert_id (int): The ID of the student expert.

    Returns:
        OrderedDict: The state dict of the new, fused expert.
    """
    fused_expert_sd = OrderedDict()
    
    teacher_pattern = f'.ffn_expert_{teacher_expert_id}.'
    base_pattern = '.ffn_expert_0.'
    student_pattern = f'.ffn_expert_{student_expert_id}.'

    # Iterate through all parameters of the teacher expert
    for key in teacher_expert_sd.keys():
        base_key = key.replace(teacher_pattern, base_pattern)
        student_key = key.replace(teacher_pattern, student_pattern)
        
        if base_key not in base_expert_sd or student_key not in student_expert_sd:
            raise KeyError(f"Mismatch in parameter keys: could not find corresponding key for '{key}'. Looking for '{base_key}' and '{student_key}'.")

        W_base = base_expert_sd[base_key].float()
        W_teacher = teacher_expert_sd[key].float()
        W_student = student_expert_sd[student_key].float()

        # 1. Calculate Variation Vectors (V_m*^0 and V_m*^1 in the paper)
        V_teacher = W_teacher - W_base
        V_student = W_student - W_base
        
        # 2. Perform SLERP on the variation vectors (no flattening needed if slerp handles it)
        # The new slerp implementation internally flattens for cosine similarity, so we pass the original tensors.
        V_fused = slerp(V_teacher, V_student, t=gamma)
        
        # 3. Calculate the new fused weight (W_m*')
        W_fused = W_base + V_fused
        
        # The output keys should match the teacher expert, as we are updating it.
        fused_expert_sd[key] = W_fused.type_as(teacher_expert_sd[key])
        
    return fused_expert_sd


# ==============================================================================
# 2. WORKFLOW MODES
# ==============================================================================

def handle_update_config_mode(args):
    """
    Handles the logic for a domain-incremental task.
    Merges a new expert's weights into the main model file and updates the config.
    """
    print(f"[MODE] Update Config for Domain-Incremental Task")
    print(f"  > Merging new expert {args.new_expert_id} into model.")

    # 1. Load the models' state dicts, extracting from the checkpoint dictionary
    base_model_sd = torch.load(args.base_weights, map_location='cpu')['model']
    trained_model_sd = torch.load(args.trained_weights, map_location='cpu')['model']
    
    # 2. Extract the newly trained expert's weights
    new_expert_sd = get_expert_state_dict(trained_model_sd, args.new_expert_id)

    if not new_expert_sd:
        raise ValueError(f"Could not extract any weights for the new expert {args.new_expert_id}. Aborting.")
    
    # 3. Merge the new expert weights into the base model state dict
    final_model_sd = base_model_sd.copy()
    final_model_sd.update(new_expert_sd)
    
    # 4. Save the new combined model in the standard checkpoint format
    torch.save({'model': final_model_sd}, args.output_weights)
    print(f"  > New model with expert {args.new_expert_id} merged saved to: {args.output_weights}")
    
    # 5. Update the experts.json config file
    with open(args.experts_config, 'r') as f:
        experts_data = json.load(f)
    
    experts_data['total_experts'] += 1
    new_expert_info = {
        "id": args.new_expert_id,
        "type": "domain",
        "routing_vector_path": os.path.join(os.path.dirname(experts_data['experts'][0]['routing_vector_path']), f"expert_{args.new_expert_id}.npy")
    }
    experts_data['experts'].append(new_expert_info)
    
    with open(args.experts_config, 'w') as f:
        json.dump(experts_data, f, indent=2)
    print(f"  > Updated {args.experts_config} with new expert info.")


def handle_fuse_mode(args):
    """
    Handles the logic for a class-incremental task.
    Fuses a student expert's knowledge into a teacher expert via SGVF.
    """
    print(f"[MODE] Fuse Knowledge for Class-Incremental Task")
    print(f"  > Fusing student (temp expert {args.student_expert_id}) into teacher (expert {args.teacher_expert_id}).")

    # 1. Load the models' state dicts, extracting from the checkpoint dictionary
    base_model_sd = torch.load(args.base_weights, map_location='cpu')['model']
    trained_model_sd = torch.load(args.trained_weights, map_location='cpu')['model']

    # 2. Extract state dicts for all three required experts
    print("  > Extracting expert weights...")
    base_expert_sd = get_expert_state_dict(base_model_sd, 0) # E0
    teacher_expert_sd = get_expert_state_dict(base_model_sd, args.teacher_expert_id) # Em*
    student_expert_sd = get_expert_state_dict(trained_model_sd, args.student_expert_id) # E'm*

    if not base_expert_sd or not teacher_expert_sd or not student_expert_sd:
        raise ValueError("Could not extract weights for one or more required experts (base, teacher, or student). Aborting fusion.")
    
    # 3. Perform the fusion using SGVF
    print(f"  > Performing SGVF with gamma = {args.gamma}...")
    fused_expert_sd = fuse_experts_with_sgvf(
        base_expert_sd, 
        teacher_expert_sd, 
        student_expert_sd, 
        args.gamma,
        teacher_expert_id=args.teacher_expert_id,
        student_expert_id=args.student_expert_id
    )
    
    # 4. Update the base model's state dict with the new fused weights
    final_model_sd = base_model_sd.copy()
    final_model_sd.update(fused_expert_sd)
    
    # 5. Save the new fused model in the standard checkpoint format
    torch.save({'model': final_model_sd}, args.output_weights)
    print(f"  > New fused model saved to: {args.output_weights}")
    
    # 6. No config change is needed for class-incremental tasks.
    print(f"  > {args.experts_config} remains unchanged as total expert count is stable.")


# ==============================================================================
# 3. SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGVF Fusion and Model Update Script for Incre-MoE")
    
    parser.add_argument('--mode', type=str, required=True, choices=['update_config', 'fuse'],
                        help="The operation mode: 'update_config' for domain-inc, 'fuse' for class-inc.")
    
    # Common path arguments
    parser.add_argument('--experts_config', type=str, required=True, help="Path to the JSON file describing experts.")
    parser.add_argument('--base_weights', type=str, required=True, help="Path to the model weights before the current training task.")
    parser.add_argument('--trained_weights', type=str, required=True, help="Path to the model weights after training (output of train_main.py).")
    parser.add_argument('--output_weights', type=str, required=True, help="Path to save the final, processed model weights.")
    
    # Arguments for 'update_config' mode
    parser.add_argument('--new_expert_id', type=int, help="ID of the new expert to add (for domain-inc).")
    
    # Arguments for 'fuse' mode
    parser.add_argument('--teacher_expert_id', type=int, help="ID of the teacher expert (for class-inc).")
    parser.add_argument('--student_expert_id', type=int, help="ID of the temporary student expert (for class-inc).")
    parser.add_argument('--gamma', type=float, default=0.5, help="Interpolation ratio for SGVF fusion.")

    args = parser.parse_args()

    # Validate arguments based on mode
    if args.mode == 'update_config' and args.new_expert_id is None:
        parser.error("--new_expert_id is required for 'update_config' mode.")
    if args.mode == 'fuse' and (args.teacher_expert_id is None or args.student_expert_id is None):
        parser.error("--teacher_expert_id and --student_expert_id are required for 'fuse' mode.")

    if args.mode == 'update_config':
        handle_update_config_mode(args)
    elif args.mode == 'fuse':
        handle_fuse_mode(args)
    
    print("\nSGVF script finished successfully.")