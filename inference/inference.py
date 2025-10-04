# inference.py
import argparse
import os
import json
import torch
import cv2

# GroundingDINO specific imports
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict

def main(args):
    # --- Step 1: Load Model Configuration ---
    print("Loading model configuration...")
    config = SLConfig.fromfile(args.model_config)
    
    # Dynamically set the number of experts based on the config file
    with open(args.experts_config, 'r') as f:
        experts_data = json.load(f)
    config.model.transformer.num_experts = experts_data['total_experts']
    
    # --- Step 2: Build and Load Model with Weights ---
    print(f"Building model with {config.model.transformer.num_experts} experts...")
    model = build_model(config)
    
    print(f"Loading weights from: {args.weights_path}")
    checkpoint = torch.load(args.weights_path, map_location='cpu')
    
    # Handle both checkpoint dictionary and raw state_dict formats
    if "model" in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    # --- Step 3: Load Image ---
    image_source, image = load_image(args.image_path)

    # --- Step 4: Run Prediction using the SPECIFIED expert ---
    print(f"Running prediction with Expert ID: {args.expert_id}...")
    
    # The 'predict' function from the library doesn't support passing expert_id.
    # We call the model directly to pass our custom argument.
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
        # This is the key modification for Incre-MoE inference
        expert_idx_to_use=args.expert_id
    )

    # --- Step 5: Annotate and Save Result ---
    print(f"Saving annotated image to: {args.output_path}")
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    cv2.imwrite(args.output_path, annotated_frame)
    print("Inference finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Incre-MoE Inference Script for GroundingDINO")

    # Path Arguments
    parser.add_argument('--model_config', type=str, required=True, help="Path to the model config file (.py).")
    parser.add_argument('--weights_path', type=str, required=True, help="Path to the model weights file (.pth).")
    parser.add_argument('--experts_config', type=str, required=True, help="Path to the JSON file describing experts.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the annotated output image.")
    
    # Inference Arguments
    parser.add_argument('--text_prompt', type=str, required=True, help="Text prompt for object detection.")
    parser.add_argument('--expert_id', type=int, required=True, help="The expert ID selected by FDR to use for this inference.")
    
    # Hyperparameter Arguments
    parser.add_argument('--box_threshold', type=float, default=0.35, help="Box detection threshold.")
    parser.add_argument('--text_threshold', type=float, default=0.25, help="Text association threshold.")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use ('cuda' or 'cpu').")

    args = parser.parse_args()

    # Argument Validation
    if not torch.cuda.is_available() and args.device == 'cuda':
        print("Warning: CUDA is not available. Switching to CPU.")
        args.device = 'cpu'
        
    main(args)