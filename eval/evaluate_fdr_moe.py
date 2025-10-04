# evaluate_fdr_moe.py

import argparse
import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import csv
import warnings
from PIL import Image

# Grounding DINO imports
from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator
import torchvision

# Suppress warnings for cleaner output
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="The `device` argument is deprecated and will be removed in v5 of Transformers.")
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

# ==============================================================================
# 1. FDR CORE LOGIC (from your new FDR code)
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

def get_best_expert_for_image(image_path: str, experts_data: dict) -> int:
    """
    Determines the best expert for a single input image based on pre-calculated routing vectors.
    """
    image_feature = calculate_frequency_feature(image_path)
    if image_feature is None:
        warnings.warn(f"Failed to calculate feature for {image_path}, defaulting to expert 0.")
        return 0

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
        warnings.warn("No valid experts found, defaulting to expert 0.")
        return 0

    _, best_expert_id = max(expert_similarities, key=lambda item: item[0])
    return best_expert_id


# ==============================================================================
# 2. MODEL & DATA HANDLING (Adapted from old and new code)
# ==============================================================================

def load_moe_model(model_config_path: str, model_checkpoint_path: str, experts_config_path: str, device: str = "cuda"):
    """
    Loads the MoE model, dynamically setting the number of experts from the config.
    """
    print("Loading model configuration...")
    args = SLConfig.fromfile(model_config_path)
    args.device = device

    # Dynamically set the number of experts based on the experts.json file
    print(f"Loading expert configuration from: {experts_config_path}")
    with open(experts_config_path, 'r') as f:
        experts_data = json.load(f)
    num_experts = experts_data.get('total_experts')
    if num_experts is None:
        raise ValueError("`total_experts` not found in experts_config JSON.")
    
    # Set number of experts in model config before building
    args.model.transformer.num_experts = num_experts
    print(f"Building model with {num_experts} experts...")
    model = build_model(args)

    print(f"Loading weights from: {model_checkpoint_path}")
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    
    if "model" not in checkpoint:
        checkpoint = {"model": checkpoint}

    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.to(device)
    model.eval()
    return model, experts_data


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        w, h = img.size
        boxes = [obj["bbox"] for obj in target]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        target_new = {}
        image_id = self.ids[idx]
        target_new["image_id"] = image_id
        target_new["boxes"] = boxes
        target_new["orig_size"] = torch.as_tensor([int(h), int(w)])

        if self._transforms is not None:
            img, target = self._transforms(img, target_new)
        return img, target


class PostProcessCocoGrounding(nn.Module):
    def __init__(self, cat_list, tokenlizer, num_select=300) -> None:
        super().__init__()
        self.num_select = num_select
        
        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
        positive_map = create_positive_map_from_span(
            tokenlizer(captions), tokenspanlist
        )
        
        # In this general case, we map the first class to ID 1, second to ID 2, etc.
        # This assumes the category IDs in the COCO json are 1, 2, 3...
        max_cat_id = 91 # Standard COCO max categories
        new_pos_map = torch.zeros((max_cat_id, 256))
        for i, _ in enumerate(cat_list):
            new_pos_map[i+1] = positive_map[i]
        self.positive_map = new_pos_map

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob_to_token = out_logits.sigmoid()
        pos_maps = self.positive_map.to(prob_to_token.device)
        prob_to_label = prob_to_token @ pos_maps.T
        
        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels = topk_indexes % prob.shape[2]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results

# ==============================================================================
# 3. MAIN EVALUATION WORKFLOW
# ==============================================================================

def main_evaluate(args):
    # Load MoE model and FDR expert configurations
    model, experts_data = load_moe_model(args.config_file, args.checkpoint_path, args.experts_config, args.device)

    # Build dataloader
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = CocoDetection(args.image_dir, args.anno_path, transforms=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # Build post processor
    cfg = SLConfig.fromfile(args.config_file)
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    
    # Use evaluation classes passed from command line
    eval_classes = [c.strip() for c in args.eval_classes.split(',')]
    postprocessor = PostProcessCocoGrounding(cat_list=eval_classes, tokenlizer=tokenlizer)

    # Build evaluator
    evaluator = CocoGroundingEvaluator(dataset.coco, iou_types=("bbox",), useCats=True)

    # Build caption from evaluation classes
    caption = " . ".join(eval_classes) + ' .'
    print(f"\nInput text prompt for evaluation: \"{caption}\"\n")

    # Run inference with FDR
    start_time = time.time()
    for i, (images, targets) in enumerate(data_loader):
        images = images.tensors.to(args.device)
        bs = images.shape[0]
        
        # This loop assumes batch size is 1 for FDR routing per image
        if bs != 1:
            raise ValueError("Evaluation with FDR requires a batch size of 1.")
            
        target = targets[0]
        image_id = target["image_id"]
        
        # Get image path for FDR
        image_info = dataset.coco.loadImgs(image_id)[0]
        image_path = os.path.join(args.image_dir, image_info['file_name'])

        # --- FDR Step: Select the best expert for this image ---
        expert_id_to_use = get_best_expert_for_image(image_path, experts_data)

        # Feed to the model with the selected expert
        outputs = model(images, captions=[caption], expert_idx_to_use=expert_id_to_use)
        
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(images.device)
        results = postprocessor(outputs, orig_target_sizes)
        
        cocogrounding_res = {image_id: results[0]}
        evaluator.update(cocogrounding_res)

        if (i + 1) % 50 == 0 or (i + 1) == len(data_loader):
            used_time = time.time() - start_time
            eta = (len(data_loader) / (i + 1e-5) * used_time) - used_time
            print(f"Processed {i+1}/{len(data_loader)} images. "
                  f"Last used Expert ID: {expert_id_to_use}. "
                  f"Time: {used_time:.2f}s, ETA: {eta:.2f}s")

    # Finalize and print results
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    print("\nFinal evaluation results:")
    print(evaluator.coco_eval["bbox"].stats.tolist())


# ==============================================================================
# 4. SCRIPT ENTRY POINT & ARGUMENT PARSING
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO FDR-MoE Evaluation Script", add_help=True)
    
    # Path Arguments
    parser.add_argument("--config_file", "-c", type=str, required=True, help="Path to model config file.")
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True, help="Path to checkpoint file.")
    parser.add_argument("--experts_config", type=str, required=True, help="Path to the JSON file describing experts.")
    parser.add_argument("--anno_path", type=str, required=True, help="Path to COCO format annotation file (.json).")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing images.")
    
    # Evaluation Arguments
    parser.add_argument("--eval_classes", type=str, required=True, help="Comma-separated list of classes to evaluate (e.g., 'smoke,fire').")
    
    # Technical Arguments
    parser.add_argument("--device", type=str, default="cuda", help="Running device ('cuda' or 'cpu').")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader.")

    args = parser.parse_args()

    # Argument Validation
    if not torch.cuda.is_available() and args.device == 'cuda':
        print("Warning: CUDA is not available. Switching to CPU.")
        args.device = 'cpu'
        
    main_evaluate(args)