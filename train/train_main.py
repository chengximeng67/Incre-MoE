# train_main.py
import argparse
import os
import sys
import time
import json
import csv
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
import copy

# GroundingDINO specific imports
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator
from groundingdino.util.train import load_model as load_model_from_util, load_image, annotate

warnings.filterwarnings('ignore', category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==============================================================================
# 1. HELPER & UTILITY FUNCTIONS
# ==============================================================================

def check_and_create_directory(directory):
    """Ensures that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_dataset(ann_file, image_path):
    """Reads annotations from a CSV file into a dictionary."""
    ann_Dict = defaultdict(lambda: defaultdict(list))
    with open(ann_file) as file_obj:
        ann_reader = csv.DictReader(file_obj)
        for row in ann_reader:
            img_n = os.path.join(image_path, row['image_name'])
            x1, y1 = int(row['bbox_x']), int(row['bbox_y'])
            x2, y2 = x1 + int(row['bbox_width']), y1 + int(row['bbox_height'])
            label = row['label_name']
            ann_Dict[img_n]['boxes'].append([x1, y1, x2, y2])
            ann_Dict[img_n]['captions'].append(label)
    return ann_Dict

def get_expert_params(model, expert_idx):
    """Dynamically finds and returns the parameters of a specific expert to be trained."""
    params_to_update = []
    # MODIFIED: Match the naming convention from the model definition, e.g., '...ffn_experts.ffn_expert_1...'
    expert_name_pattern = f'.ffn_expert_{expert_idx}.'
    
    print(f"Searching for trainable parameters containing '{expert_name_pattern}'...")
    for name, param in model.named_parameters():
        if expert_name_pattern in name:
            print(f"  - Found trainable parameter: {name}")
            params_to_update.append(param)
    
    if not params_to_update:
        warnings.warn(f"Warning: No parameters found for expert index {expert_idx}. Training might not have any effect.")
    return params_to_update

def train_image_step(model, image, caption_objects, box_target, task_type, teacher_expert_idx, student_expert_idx):
    """Performs a single training step, including detection loss and optional cosine similarity loss."""
    device = next(model.parameters()).device
    image = image.to(device)
    box_target = [torch.Tensor(box_target).to(device)]
    
    # Standard GroundingDINO processing
    captions = " . ".join(caption_objects)
    # The model's forward pass now needs the expert_idx_to_use
    model.forward_expert_id = student_expert_idx # Pass student expert ID to the model
    
    tokenlizer = get_tokenlizer.get_tokenlizer("bert-base-uncased")
    _, cat2tokenspan = build_captions_and_token_span([captions], True)
    tokenspanlist = [cat2tokenspan[cat] for cat in [captions]]
    positive_map = create_positive_map_from_span(tokenlizer(captions), tokenspanlist)

    outputs = model(image[None], captions=[captions], expert_idx_to_use=student_expert_idx)
    loss_dict = outputs['loss_dict']
    detection_loss = sum(loss_dict.values())
    total_loss = detection_loss
    
    # Cosine Similarity Loss for Class-Incremental Learning
    if task_type == 'class_incremental' and teacher_expert_idx != -1:
        def get_flat_weights(m, idx):
            # MODIFIED: Match the expert naming convention from the model definition.
            weights = [p.view(-1) for n, p in m.named_parameters() if f'.ffn_expert_{idx}.' in n]
            return torch.cat(weights) if weights else torch.tensor([], device=device)

        # Base weights are not used in the paper's cosine loss, only teacher and student.
        # But for SGVF logic, base is E0. Here we directly compare teacher and student variation from base.
        base_weights = get_flat_weights(model, 0).detach() # Detach base weights
        teacher_weights = get_flat_weights(model, teacher_expert_idx).detach()
        student_weights = get_flat_weights(model, student_expert_idx)

        if all(w.numel() > 0 for w in [base_weights, teacher_weights, student_weights]):
            v_teacher = teacher_weights - base_weights
            v_student = student_weights - base_weights
            cosine_loss = 1 - F.cosine_similarity(v_teacher, v_student, dim=0, eps=1e-8)
            lambda_cos = 1.0  # Hyperparameter to balance losses
            total_loss += lambda_cos * cosine_loss
            loss_dict['cosine_loss'] = cosine_loss.item()
        else:
            warnings.warn("Could not compute cosine similarity loss due to missing expert weights.")
    
    loss_dict['total_loss'] = total_loss.item()
    return total_loss, loss_dict


# ==============================================================================
# 2. EVALUATION & COCO FORMATTING FUNCTIONS
# ==============================================================================

def convert_csv_to_coco(csv_file, json_file, category_names):
    """Dynamically converts a CSV annotation file to COCO JSON format."""
    if not category_names:
        raise ValueError("category_names list cannot be empty.")

    category_mapping = {name: i + 1 for i, name in enumerate(category_names)}
    coco_categories = [{"id": i + 1, "name": name, "supercategory": name} for i, name in enumerate(category_names)]

    coco_output = {
        "info": {"description": "COCO Dataset"},
        "licenses": [{"id": 1, "name": "Unknown"}],
        "images": [], "annotations": [], "categories": coco_categories
    }

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        annotation_id, current_image_id = 1, 1
        image_name_to_id = {}
        for row in reader:
            image_name = row['image_name']
            if image_name not in image_name_to_id:
                image_name_to_id[image_name] = current_image_id
                image = {"id": current_image_id, "file_name": image_name, "width": int(row['image_width']), "height": int(row['image_height']), "license": 1}
                coco_output["images"].append(image)
                current_image_id += 1
            
            label_name = row['label_name']
            category_id = category_mapping.get(label_name)
            if category_id is None: continue

            annotation = {"id": annotation_id, "image_id": image_name_to_id[image_name], "category_id": category_id,
                          "bbox": [float(row['bbox_x']), float(row['bbox_y']), float(row['bbox_width']), float(row['bbox_height'])],
                          "area": float(row['bbox_width']) * float(row['bbox_height']), "iscrowd": 0}
            coco_output["annotations"].append(annotation)
            annotation_id += 1
    
    with open(json_file, 'w') as output_json_file:
        json.dump(coco_output, output_json_file, indent=4)

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        w, h = img.size
        boxes = torch.as_tensor([obj["bbox"] for obj in target], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w); boxes[:, 1::2].clamp_(min=0, max=h)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        target_new = {"image_id": self.ids[idx], "boxes": boxes, "orig_size": torch.as_tensor([int(h), int(w)])}
        if self._transforms is not None: img, target = self._transforms(img, target_new)
        return img, target

class PostProcessCocoGrounding(nn.Module):
    def __init__(self, num_select=300, coco_api=None, tokenlizer=None):
        super().__init__()
        self.num_select = num_select
        cat_list = [item['name'] for item in coco_api.dataset['categories']]
        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
        positive_map = create_positive_map_from_span(tokenlizer(captions), tokenspanlist)
        id_map = {item['id']: i for i, item in enumerate(coco_api.dataset['categories'])}
        max_id = max(id_map.keys())
        new_pos_map = torch.zeros((max_id + 1, 256))
        for k, v in id_map.items(): new_pos_map[k] = positive_map[v]
        self.positive_map = new_pos_map
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob = (out_logits.sigmoid() @ self.positive_map.to(out_logits.device).T)
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels = topk_indexes % prob.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        return [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

def eval_main(args, model, epoch, test_categories):
    """Main evaluation function."""
    print(f"\n--- Running evaluation for epoch {epoch} on categories: {test_categories} ---")
    model.eval()
    
    temp_json_file = os.path.join(args.output_dir, "temp_eval_coco.json")
    convert_csv_to_coco(args.test_ann_file, temp_json_file, test_categories)
    
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = CocoDetection(args.test_image_dir, temp_json_file, transforms=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    tokenlizer = get_tokenlizer.get_tokenlizer("bert-base-uncased")
    postprocessor = PostProcessCocoGrounding(coco_api=dataset.coco, tokenlizer=tokenlizer)
    evaluator = CocoGroundingEvaluator(dataset.coco, iou_types=("bbox",), useCats=True)
    caption = " . ".join(test_categories) + ' .'
    
    for i, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating")):
        images = images.tensors.to(args.device)
        # During evaluation, we use the student expert that was just trained
        outputs = model(images, captions=[caption] * images.shape[0], expert_idx_to_use=args.student_expert_idx)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(images.device)
        results = postprocessor(outputs, orig_target_sizes)
        cocogrounding_res = {target["image_id"]: output for target, output in zip(targets, results)}
        evaluator.update(cocogrounding_res)
        
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    
    model.train() # Set model back to training mode
    stats = evaluator.coco_eval["bbox"].stats.tolist()
    print("Evaluation results (AP/AR):", stats)
    return stats


# ==============================================================================
# 3. MAIN TRAINING LOGIC
# ==============================================================================

def train(args):
    """Main training loop orchestrated by command-line arguments."""
    check_and_create_directory(args.output_dir)
    
    # 1. Model Loading and Setup
    # MODIFIED: Pass num_experts from the config to correctly build the model architecture
    print("Loading model configuration...")
    config = SLConfig.fromfile(args.model_config)
    
    with open(args.experts_config, 'r') as f:
        experts_data = json.load(f)
    config.model.transformer.num_experts = experts_data['total_experts']
    
    print(f"Building model with {config.model.transformer.num_experts} experts...")
    model = build_model(config)
    
    print(f"Loading weights from: {args.base_weights}")
    checkpoint = torch.load(args.base_weights, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    
    # 2. Parameter Freezing and Unfreezing
    print("Setting up model parameters for training...")
    for param in model.parameters(): param.requires_grad = False
    
    params_to_update = get_expert_params(model, args.student_expert_idx)
    if not params_to_update:
        print(f"Error: No parameters found for student expert {args.student_expert_idx}. Aborting.")
        return
    for param in params_to_update: param.requires_grad = True

    # 3. Optimizer and Data Setup
    optimizer = optim.Adam(params_to_update, lr=args.lr)
    model.to(args.device)
    model.train()
    
    print("Reading dataset...")
    ann_Dict = read_dataset(args.train_ann_file, args.train_image_dir)
    test_categories = [cat.strip() for cat in args.test_categories.split(',')]
    
    # 4. Training Loop
    best_metric = -1.0
    best_model_state = None
    all_results_csv = os.path.join(args.output_dir, 'all_epoch_results.csv')

    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(enumerate(ann_Dict.items()), desc=f'Epoch {epoch+1}/{args.epochs}', total=len(ann_Dict))
        
        for idx, (image_path, vals) in pbar:
            image_source, image_tensor = load_image(image_path)
            
            optimizer.zero_grad()
            loss, loss_dict_log = train_image_step(
                model=model, image=image_tensor, caption_objects=vals['captions'], box_target=vals['boxes'],
                task_type=args.task_type, teacher_expert_idx=args.teacher_expert_idx, student_expert_idx=args.student_expert_idx
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 5. Evaluation after each epoch
        eval_stats = eval_main(args, model, epoch + 1, test_categories)
        
        # Save results to CSV
        file_exists = os.path.isfile(all_results_csv)
        with open(all_results_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Epoch", "AP@.50:.95", "AP@.50", "AP@.75", "AP(S)", "AP(M)", "AP(L)", 
                                 "AR@1", "AR@10", "AR@100", "AR(S)", "AR(M)", "AR(L)"])
            writer.writerow([epoch + 1] + eval_stats)

        current_metric = eval_stats[1] # Use AP@.50 as the primary metric for saving best model
        if current_metric > best_metric:
            best_metric = current_metric
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"** New best model found at epoch {epoch+1} with AP@.50: {best_metric:.4f}. Saving state. **")
            best_weight_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({'model': best_model_state}, best_weight_path)

    # 6. Save the final model state
    print("\nTraining finished.")
    if best_model_state:
        print(f"Final saved model is the best performing one (AP@.50: {best_metric:.4f}).")
    else:
        print("No improvement over initial state was observed. Saving the model from the last epoch.")
        final_weight_path = os.path.join(args.output_dir, "last_epoch_model.pth")
        torch.save({'model': model.state_dict()}, final_weight_path)


# ==============================================================================
# 4. SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Incre-MoE Training Script for GroundingDINO")
    
    # Task Definition Arguments
    parser.add_argument('--task_type', type=str, required=True, choices=['domain_incremental', 'class_incremental'], help="Type of incremental task.")
    parser.add_argument('--student_expert_idx', type=int, required=True, help="Index of the new/student expert module to be trained.")
    parser.add_argument('--teacher_expert_idx', type=int, default=-1, help="Index of the teacher expert. Required for 'class_incremental' task.")
    
    # Path Arguments
    parser.add_argument('--model_config', type=str, required=True, help="Path to the model config file (.py).")
    parser.add_argument('--base_weights', type=str, required=True, help="Path to the starting model weights (.pth) containing all experts.")
    parser.add_config_file = parser.add_argument('--experts_config', type=str, required=True, help="Path to the JSON file describing the current experts.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save weights and logs.")
    parser.add_argument('--train_ann_file', type=str, required=True, help="Path to training annotations CSV file.")
    parser.add_argument('--train_image_dir', type=str, required=True, help="Path to training images directory.")
    parser.add_argument('--test_ann_file', type=str, required=True, help="Path to test annotations CSV file.")
    parser.add_argument('--test_image_dir', type=str, required=True, help="Path to test images directory.")
    
    # Evaluation Arguments
    parser.add_argument('--test_categories', type=str, required=True, help="Comma-separated list of category names for evaluation (e.g., 'car,person').")

    # Hyperparameter Arguments
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate.")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use for training (e.g., 'cuda', 'cpu').")

    args = parser.parse_args()
    
    # Argument Validation
    if args.task_type == 'class_incremental' and args.teacher_expert_idx == -1:
        parser.error("--teacher_expert_idx is required for 'class_incremental' task type.")
    if not torch.cuda.is_available() and args.device == 'cuda':
        print("Warning: CUDA is not available. Switching to CPU.")
        args.device = 'cpu'

    print("Starting training with the following configuration:")
    print(json.dumps(vars(args), indent=2))
    
    train(args)