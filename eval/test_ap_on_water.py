# 类别映射
category_mapping = {
    'fish': 1,
    'starfish': 2,
    'jellyfish': 3,
    'shrimp': 4,
    'crab': 5
}

import argparse
import os
import sys
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.util.slconfig import SLConfig

import torchvision

from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator

import csv
import json
import warnings
warnings.filterwarnings('ignore')

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def get_annotation_eval(csv_file, start_index, end_index, output_file, json_file):
    df = pd.read_csv(csv_file)
    
    # No filtering or changes made to the dataframe
    
    ensure_directory_exists(os.path.dirname(output_file))
    
    # Save the original data to the output file
    df.to_csv(output_file, index=False)
    
    # Convert the unmodified CSV to COCO format (assuming this function is defined elsewhere)
    convert_csv_to_coco(output_file, json_file=json_file)

def convert_csv_to_coco(csv_file, json_file):
    coco_output = {
        "info": {
            "description": "COCO Dataset",
            "url": "",
            "version": "1.0",
            "year": 2024,
            "contributor": "",
            "date_created": "2024-07-14"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "fish", "supercategory": "animal"},
            {"id": 2, "name": "starfish", "supercategory": "animal"},
            {"id": 3, "name": "jellyfish", "supercategory": "animal"},
            {"id": 4, "name": "shrimp", "supercategory": "animal"},
            {"id": 5, "name": "crab", "supercategory": "animal"}
        ]
    }



    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        annotation_id = 1
        image_id = 1
        image_set = set()
        
        for row in reader:
            if row['image_name'] not in image_set:
                image_set.add(row['image_name'])
                image = {
                    "id": image_id,
                    "file_name": row['image_name'],
                    "width": int(row['image_width']),
                    "height": int(row['image_height']),
                    "license": 1,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": ""
                }
                coco_output["images"].append(image)
                image_id += 1

            category_id = category_mapping.get(row['label_name'])
            if category_id is None:
                continue  # Skip unknown labels

            annotation = {
                "id": annotation_id,
                "image_id": image_id - 1,
                "category_id": category_id,
                "bbox": [
                    int(row['bbox_x']),
                    int(row['bbox_y']),
                    int(row['bbox_width']),
                    int(row['bbox_height'])
                ],
                "area": int(row['bbox_width']) * int(row['bbox_height']),
                "segmentation": [],
                "iscrowd": 0
            }
            coco_output["annotations"].append(annotation)
            annotation_id += 1

    with open(json_file, 'w') as output_json_file:
        json.dump(coco_output, output_json_file, indent=4)

def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    if "model" not in checkpoint:
        checkpoint = {"model": checkpoint}
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

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
    def __init__(self, num_select=300, coco_api=None, tokenlizer=None) -> None:
        super().__init__()
        self.num_select = num_select

        assert coco_api is not None
        category_dict = coco_api.dataset['categories']
        cat_list = [item['name'] for item in category_dict]
        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
        positive_map = create_positive_map_from_span(
            tokenlizer(captions), tokenspanlist)

        id_map = {i: i+1 for i in range(len(cat_list))}

        new_pos_map = torch.zeros((91, 256))
        for k, v in id_map.items():
            new_pos_map[v] = positive_map[k]
        self.positive_map = new_pos_map

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False):
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        prob_to_token = out_logits.sigmoid()
        pos_maps = self.positive_map.to(prob_to_token.device)
        prob_to_label = prob_to_token @ pos_maps.T

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels = topk_indexes % prob.shape[2]

        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        boxes = torch.gather(
            boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b}
                   for s, l, b in zip(scores, labels, boxes)]

        return results

def main(args):
    cfg = SLConfig.fromfile(args.config_file)

    model = load_model(args.config_file, args.checkpoint_path)
    model = model.to(args.device)
    model = model.eval()

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = CocoDetection(
        args.image_dir, args.anno_path, transforms=transform)
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessCocoGrounding(
        coco_api=dataset.coco, tokenlizer=tokenlizer)

    evaluator = CocoGroundingEvaluator(
        dataset.coco, iou_types=("bbox",), useCats=True)

    category_dict = dataset.coco.dataset['categories']
    cat_list = [item['name'] for item in category_dict]
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    start = time.time()
    for i, (images, targets) in enumerate(data_loader):
        images = images.tensors.to(args.device)
        bs = images.shape[0]
        input_captions = [caption] * bs

        outputs = model(images, captions=input_captions)

        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0).to(images.device)
        results = postprocessor(outputs, orig_target_sizes)
        cocogrounding_res = {
            target["image_id"]: output for target, output in zip(targets, results)}
        evaluator.update(cocogrounding_res)

        if (i+1) % 30 == 0:
            used_time = time.time() - start
            eta = len(data_loader) / (i+1e-5) * used_time - used_time
            print(
                f"processed {i}/{len(data_loader)} images. time: {used_time:.2f}s, ETA: {eta:.2f}s")

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    print("Final results:", evaluator.coco_eval["bbox"].stats.tolist())

if __name__ == "__main__":
    get_annotation_eval(
        csv_file="/home/enbo/chxm/Grounding-Dino-FineTuning/dataset/water_dataset/new_test_annotations.csv", 
        start_index=1,
        end_index=5000, 
        output_file="/home/enbo/chxm/Grounding-Dino-FineTuning/data_process/eval_csv/temp.csv",
        json_file="/home/enbo/chxm/Grounding-Dino-FineTuning/data_process/eval_csv/temp.json")
    
    parser = argparse.ArgumentParser("Grounding DINO eval on COCO", add_help=True)
    
    config_file = "/home/enbo/chxm/Grounding-Dino-FineTuning/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    checkpoint_path = "/home/enbo/chxm/Grounding-Dino-FineTuning/weights/water_4.pth"
    device = "cuda"
    num_select = 300
    anno_path = "/home/enbo/chxm/Grounding-Dino-FineTuning/data_process/eval_csv/temp.json"
    image_dir = "/home/enbo/chxm/Grounding-Dino-FineTuning/dataset/water_dataset/test"
    num_workers = 4

    parser.add_argument("--config_file", "-c", type=str, default=config_file, required=False, help="path to config file")
    parser.add_argument("--checkpoint_path", "-p", type=str, default=checkpoint_path, required=False, help="path to checkpoint file")
    parser.add_argument("--device", type=str, default=device, help="running device (default: cuda)")
    parser.add_argument("--num_select", type=int, default=num_select, help="number of topk to select")
    parser.add_argument("--anno_path", type=str, default=anno_path, required=False, help="coco root")
    parser.add_argument("--image_dir", type=str, default=image_dir, required=False, help="coco image dir")
    parser.add_argument("--num_workers", type=int, default=num_workers, help="number of workers for dataloader")
    
    args = parser.parse_args()

    main(args)
