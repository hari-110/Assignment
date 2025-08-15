"""
Multi-Model Benchmark Script for Object Detection
--------------------------------------------------
This script benchmarks multiple object detection models:
1. MobileNet-SSD (TorchVision SSDLite320_MobileNet_V3_Large)
2. Faster R-CNN (TorchVision FasterRCNN_ResNet50_FPN)
3. YOLOv8 (Ultralytics Hub - auto-download)

Outputs:
- COCO evaluation (mAP, AP50, AP50:90)
- Extra metrics (Precision, Recall, F1-score, IoU, FPS)
- Comparison table
"""

import os
import time
import json
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
from tabulate import tabulate


# Dataset Loader 
class COCODataset(Dataset):
    """
    Dataset loader for COCO or BDD100K JSON annotations.
    Converts BDD100K format to COCO if needed.
    """
    def __init__(self, img_dir, ann_file):
        self.img_dir = img_dir

        # Load annotation JSON
        with open(ann_file, 'r') as f:
            data = json.load(f)

        # Convert BDD100K list format → COCO dict format 
        if isinstance(data, list):
            data = self._bdd_to_coco(data)

        # Create COCO object 
        self.coco = COCO()
        self.coco.dataset = data
        self.coco.createIndex()

        self.image_ids = list(self.coco.imgs.keys())

    def _bdd_to_coco(self, bdd_data):
        """
        Converts BDD100K labels to COCO-style JSON dict.
        """
        images, annotations, categories = [], [], {}
        ann_id = 1
        for img_id, item in enumerate(bdd_data):
            file_name = item['name']
            width, height = 1280, 720  
            images.append({
                "id": img_id,
                "file_name": file_name,
                "width": width,
                "height": height
            })
            for label in item.get('labels', []):
                if 'box2d' not in label:
                    continue
                cat_name = label['category']
                if cat_name not in categories:
                    categories[cat_name] = len(categories) + 1
                x1, y1 = label['box2d']['x1'], label['box2d']['y1']
                x2, y2 = label['box2d']['x2'], label['box2d']['y2']
                w, h = x2 - x1, y2 - y1
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": categories[cat_name],
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1
        return {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": v, "name": k} for k, v in categories.items()]
        }

    def __getitem__(self, idx):
        """
        Returns:
        - img_path: path to image file
        - img_id: COCO image ID
        """
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        return img_path, img_id

    def __len__(self):
        return len(self.image_ids)
    

def evaluate_predictions(coco_gt, predictions):
    """
    Evaluates predictions using COCO API and computes extra metrics:
    - mAP@0.5
    - mAP@[0.5:0.9]
    - Precision, Recall, F1-score
    - Mean IoU
    """
    # Safeguard for COCO API
    if 'info' not in coco_gt.dataset:
        coco_gt.dataset['info'] = {}
    if 'licenses' not in coco_gt.dataset:
        coco_gt.dataset['licenses'] = []

    # Load results into COCO
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.iouThrs = np.linspace(0.5, 0.9, int((0.9 - 0.5) / 0.05) + 1)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract mAPs
    mAP_50 = coco_eval.stats[1]
    mAP_50_90 = coco_eval.stats[0]

    # Extra metrics
    tp = fp = fn = 0
    total_iou = []

    img_ids = coco_gt.getImgIds()
    for img_id in img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        gt_anns = coco_gt.loadAnns(ann_ids)
        gt_boxes = np.array([ann['bbox'] for ann in gt_anns])

        pred_boxes = np.array([p['bbox'] for p in predictions if p['image_id'] == img_id])
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue

        for gt in gt_boxes:
            if len(pred_boxes) == 0:
                fn += 1
                continue
            ixmin = np.maximum(gt[0], pred_boxes[:, 0])
            iymin = np.maximum(gt[1], pred_boxes[:, 1])
            ixmax = np.minimum(gt[0] + gt[2], pred_boxes[:, 0] + pred_boxes[:, 2])
            iymax = np.minimum(gt[1] + gt[3], pred_boxes[:, 1] + pred_boxes[:, 3])
            iw = np.maximum(ixmax - ixmin, 0)
            ih = np.maximum(iymax - iymin, 0)
            inters = iw * ih
            uni = gt[2] * gt[3] + (pred_boxes[:, 2] * pred_boxes[:, 3]) - inters
            iou = inters / (uni + 1e-6)
            max_iou = np.max(iou)
            total_iou.append(max_iou)
            if max_iou > 0.5:
                tp += 1
            else:
                fn += 1
        fp += max(0, len(pred_boxes) - tp)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    mean_iou = np.mean(total_iou) if total_iou else 0

    return mAP_50, mAP_50_90, precision, recall, f1, mean_iou



# Model Runner Wrappers
def run_torchvision_model(model, dataloader, device):
    """
    Runs TorchVision detection models (MobileNet-SSD, Faster R-CNN)
    """
    model.eval() 
    predictions = []
    start_time = time.time()

    with torch.no_grad():
        for img_paths, img_ids in dataloader:
            images = [torchvision.io.read_image(p).float() / 255.0 for p in img_paths]
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                img_id = int(img_ids[i])
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    predictions.append({
                        "image_id": img_id,
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score)
                    })

    total_time = time.time() - start_time
    fps = len(dataloader.dataset) / total_time
    return predictions, fps


def run_yolov8_model(model_name, dataloader, device):
    """
    Runs YOLOv8 model from Ultralytics Hub
    """
    model = YOLO(model_name + ".pt")
    predictions = []
    start_time = time.time()

    for img_paths, img_ids in dataloader:
        preds = model.predict(img_paths, device=device, verbose=False)
        for i, pred in enumerate(preds):
            img_id = int(img_ids[i])
            if pred.boxes is None:
                continue
            boxes = pred.boxes.xyxy.cpu().numpy()
            scores = pred.boxes.conf.cpu().numpy()
            labels = pred.boxes.cls.cpu().numpy().astype(int) + 1
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                predictions.append({
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score)
                })

    total_time = time.time() - start_time
    fps = len(dataloader.dataset) / total_time
    return predictions, fps


# Main Benchmark Runner
def run_benchmark(img_dir, ann_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = COCODataset(img_dir, ann_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    results_table = []

    # MobileNet-SSD
    mobilenet_model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True).to(device)
    preds, fps = run_torchvision_model(mobilenet_model, dataloader, device)
    m50, m5090, prec, rec, f1, miou = evaluate_predictions(dataset.coco, preds)
    results_table.append(["MobileNet-SSD", m50, m5090, prec, rec, f1, miou, fps])

    # Faster R-CNN
    frcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    preds, fps = run_torchvision_model(frcnn_model, dataloader, device)
    m50, m5090, prec, rec, f1, miou = evaluate_predictions(dataset.coco, preds)
    results_table.append(["Faster R-CNN", m50, m5090, prec, rec, f1, miou, fps])

    # YOLOv8
    preds, fps = run_yolov8_model("yolov8s", dataloader, device)
    m50, m5090, prec, rec, f1, miou = evaluate_predictions(dataset.coco, preds)
    results_table.append(["YOLOv8s", m50, m5090, prec, rec, f1, miou, fps])


    print(tabulate(
    results_table,
    headers=["Model", "mAP@0.5", "mAP@[0.5:0.9]", "Precision", "Recall", "F1", "IoU", "FPS"],
    floatfmt=".4f"
))


if __name__ == "__main__":
    IMG_DIR = "~/bdd100k_images_100k/bdd100k/images/100k/val_2k"
    ANN_FILE = "~/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val_2k.json"
    run_benchmark(IMG_DIR, ANN_FILE)
