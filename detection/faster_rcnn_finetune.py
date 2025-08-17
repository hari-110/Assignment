"""
faster_rcnn_finetune.py
--------------------------------------------------
- Fine-tune Faster-RCNN on BDD100K dataset.
- Albumentations augmentations for small objects
- Calculate Inverse frequency class weights 
- Save the best model for final metrics evaluations

"""

import os
import time
import json
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter
from torch.nn import CrossEntropyLoss



# Dataset Loader
class COCODataset(Dataset):
    """
    Dataset loader for COCO or BDD100K JSON annotations.
    Converts BDD100K format to COCO.
    """
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        with open(ann_file, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            data = self._bdd_to_coco(data)

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
        image = Image.open(img_path).convert("RGB")
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue  
            boxes.append([x, y, x + w, y + h])  # convert to [xmin, ymin, xmax, ymax]
            labels.append(ann['category_id'])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}

        if self.transform:
            transformed = self.transform(image=np.array(image))
            image = transformed['image']

        return image, target
    

    def __len__(self):
        return len(self.image_ids)


# Validation Loss Function
def compute_validation_loss(model, dataloader, device):
    model.train()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            total_loss += loss.item()
    return total_loss / len(dataloader)


# Evaluation Metrics
def evaluate_model(model, dataloader, coco_gt, device):
    """
    Evaluates predictions using COCO API and computes extra metrics:
    - mAP@0.5
    - mAP@[0.5:0.9]
    - Precision, Recall, F1-score
    - IoU
    """
    model.eval()
    results = []
    total_time = 0
    total_iou = []
    tp = fp = fn = 0

    if 'info' not in coco_gt.dataset:
        coco_gt.dataset['info'] = {}
    if 'licenses' not in coco_gt.dataset:
        coco_gt.dataset['licenses'] = []

    with torch.no_grad():
        for images, targets in dataloader:
            img_ids = [int(t["image_id"].item()) for t in targets]
            images = [img.to(device) for img in images]

            start = time.time()
            outputs = model(images)
            end = time.time()
            total_time += (end - start)

            for i, output in enumerate(outputs):
                img_id = img_ids[i]
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()

                ann_dicts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id]))
                gt_boxes = np.array([ann['bbox'] for ann in ann_dicts])

                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    results.append({
                        "image_id": img_id,
                        "category_id": int(label)+1,
                        "bbox": [float(x1), float(y1), float(w), float(h)],
                        "score": float(score)
                    })

                for gt in gt_boxes:
                    if len(pred_boxes) == 0:
                        fn += 1
                        continue
                    ixmin = np.maximum(gt[0], pred_boxes[:, 0])
                    iymin = np.maximum(gt[1], pred_boxes[:, 1])
                    ixmax = np.minimum(gt[0] + gt[2], pred_boxes[:, 2])
                    iymax = np.minimum(gt[1] + gt[3], pred_boxes[:, 3])
                    iw = np.maximum(ixmax - ixmin, 0)
                    ih = np.maximum(iymax - iymin, 0)
                    inters = iw * ih
                    uni = gt[2] * gt[3] + (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1]) - inters
                    iou = inters / (uni + 1e-6)
                    max_iou = np.max(iou)
                    total_iou.append(max_iou)
                    if max_iou > 0.5:
                        tp += 1
                    else:
                        fn += 1
                fp += max(0, len(pred_boxes) - tp)

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.iouThrs = np.linspace(0.5, 0.9, 9)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP_50 = coco_eval.stats[1]
    mAP_50_90 = coco_eval.stats[0]
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    mean_iou = np.mean(total_iou) if total_iou else 0
    fps = len(dataloader.dataset) / total_time if total_time > 0 else 0

    return {
        "mAP@0.5": mAP_50,
        "mAP@[0.5:0.9]": mAP_50_90,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "IoU": mean_iou,
        "FPS": fps
    }


# Training Loop
def train_and_finetune(img_dir_train, ann_file_train, img_dir_val, ann_file_val, num_epochs=5, lr=0.005):
    """
    Apply Augmentations and Normalization
    Calculate Inverse frequency class weights 
    Save the best model for final metrics evaluations
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Albumentations transforms
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.MotionBlur(p=0.2),
        A.ColorJitter(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_dataset = COCODataset(img_dir_train, ann_file_train, transform=train_transform)
    val_dataset   = COCODataset(img_dir_val, ann_file_val, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = len(train_dataset.coco.cats) + 1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # Inverse frequency class weights
    label_counts = Counter()
    for _, tgt in train_dataset:
        label_counts.update(tgt["labels"].tolist())
    freqs = np.array([label_counts.get(c, 1) for c in range(num_classes)])
    class_weights = torch.tensor(1.0 / freqs, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)
    model.roi_heads.box_predictor.loss_cls = CrossEntropyLoss(weight=class_weights)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0005)

    best_val_loss = float("inf")
    best_model_path = "best_faster_rcnn_model.pth"

    for epoch in range(num_epochs):
        print(f"Epoch : {epoch}")
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            train_loss += losses.item()

        avg_train_loss = train_loss / len(train_loader)
        val_loss = compute_validation_loss(model, val_loader, device)
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved.")

    # Full Evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    metrics = evaluate_model(model, val_loader, val_dataset.coco, device)
    print("\n Final Metrics ")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    train_and_finetune(
        img_dir_train="~/bdd100k_images_100k/bdd100k/images/100k/train_40k",
        ann_file_train="~/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train_40k.json",
        img_dir_val="~/bdd100k_images_100k/bdd100k/images/100k/val_8k",
        ann_file_val="~/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val_8k.json",
        num_epochs=10,
        lr=0.0001
    )

 
