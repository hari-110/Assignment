"""
fcn_resnet_finetuning.py
--------------------------------------------------
Fine-tune FCN-ResNet50 on BDD100K segmentation .
- Albumentations augmentations for small objects
- Class-weighted CrossEntropyLoss
"""

import os
import glob
import time
import math
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Dataset: BDD100K segmentation
class BDD100KSegDataset(Dataset):
    """
    Dataset expects:
      - img_dir: folder containing images (.jpg/.png)
      - mask_dir: folder containing mask pngs (pixel values = class ids, ignore label=255)
      - transform: albumentations.Compose with `image` and `mask` keys (or None)
    Returns:
      image_tensor: CxHxW float32 normalized for torchvision models (ToTensorV2 + Normalize)
      mask_tensor: HxW long (class ids, 255 for ignore)
    """
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # collect images
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        img_paths = []
        for e in exts:
            img_paths.extend(glob.glob(os.path.join(img_dir, e)))
        img_paths.sort()

        # pair with masks 
        self.samples = []
        for ip in img_paths:
            stem = os.path.splitext(os.path.basename(ip))[0]
            matches = glob.glob(os.path.join(mask_dir, stem + "*.png")) + \
                      glob.glob(os.path.join(mask_dir, stem + "*.PNG"))
            if matches:
                self.samples.append((ip, matches[0]))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No (image, mask) pairs found.\n"
                f"Checked IMG_DIR={img_dir}, MASK_DIR={mask_dir}.\n"
                f"Ensure mask filenames share the same stem as images."
            )


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]   
            mask = augmented["mask"]
        else:
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(mask).long()

        # Ensure mask is long type
        if isinstance(mask, torch.Tensor):
            mask = mask.long()
        else:
            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask


# Compute class weights
def compute_class_weights(mask_dir, num_classes, sample_limit=None):
    """
    method: "inv" uses 1 / freq
    sample_limit: optional number of masks to sample 
    Returns: torch.tensor(weights) size (num_classes,)
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    mask_paths = glob.glob(os.path.join(mask_dir, "*.png"))
    mask_paths.sort()
    if sample_limit:
        mask_paths = mask_paths[:sample_limit]

    for mp in mask_paths:
        m = np.array(Image.open(mp))
        m = m[m != 255]  # ignore index
        if m.size == 0:
            continue
        binc = np.bincount(m.ravel(), minlength=num_classes)
        counts[:len(binc)] += binc

    counts = np.maximum(counts, 1.0)  # avoid zero
    freq = counts / counts.sum()
    weights = 1.0 / (freq + 1e-6)

    # normalize so sum(weights) == num_classes (keeps range stable)
    weights = weights / (weights.sum() / float(num_classes))
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights


# metrics
@torch.no_grad()
def update_confusion_matrix(conf_matrix: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """
    preds: (B,H,W) long
    targets: (B,H,W) long
    Ignores target==255 (ignore_index)
    """
    if preds.dim() == 2:
        preds = preds.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)

    # flatten valid pixels (CPU)
    valid_mask = (targets >= 0) & (targets < num_classes)
    if valid_mask.sum() == 0:
        return

    t = targets[valid_mask].view(-1).to(torch.int64).cpu()
    p = preds[valid_mask].view(-1).to(torch.int64).cpu()
    k = (t * num_classes + p).to(torch.int64)
    bincount = torch.bincount(k, minlength=num_classes * num_classes)
    conf_matrix += bincount.view(num_classes, num_classes)


def metrics_from_confusion(conf_matrix: torch.Tensor):
    """
    Returns: mIoU, pixel_acc, precision_mean, recall_mean, f1 (scalar)
    """
    conf = conf_matrix.double()
    TP = torch.diag(conf)
    FP = conf.sum(0) - TP
    FN = conf.sum(1) - TP
    denom_iou = (TP + FP + FN).clamp(min=1e-6)
    iou_per_class = TP / denom_iou
    mIoU = torch.nanmean(iou_per_class).item()

    pixel_acc = (TP.sum() / conf.sum().clamp(min=1e-6)).item()
    precision = (TP / (TP + FP).clamp(min=1e-6)).mean().item()
    recall = (TP / (TP + FN).clamp(min=1e-6)).mean().item()
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)
    return mIoU, pixel_acc, precision, recall, f1, iou_per_class.cpu().numpy()


# Validation (streaming) function
@torch.no_grad()
def validate(model, dataloader, device, num_classes):
    model.eval()
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    n_images = 0
    t_start = time.time()

    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        out = model(images)
        logits = out['out'] if isinstance(out, dict) and 'out' in out else out
        preds = torch.argmax(logits, dim=1)
        update_confusion_matrix(conf_matrix, preds, masks, num_classes)
        n_images += images.size(0)

        # free
        del images, masks, logits, preds, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - t_start
    fps = n_images / total_time if total_time > 0 else 0.0
    mIoU, pix_acc, prec, rec, f1, iou_per_class = metrics_from_confusion(conf_matrix)
    return {
        "mIoU": mIoU,
        "PixelAcc": pix_acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "per_class_iou": iou_per_class,
        "FPS": fps
    }

@torch.no_grad()
def compute_val_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        out_main = outputs['out'] if isinstance(outputs, dict) else outputs
        loss = criterion(out_main, masks)
        total_loss += loss.item()
        total_batches += 1
    return total_loss / max(1, total_batches)


# Training loop
def train(
    img_dir_train,
    mask_dir_train,
    img_dir_val,
    mask_dir_val,
    num_classes=19,
    in_channels=3,
    batch_size=4,
    num_workers=4,
    lr=3e-4,
    num_epochs=8,
    aux_loss_weight=0.4,
    device=None,
    out_dir="outputs",
    sample_limit_for_weights=5000,
):
    os.makedirs(out_dir, exist_ok=True)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print("Device:", device)

        # Augmentations
    train_transform = A.Compose([
        # A.RandomScale(scale_limit=(0.5, 1.5), p=0.6),
        A.RandomCrop(height=512, width=1024, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.2),
        # A.CoarseDropout(max_holes=1, max_height=64, max_width=64, p=0.3),
        A.GaussianBlur(p=0.15),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], additional_targets={})

    val_transform = A.Compose([
        # A.PadIfNeeded(min_height=512, min_width=1024, border_mode=0, value=0, mask_value=255),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Datasets & loaders
    train_ds = BDD100KSegDataset(img_dir_train, mask_dir_train, transform=train_transform)
    val_ds = BDD100KSegDataset(img_dir_val, mask_dir_val, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, batch_size//2), shuffle=False,
                            num_workers=max(0, num_workers//2), pin_memory=True, drop_last=False)

    # Compute class weights (fast sample)
    print("Computing class weights ", sample_limit_for_weights, "masks")
    class_weights = compute_class_weights(mask_dir_train, num_classes, sample_limit=sample_limit_for_weights)
    print("Class weights:", class_weights.cpu().numpy())

    # Model: load pretrained FCN-ResNet50 and adapt heads
    model = torchvision.models.segmentation.fcn_resnet50(pretrained=True, pretrained_backbone=True)
    # replace final conv layers to match num_classes
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        # replace final 1x1 conv
        last = list(model.classifier.children())[-1]
        if isinstance(last, nn.Conv2d):
            in_ch = last.in_channels
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1],
                                             nn.Conv2d(in_ch, num_classes, kernel_size=1))
        else:
           # just set classifier[4]
            model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    else:
        model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    # aux classifier
    if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
        last_aux = list(model.aux_classifier.children())[-1]
        if isinstance(last_aux, nn.Conv2d):
            in_ch = last_aux.in_channels
            model.aux_classifier = nn.Sequential(*list(model.aux_classifier.children())[:-1],
                                                 nn.Conv2d(in_ch, num_classes, kernel_size=1))
        else:
            model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    model.to(device)

    # weighted CE and optimizer / scheduler / scaler
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), ignore_index=255)
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler: warmup via lambda + cosine (simple)
    def lr_lambda(epoch):
        # linear warmup 1 epoch then cosine
        if epoch < 1:
            return 0.1
        else:
            progress = (epoch - 1) / max(1, num_epochs - 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Training loop
    best_miou = -1.0
    best_ckpt = os.path.join(out_dir, "best_fcn_resnet50.pth")
    log_every = 10

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()

        #freeze the backbone permanently
        for param in model.backbone.parameters():
            param.requires_grad = False

        running_loss = 0.0
        iters = 0
        print(f"Epoch : {epoch}")
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                out_main = outputs['out'] if isinstance(outputs, dict) and 'out' in outputs else outputs
                loss_main = criterion(out_main, masks)
                loss = loss_main
                # aux
                if isinstance(outputs, dict) and 'aux' in outputs and outputs['aux'] is not None:
                    loss_aux = criterion(outputs['aux'], masks)
                    loss = loss + aux_loss_weight * loss_aux

            scaler.scale(loss).backward()
            # gradient clipping (optional)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            iters += 1

            if (batch_idx + 1) % log_every == 0:
                avg_loss = running_loss / max(1, iters)

            # free
            del images, masks, outputs, out_main
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        epoch_time = time.time() - epoch_start
        avg_train_loss = running_loss / max(1, iters)
        print(f"Epoch {epoch+1} finished. Avg train loss: {avg_train_loss:.4f}. Time: {epoch_time:.1f}s")

        # scheduler step after epoch
        scheduler.step()

        best_val_loss = float("inf")

        val_loss = compute_val_loss(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss
            }, best_ckpt)
            print("Saved best checkpoint (by val loss):", best_ckpt)

    # final evaluation load best
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        final_stats = validate(model, val_loader, device, num_classes)
        print("Final evaluation (best model):", final_stats)
    else:
        print("No checkpoint found; final evaluation skipped.")

    return model, best_ckpt


if __name__ == "__main__":

    train(
        img_dir_train="~/bdd100k_seg/bdd100k/seg/images/train",
        mask_dir_train="~/bdd100k_seg/bdd100k/seg/labels/train",
        img_dir_val="~/bdd100k_seg/bdd100k/seg/images/val",
        mask_dir_val="~/bdd100k_seg/bdd100k/seg/labels/val",
        num_classes=19,
        batch_size=8,
        num_workers=0,
        lr=0.0001,
        num_epochs=10,
        out_dir="outputs",
        aux_loss_weight=0.4,
        sample_limit_for_weights=2000  
    )
