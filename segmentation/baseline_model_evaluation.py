import os
import glob
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn.functional as F
from tabulate import tabulate

# Dataset Loader (with resize + normalization)
class BDD100KSegDataset(Dataset):
    """
    Loads (image, mask) pairs for segmentation.
    - Finds masks by matching image stem prefix.
    - Optionally resizes to size_wh (W, H).
    - Normalizes images with ImageNet stats for torchvision pretrained models.
    """
    def __init__(self, img_dir: str, mask_dir: str, size_wh: tuple[int, int] | None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size_wh = size_wh

        # collect images
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        img_paths = []
        for e in exts:
            img_paths.extend(glob.glob(os.path.join(img_dir, e)))
        img_paths.sort()

        # pair with masks (case-insensitive, allow suffixes)
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

        # transforms
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def __len__(self):
        return len(self.samples)

    def _resize_pair(self, img_t: torch.Tensor, mask_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Resize image & mask to size_wh (W,H)."""
        if self.size_wh is None:
            return img_t, mask_t
        W, H = self.size_wh
        # image: bilinear
        img_t = F.interpolate(img_t.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
        # mask: nearest
        mask_t = F.interpolate(mask_t.unsqueeze(0).unsqueeze(0).float(), size=(H, W), mode='nearest').squeeze(0).squeeze(0).long()
        return img_t, mask_t

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        img_t = torchvision.transforms.functional.to_tensor(img)  # float in [0,1], CxHxW
        mask_np = np.array(mask, dtype=np.int64)
        mask_t = torch.from_numpy(mask_np)  # HxW, class ids (255 for ignore)

        # resize if requested
        img_t, mask_t = self._resize_pair(img_t, mask_t)

        # normalize for torchvision pretrained backbones
        img_t = (img_t - self.mean) / self.std

        return img_t, mask_t


# Metrics (streaming via confusion matrix)
@torch.no_grad()
def update_confusion_matrix(conf_mat: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """
    Update confusion matrix in place.
    preds, targets: (N, H, W) or (H, W) tensors with class ids.
    Ignores label 255.
    """
    if preds.dim() == 2:
        preds = preds.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)

    # mask ignore
    valid = (targets >= 0) & (targets < num_classes)
    # flatten valid pixels
    t = targets[valid]
    p = preds[valid]
    k = (t * num_classes + p).to(torch.int64)
    binc = torch.bincount(k.cpu(), minlength=num_classes * num_classes)
    conf_mat += binc.view(num_classes, num_classes)


def metrics_from_confusion_matrix(conf_mat: torch.Tensor):
    """
    Compute mIoU, pixel acc, precision, recall, F1 from confusion matrix.
    """
    conf = conf_mat.double()
    TP = torch.diag(conf)
    FP = conf.sum(0) - TP
    FN = conf.sum(1) - TP
    denom_iou = (TP + FP + FN).clamp(min=1e-6)
    iou_per_class = TP / denom_iou
    mIoU = iou_per_class.mean().item()

    pixel_acc = (TP.sum() / conf.sum().clamp(min=1e-6)).item()
    precision = (TP / (TP + FP).clamp(min=1e-6)).mean().item()
    recall    = (TP / (TP + FN).clamp(min=1e-6)).mean().item()
    f1        = (2 * precision * recall) / (precision + recall + 1e-6)

    return mIoU, pixel_acc, precision, recall, f1



# Evaluation
@torch.no_grad()
def evaluate_model(model, dataloader, device, num_classes):
    """
    Returns (mIoU, pixel_acc, precision, recall, f1, fps)
    """
    model.eval()
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    n_images = 0
    start = time.time()

    for imgs, masks in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # forward
        out = model(imgs)
        if isinstance(out, dict) and 'out' in out:
            logits = out['out']
        else:
            logits = out  

        preds = torch.argmax(logits, dim=1)  # (B,H,W)

        update_confusion_matrix(conf_mat, preds, masks, num_classes)
        n_images += imgs.size(0)

        # free
        del imgs, masks, logits, preds, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - start
    fps = n_images / total_time if total_time > 0 else 0.0

    mIoU, pix_acc, prec, rec, f1 = metrics_from_confusion_matrix(conf_mat)
    return mIoU, pix_acc, prec, rec, f1, fps


# 
# Main
# 
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    #  Paths & config 
    IMG_DIR = "~/images_seg/bdd100k_seg/bdd100k/seg/images/val"
    MASK_DIR = "~/images_seg/bdd100k_seg/bdd100k/seg/labels/val"
    NUM_CLASSES = 19                 # BDD100K classes, ignore=255 in your masks
    INFER_SIZE = (224, 224)          # keep it small to avoid CPU RAM spikes
    BATCH_SIZE = 1                   # small batch is safer for RAM
    NUM_WORKERS = 0                  
    PIN_MEMORY = False               # keep False to avoid pinning large tensors

    #  Dataset & loader 
    dataset = BDD100KSegDataset(IMG_DIR, MASK_DIR, INFER_SIZE)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    results = []

    #  Model 1: DeepLabV3-MobileNetV3 
    dl = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True).to(device)
    mIoU, pix_acc, prec, rec, f1, fps = evaluate_model(dl, dataloader, device, NUM_CLASSES)
    results.append(["DeepLabV3-MobV3", mIoU, pix_acc, prec, rec, f1, fps])
    # free model weights before loading the next
    del dl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

     #  Model 2: DeepLabV3-ResNet50 
    dl_resnet = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True).to(device)
    mIoU, pix_acc, prec, rec, f1, fps = evaluate_model(dl_resnet, dataloader, device, NUM_CLASSES)
    results.append(["DeepLabV3-ResNet50", mIoU, pix_acc, prec, rec, f1, fps])
    del dl_resnet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    #  Model 3: FCN-ResNet50 
    fcn = torchvision.models.segmentation.fcn_resnet50(pretrained=True).to(device)
    mIoU, pix_acc, prec, rec, f1, fps = evaluate_model(fcn, dataloader, device, NUM_CLASSES)
    results.append(["FCN-ResNet50", mIoU, pix_acc, prec, rec, f1, fps])
    del fcn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    #  Print table 
    print(tabulate(
        results,
        headers=["Model", "mIoU", "Pixel Acc", "Precision", "Recall", "F1", "FPS"],
        floatfmt=".4f"
    ))

    #  Plots 
    labels = [r[0] for r in results]
    miou_vals = [r[1] for r in results]
    fps_vals = [r[6] for r in results]

    plt.figure()
    plt.bar(labels, miou_vals)
    plt.ylabel("mIoU")
    plt.title("Segmentation Model Comparison")
    plt.show()

    plt.figure()
    plt.bar(labels, fps_vals)
    plt.ylabel("FPS")
    plt.title("Segmentation Model Inference Speed")
    plt.show()


if __name__ == "__main__":
    main()
