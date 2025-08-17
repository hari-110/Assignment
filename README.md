# BDD100K Object Detection & Segmentation Benchmark

This repository benchmarks and fine-tune the good model for **object detection** and **semantic segmentation** on the BDD100K dataset.

---

## Repository Structure

- `data_exploration.ipynb`  
  Exploratory Data Analysis (EDA) for BDD100K:
  - Weather distribution  
  - Time-of-day distribution  
  - Object size distribution (bbox area/aspect ratio)  
  - Occlusion and truncation statistics  
  - Scene attributes and class imbalance analysis  
  - Weather vs Time-of-day distribution 

- `detection/`  
  - **Baseline scripts**: MobileNet-SSD, Faster R-CNN, YOLOv8  
  - **Multi-model benchmark**: Evaluates mAP@0.5, mAP@[0.5:0.95], Precision, Recall, F1, IoU, FPS  
  - **Fine-tuning**: Faster R-CNN with Albumentations augmentations + class-weighted loss

- `segmentation/`  
  - **Baseline scripts**: DeepLabV3-MobileNetV3, DeepLabV3-ResNet50, FCN-ResNet50  
  - **Multi-model benchmark**: Evaluates mIoU, Pixel Accuracy, Precision, Recall, F1  
  - **Fine-tuning**: FCN-ResNet50 with small-object-aware augmentations and class imbalance handling

- `requirements.txt`  
  Lists dependencies  

---

## Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run EDA
```bash
jupyter notebook data_exploration.ipynb
```

### 3. Object Detection
Edit dataset paths inside `detection/multi_model_benchmark.py`, then run:
```bash
python detection/multi_model_benchmark.py
```
For fine-tuning Faster R-CNN:
```bash
python detection/faster_rcnn_finetune.py
```

### 4. Semantic Segmentation
Edit dataset paths inside `segmentation/segmentation_benchmark.py`, then run:
```bash
python segmentation/segmentation_benchmark.py
```
For fine-tuning FCN-ResNet50:
```bash
python segmentation/fcn_resnet_finetuning.py
```

---

## Highlights
### 1. EDA Findings
    - Severe class imbalance
	- Majority of objects are large (YOLO less suitable, SSD/Faster R-CNN better)
	- Small objects require augmentation and special handling
### 2. Detection Benchmarks
	- Compared MobileNet-SSD, Faster R-CNN, YOLOv8
	- Faster R-CNN selected for fine-tuning baseline
### 3. Segmentation Benchmarks
	- Compared DeepLabV3-MobileNetV3, DeepLabV3-Resnet50, FCN-ResNet50
	- FCN-ResNet50 chosen for fine-tuning baseline
### 4. Techniques Used
	- Albumentations augmentations
	- Inverse-frequency class weighting for imbalance
	- Final metrics: mAP, IoU, Pixel Accuracy, F1, FPS
	
## Reference
	- [BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning](https://arxiv.org/pdf/1805.04687)
	- https://bair.berkeley.edu/blog/2018/05/30/bdd/
	

	

