# Semantic Segmentation using DINOv2

## Overview

This project implements a semantic segmentation pipeline using a pretrained DINOv2 Vision Transformer backbone with a custom segmentation head.

* DINOv2 as feature extractor
* ConvNeXt-style segmentation head
* Training, evaluation, and visualization pipeline

---

## Project Structure

```
train_segmentation.py        # Base training
train_segmentation_v2.py     # Improved training
test_segmentation.py         # Evaluation & prediction
visualize.py                 # Mask visualization
segmentation_head.pth        # Trained model
predictions/                 # Outputs
```

---

## Model Architecture

* Backbone: DINOv2 (Vision Transformer)
* Head: ConvNeXt-style segmentation head
* Output: Pixel-wise classification (10 classes)

---

## Classes

| ID | Class          |
| -- | -------------- |
| 0  | Background     |
| 1  | Trees          |
| 2  | Lush Bushes    |
| 3  | Dry Grass      |
| 4  | Dry Bushes     |
| 5  | Ground Clutter |
| 6  | Logs           |
| 7  | Rocks          |
| 8  | Landscape      |
| 9  | Sky            |

---

## Features

### Training

* CrossEntropy Loss
* IoU, Dice Score, Pixel Accuracy

### Improved Training (v2)

* Data augmentation
* Deeper segmentation head
* Combined Loss (CrossEntropy + Dice)
* Partial backbone fine-tuning
* Learning rate scheduling

### Evaluation

* Mean IoU
* Per-class IoU
* Dice Score
* Pixel Accuracy

### Visualization

* Colorized segmentation masks
* Prediction comparisons

---

## Installation

```
pip install torch torchvision matplotlib opencv-python tqdm pillow
```

---

## Training

```
python train_segmentation.py
```

### Better version:

```
python train_segmentation_v2.py
```

---

## Testing

```
python test_segmentation.py \
  --model_path segmentation_head.pth \
  --data_dir path_to_dataset \
  --output_dir predictions
```

---

## Output

* masks/ → raw predictions
* masks_color/ → colored masks
* comparisons/ → visual comparisons
* evaluation_metrics.txt
* per_class_metrics.png

---

## Key Learnings

* Transformer-based feature extraction
* Semantic segmentation pipeline
* Evaluation metrics (IoU, Dice, Accuracy)
* Model optimization techniques

---

## Future Improvements

* Larger backbone (ViT-L)
* Attention-based decoder
* Real-time inference
* Web deployment
  
---

## Vision

To develop an efficient and accurate semantic segmentation system using transformer-based models, and explore its applications in real-world computer vision tasks like scene understanding and autonomous systems.

