# Few-shot Teacher-Student Segmentation for Deployable Piglet Litter Average Weight Estimation in Farrowing Pens

This repository contains project scripts for the paper:

**Few-shot teacher-student segmentation for deployable piglet litter average weight estimation in farrowing pens**

Piglet litter average weight during the farrowing-pen stage is an important phenotypic indicator for evaluating piglet growth, sow lactation performance, and pen-level management. Manual weighing is labor-intensive and discontinuous, making it difficult to meet the demand for continuous and automated weight monitoring in large-scale pig production.

This project provides a video-based pipeline for estimating piglet litter average weight in farrowing pens by combining multi-pen monitoring, few-shot teacher-student segmentation, and back-projected area regression.

## Overview

The method follows three main stages:

1. **Few-shot teacher segmentation**
   - A SAM3 teacher model is adapted to the farrowing-pen scene using 25 annotated images.
   - The adapted teacher model generates pseudo-labels for unlabeled piglet images.

2. **Student segmentation model**
   - Pseudo-labeled data are used to train a deployable YOLO26s-MoE student model.
   - A Soft MoE module is introduced to improve feature representation while keeping the model suitable for deployment.

3. **Litter average weight estimation**
   - Instance segmentation masks are used to extract piglet back-projected area.
   - Pen-level area features are aggregated and fitted to a regression model for litter average weight estimation.

## Reported Results

Under semi-supervised training, YOLO26s-MoE achieved:

| Metric | Value |
| --- | ---: |
| mAP0.5:0.95 | 57.7% |
| mAP0.5 | 93.9% |
| mAP0.75 | 67.8% |
| mIoU | 74.6% |

The final litter average weight estimation achieved:

| Metric | Value |
| --- | ---: |
| MAE | 0.43 kg |
| R2 | 0.865 |

## Repository Structure

```text
Piglet_weight/
+-- Deploy_using_YOLOv5s/
|   +-- segmentation.py          # Real-time camera frame segmentation and area extraction
|   +-- Regression_analysis.py   # Daily area aggregation and litter average weight regression
+-- soft_moe/
|   +-- README.md                # Soft MoE integration notes for YOLO26
|   +-- modules.py               # MoE modules and compatibility aliases
|   +-- experts.py               # Expert blocks
|   +-- routers.py               # Routing layers
|   +-- loss.py                  # Auxiliary MoE loss
|   +-- utils.py                 # Utility functions
|   +-- __init__.py
+-- data/
    +-- README.md                # Dataset placeholder and expected layout
```

## Data

```text
data/
```

## Deployment Scripts

### Segmentation and Area Extraction

`Deploy_using_YOLOv5s/segmentation.py` loads a trained segmentation model and periodically captures frames from multiple farrowing-pen cameras. For each frame, it computes:

- number of detected piglet instances
- average segmentation mask area

The script writes per-pen daily records to:

```text
temp_data/YYYY-MM-DD/<pen_id>.txt
```

Expected record format:

```text
num_segments, average_area
```

Before running, configure local environment variables. A template is provided in:

```text
.env.example
```

Required and optional variables:

| Variable | Description |
| --- | --- |
| `PIGLET_MODEL_PATH` | Trained segmentation weight path, for example `best.pt` |
| `PIGLET_OUTPUT_DIR` | Output directory for daily area records |
| `PIGLET_CAMERA_USERNAME` | RTSP camera username |
| `PIGLET_CAMERA_PASSWORD` | RTSP camera password |
| `PIGLET_CAMERA_IP_PREFIX` | Camera IP prefix, for example `192.168.1` |
| `PIGLET_CAMERA_PORT` | RTSP port |
| `PIGLET_CAMERA_CHANNEL` | RTSP channel |

Do not commit real camera credentials. Keep them in your shell environment or a private `.env` file.

Example:

```bash
python Deploy_using_YOLOv5s/segmentation.py
```

### Regression Analysis

`Deploy_using_YOLOv5s/Regression_analysis.py` aggregates daily mask-area records, removes simple outliers, and applies pen-specific quadratic regression coefficients to estimate litter average weight.

Before running, fill the `REGRESSION` dictionary with calibrated coefficients:

```python
REGRESSION = {
    # pen_index: (a2, a1, a0)
    # 27: (2.116703234653e-06, -1.697350720881e-03, 1.334891931835e+00),
}
```

The output is written to:

```text
temp_data/weight.txt
```

Example:

```bash
python Deploy_using_YOLOv5s/Regression_analysis.py
```

## Soft MoE Module

The `soft_moe/` directory contains a portable Soft MoE package for YOLO26 integration. See:

```text
soft_moe/README.md
```

for details about copying the module into an Ultralytics/YOLO26 tree, registering the modules, adding YAML entries, and optionally wiring auxiliary MoE losses during training.

## Environment

The deployment scripts require Python with common computer-vision and deep-learning dependencies:

```bash
pip install ultralytics opencv-python numpy torch
```

The exact training environment should be documented here after the training code, dataset layout, and model weights are finalized.

## Notes Before Public Release

- Do not commit private camera credentials, internal IP addresses, or farm-identifying metadata.
- Large video files and model weights should be tracked with Git LFS or released through a separate dataset archive.
- If the dataset cannot be public, provide a data availability statement and a small example file showing the expected format.
