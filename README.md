# CSHNet: Cross-modal Semantic Hypergraph Network for UAV Multimodal Object Detection

A dual-stream RGB + thermal infrared object detection framework built on RT-DETR-R18, designed for UAV aerial imagery.

## Overview

CSHNet introduces four core modules to address cross-modal semantic alignment and feature fusion challenges in multimodal UAV detection:

- **CSAM** (Cross-modal Semantic Alignment Module): aligns semantic representations between RGB and infrared streams
- **PGRM** (Progressive Gated Recalibration Module): recalibrates cross-modal features through gated mechanisms
- **HGAM** (Hypergraph-guided Aggregation Module): constructs hypergraph structures to model high-order cross-modal relationships
- **HFIM** (Hierarchical Feature Integration Module): integrates multi-scale features across modalities

## Datasets

| Dataset | Modalities | Classes | Description |
|---------|-----------|---------|-------------|
| M3FD | RGB + Infrared | 6 (People, Car, Bus, Motorcycle, Lamp, Truck) | UAV multimodal detection |
| VEDAI | RGB + Infrared | — | Vehicle detection in aerial imagery |
| DroneVehicle | RGB + Infrared | — | UAV vehicle detection |

## Results

| Dataset | mAP50 | mAP50:95 |
|---------|-------|----------|
| M3FD | 60.2% | — |
| VEDAI | 82.5% | — |
| DroneVehicle (zero-shot) | 61.6% | 31.6% |

## Requirements

```
Python >= 3.10
PyTorch >= 2.0
CUDA >= 11.8
ultralytics
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

Organize your dataset in the following structure:

```
datasets/
└── M3FD_split/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── images_ir/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

Then update the `path` field in `ultralytics/cfg/datasets/mmdata/data.yaml` to your local dataset root.

## Training

```python
from ultralytics import RTDETRMM

model = RTDETRMM('CSHNet.yaml')
model.train(
    data='ultralytics/cfg/datasets/mmdata/data.yaml',
    epochs=50,
    device=0,
    batch=16,
    project='runs',
    name='CSHNet'
)
```

Or run directly:

```bash
python trainRT.py
```

## Inference

Dual-modal inference (RGB + Infrared):

```python
from ultralytics import RTDETRMM

model = RTDETRMM('path/to/best.pt')
model.predict(
    rgb_source='path/to/rgb_image.png',
    x_source='path/to/ir_image.png',
    save=True
)
```

Single-modal inference is also supported by setting the unused modality to `None`.

## Model Architecture

The full model configuration is defined in `CSHNet.yaml`. Module-level configurations are available in:

- `CSAM.yaml` — Cross-modal Semantic Alignment Module
- `PGRM.yaml` — Progressive Gated Recalibration Module
- `HGAM.yaml` — Hypergraph-guided Aggregation Module
- `HFIM.yaml` — Hierarchical Feature Integration Module

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{cshnet2025,
  title={CSHNet: Cross-modal Semantic Hypergraph Network for UAV Multimodal Object Detection},
  author={},
  journal={},
  year={2025}
}
```

## Acknowledgements

This codebase is built upon [Ultralytics](https://github.com/ultralytics/ultralytics). We thank the authors for their excellent framework.
