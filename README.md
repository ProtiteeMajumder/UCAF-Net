# UCAF-Net

**UCAF-Net** is an uncertainty-guided dual-backbone image-classification framework combining ResNet50 and MobileNetV2 with Dual-Path Statistical Channel-Spatial Attention (DSCSA), bidirectional cross-attention, and uncertainty-guided feature fusion. This repository provides the **dataset-independent architecture and training/evaluation functions**, rather than a complete dataset-specific experiment.


## Architecture

**Overall architecture.** Input images are processed by parallel ResNet50 and MobileNetV2 branches enhanced with DSCSA. Aligned features are tokenized, exchanged through uncertainty-conditioned bidirectional cross-attention, fused through a gated mechanism, and aggregated by attention pooling before classification.


<img src="https://github.com/user-attachments/assets/7aa4084d-22f1-4f5c-a4ed-f69c36c139d4" alt="UCAF-Net overall architecture" width="700" />

**DSCSA module.** The attention block uses average, maximum, and standard-deviation pooling to capture complementary channel statistics, followed by spatial attention to refine the extracted feature maps.


<img src="https://github.com/user-attachments/assets/d50d8095-0687-4a60-81e7-f62302022f1c" alt="DSCSA architecture" width="550" />


## Repository structure

```text
UCAF-Net/
├── README.md
├── final_overall_archi.jpg
├── CBAM.jpg
├── modified_cbam.py
├── resnet50_cbam.py
├── mobilenetv2_cbam.py
├── fusion_modules.py
├── hybrid_model.py
├── loss_metrics.py
└── training.py
```

| File | Description |
| --- | --- |
| `modified_cbam.py` | Implements DSCSA, including statistical pooling and channel/spatial attention. |
| `resnet50_cbam.py` | ResNet50 feature-extraction branch with the modified attention module. |
| `mobilenetv2_cbam.py` | MobileNetV2 feature-extraction branch with the modified attention module. |
| `fusion_modules.py` | Feature alignment, uncertainty estimation, bidirectional cross-attention, and gated feature fusion. |
| `hybrid_model.py` | Integrates both backbones, attention and fusion modules, attention pooling, and the classification head into `HybridModel`. |
| `loss_metrics.py` | Uncertainty-aware classification loss and accuracy, precision, recall, and F1-score calculations. |
| `training.py` | Reusable `train_one_epoch`, `validate_one_epoch`, `test_model`, and `run_training` functions, including validation-loss-based checkpoint selection. |

## Datasets used in our experiments

The same core UCAF-Net architecture was applied to two microscopy classification tasks, with the output-class count adjusted for each dataset:

- **BreakHis:** Breast histopathology images from the 40×, 100×, 200×, and 400× magnification groups, combined for **binary classification** into *benign* and *malignant*.
- **PBC (8-class):** Peripheral blood-cell microscopy images classified into **basophil, eosinophil, erythroblast, immature granulocyte (IG), lymphocyte, monocyte, neutrophil, and platelet**.

In the original experiments, images were resized to **256 × 256**, training images were augmented, and ImageNet mean/standard-deviation normalization was used. The experiments used AdamW with an initial learning rate of **3 × 10⁻⁴**, weight decay **10⁻⁴**, and a cosine-annealing learning-rate scheduler. These are details of our original experimental setup, **not requirements imposed on other datasets**.

**Dataset images, preprocessing code, split definitions, and dataset loaders are not included in this repository.** Users should construct their own training, validation, and test loaders, with images and integer class labels in the format expected by PyTorch.

## Installation

The code uses Python with PyTorch, TorchVision, and scikit-learn. Install compatible versions for your system:

```bash
pip install torch torchvision scikit-learn
```

No particular package-version combination is pinned in this release.

## Using UCAF-Net on another dataset

Initialize the model with the required number of output classes:

```python
import torch
from hybrid_model import HybridModel

num_classes = 2  # Replace with the number of classes in your dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridModel(num_classes=num_classes).to(device)
```

The model returns three outputs: **classification logits**, **ResNet50-stream uncertainty**, and **MobileNetV2-stream uncertainty**.

Once you have created your own `train_loader`, `val_loader`, and `test_loader`, the provided training loop can be used as follows:

```python
from training import run_training

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)

history = run_training(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    num_epochs=100
)
```

This example assumes the three data loaders have already been defined. The supplied training function saves the lowest-validation-loss checkpoint as `best_model.pth` in the current working directory and evaluates it on the test loader after training. Its learning-rate scheduler is stepped **once per training batch**, matching the provided implementation.
