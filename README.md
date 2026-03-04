# ToolScan — CNN-Based Tool Recognition & Price Estimation

A deep learning system that identifies **23 categories of building and construction tools** from a single image and returns the estimated price. Built for Deep Learning Course — Project 1.

---

## Results Summary

| Model | Val Accuracy | Test Accuracy | Top-5 | F1-Score | Size |
|-------|-------------|---------------|-------|----------|------|
| **ResNet50 (tuned)** | **89.74%** | **94.59%** | **100%** | **94.20%** | 94 MB |
| VGG16 (tuned) | 92.31% | 78.38% | 97.30% | 75.07% | 537 MB |
| AlexNet | 66.67% | 64.86% | 94.59% | 56.23% | 228 MB |
| EfficientNet-B0 | 69.23% | 59.46% | 83.78% | 50.04% | 16 MB |
| GoogLeNet | 58.97% | 45.95% | 81.08% | 37.38% | 23 MB |

**Best model: ResNet50 (tuned)** — 94.59% Top-1, 100% Top-5 on test set.

---

## Project Structure

```
projectDL/
├── train/                    # Training images (205, 23 classes)
├── val/                      # Validation images (39, 23 classes)
├── test/                     # Test images (37, 23 classes)
├── models/                   # Saved model weights (.pth)
│   ├── ResNet50_best.pth     # Best model for inference
│   ├── VGG16_best.pth
│   ├── AlexNet_best.pth
│   ├── GoogLeNet_best.pth
│   └── EfficientNet_best.pth
├── results/                  # Training metrics and plots
│   ├── comparison.csv        # Val accuracy comparison
│   ├── test_results.csv      # Test set evaluation
│   ├── plots/                # 6 comparison charts
│   └── grid_search/          # Grid search results
├── static/
│   └── index.html            # Web app frontend
├── app.py                    # FastAPI web application
├── prices.json               # Price database (23 classes)
├── train_alexnet.py
├── train_vgg16.py
├── train_googlenet.py
├── train_resnet50.py
├── train_efficientnet.py
├── grid_search.py            # Hyperparameter grid search
├── retrain_best.py           # Retrain with best configs
├── evaluate_test.py          # Test set evaluation
└── compare_models.py         # Generate comparison plots
```

---

## Setup

### Requirements

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn python-multipart scikit-learn matplotlib seaborn pillow
```

### GPU Check

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Training

Train each model individually:

```bash
python train_alexnet.py
python train_vgg16.py
python train_googlenet.py
python train_resnet50.py
python train_efficientnet.py
```

### Hyperparameter Tuning (optional)

```bash
python grid_search.py      # ~1.5 hours, searches 24 combos per model
python retrain_best.py     # retrain best configs for 40 epochs
```

### Evaluate on Test Set

```bash
python evaluate_test.py    # Top-1, Top-5, F1, Precision, Recall
python compare_models.py   # generates 6 comparison plots
```

---

## Web Application

```bash
python app.py
```

Open **http://localhost:8000** in your browser.

### Features
- Upload any tool image (JPG, PNG, WEBP)
- Drag & drop support
- Top-5 predictions with confidence scores
- Price lookup from `prices.json`
- GPU/CPU status indicator

---

## Dataset

- **Domain**: Building and construction tools
- **Categories**: 23 classes
- **Total images**: 281 (205 train / 39 val / 37 test)
- **Images per class**: 6–12 original photos
- **Augmentation**: RandomResizedCrop, Flip, Rotation, ColorJitter, Perspective

### Classes

```
bopp_packing_tape    box_cutter           combination_pliers
combination_pliers_blue  crescent_wrench  crowbar
darby                fast_adhesive_kit    flat_head_screwdriver
gloves               hammer               hand_saw
hatchet              heart_shaped_trowel  masking_tape
masonry_roller       paintbrush           polyurethane_float
pump_pliers          putty_knife          rubber_mallet
tape_measure         tin_snips
```

---

## Model Details

All models use **transfer learning** with ImageNet pre-trained weights.

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (per-layer LR) |
| Loss | CrossEntropyLoss (label_smoothing=0.1) |
| Scheduler | CosineAnnealingLR |
| Batch size | 16 |
| Max epochs | 30 |
| Early stopping | patience=10 |
| Input size | 224×224 |
| Hardware | NVIDIA RTX 3050 4GB |

### ResNet50 Best Config (from grid search)

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.001 |
| Batch size | 16 |
| Unfrozen layers | layer4 only |
| Dropout | 0.3 |

---

## Key Findings

- **ResNet50 generalizes best** — improved from 89.74% val to 94.59% test
- **VGG16 overfits** — 92.31% val but only 78.38% test (138M params, small dataset)
- **Grid search** improved ResNet50 by +5.12% validation accuracy
- **Label smoothing** (0.1) and **CosineAnnealingLR** improved stability
- **Top-5 accuracy**: ResNet50 achieves 100% — correct class always in top 5

---

## Author

**Zhannur** — Deep Learning Course, March 2026