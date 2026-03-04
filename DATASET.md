# Dataset Documentation

## 1. Overview

| Property | Value |
|----------|-------|
| Domain | Building and Construction Tools |
| Total Images | 281 |
| Categories | 23 |
| Format | JPEG / PNG |
| Input Size | 224 × 224 (resized during training) |

---

## 2. Data Collection & Physical Market Approval

This dataset was curated for the "CNN-Based Product Recognition and Price Estimation System" project. It consists of images of professional building and construction tools collected at a physical hardware market.

- **Approval**: Verbal approval was obtained from the store manager to photograph their inventory for this deep learning project.
- **Process**: A smartphone camera was used to capture tools from multiple angles and under various lighting conditions (natural storefront light and indoor fluorescent light) to ensure the model can generalize to real-world user uploads.

---

## 3. Real-Time Price Database

In accordance with Project Requirement 6, a `prices.json` file was created using real-time market data collected at the hardware store. All prices are in Kazakhstani Tenge (KZT).

| # | Class Name | Label | Price (KZT) |
|---|-----------|-------|-------------|
| 1 | bopp_packing_tape | Bopp Packing Tape | 500 |
| 2 | box_cutter | Box Cutter | 650 |
| 3 | combination_pliers | Combination Pliers | 2500 |
| 4 | combination_pliers_blue | Combination Pliers Blue | 1600 |
| 5 | crescent_wrench | Crescent Wrench | 1200 |
| 6 | crowbar | Crowbar | 1350 |
| 7 | darby | Darby | 1300 |
| 8 | fast_adhesive_kit | Fast Adhesive Kit | 1000 |
| 9 | flat_head_screwdriver | Flat Head Screwdriver | 400 |
| 10 | gloves | Gloves | 300 |
| 11 | hammer | Hammer | 2800 |
| 12 | hand_saw | Hand Saw | 1800 |
| 13 | hatchet | Hatchet | 7500 |
| 14 | heart_shaped_trowel | Heart Shaped Trowel | 1700 |
| 15 | masking_tape | Masking Tape | 2000 |
| 16 | masonry_roller | Masonry Roller | 4000 |
| 17 | paintbrush | Paintbrush | 350 |
| 18 | polyurethane_float | Polyurethane Float | 1100 |
| 19 | pump_pliers | Pump Pliers | 5500 |
| 20 | putty_knife | Putty Knife | 350 |
| 21 | rubber_mallet | Rubber Mallet | 1800 |
| 22 | tape_measure | Tape Measure | 900 |
| 23 | tin_snips | Tin Snips | 2500 |

---

## 4. Dataset Split

| Split | Images | Classes | Images per Class |
|-------|--------|---------|-----------------|
| Train | 205 | 23 | ~9 |
| Validation | 39 | 23 | ~2 |
| Test | 37 | 23 | ~2 |
| **Total** | **281** | **23** | **~12** |

Split strategy:
- 67% of images per class assigned to training
- Remaining images split equally between validation and test
- All 23 classes appear in every split
- Split performed manually to ensure class balance

---

## 5. Preprocessing

All images go through the following pipeline before inference:

```python
transforms.Resize((224, 224))
transforms.ToTensor()
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])  # ImageNet statistics
```

---

## 6. Data Augmentation (Training Only)

To compensate for the small dataset size, the following augmentations were applied during training:

| Augmentation | Parameters | Purpose |
|-------------|-----------|---------|
| RandomResizedCrop | size=224, scale=(0.5, 1.0) | Simulate different distances |
| RandomHorizontalFlip | p=0.5 | Orientation invariance |
| RandomVerticalFlip | p=0.5 | Orientation invariance |
| RandomRotation | degrees=30 | Rotated tool placement |
| ColorJitter | brightness=0.5, contrast=0.5, saturation=0.4, hue=0.15 | Lighting variation |
| RandomGrayscale | p=0.1 | Color robustness |
| RandomPerspective | p=0.3 | Camera angle variation |

---

## 7. Folder Structure

```
projectDL/
├── train/
│   ├── bopp_packing_tape/   (~9 images)
│   ├── box_cutter/          (~9 images)
│   └── ...                  (23 folders total)
├── val/
│   ├── bopp_packing_tape/   (~2 images)
│   └── ...
└── test/
    ├── bopp_packing_tape/   (~2 images)
    └── ...
```

Each subfolder name is the class label used by `torchvision.datasets.ImageFolder`.

---

## 8. Limitations

- Small dataset (~12 images per class) — mitigated by data augmentation and transfer learning
- Some classes are visually similar (e.g. combination_pliers vs combination_pliers_blue)
- Images collected under semi-controlled conditions — real-world performance may vary