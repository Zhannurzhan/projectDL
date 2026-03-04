import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import time, os, csv, json, itertools
import numpy as np

TRAIN_PATH  = r'C:\Users\szhan\DeepLearning\projectDL\train'
VAL_PATH    = r'C:\Users\szhan\DeepLearning\projectDL\val'
MODEL_DIR   = r'C:\Users\szhan\DeepLearning\projectDL\models'
RESULTS_DIR = r'C:\Users\szhan\DeepLearning\projectDL\results'
GRID_DIR    = r'C:\Users\szhan\DeepLearning\projectDL\results\grid_search'

os.makedirs(GRID_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# GRID PARAMETERS
# ============================================================
GRID = {
    'lr_head':    [1e-3, 1e-4, 1e-5],
    'batch_size': [16, 32],
    'unfreeze':   ['last1', 'last2'],
    'dropout':    [0.0, 0.3],
}

# 3x2x2x2 = 24 combos x 2 models = 48 total
# 5 epochs each ~ 1-2 hours on GPU
SEARCH_EPOCHS  = 5
# Full retrain of best config
FULL_EPOCHS    = 40
PATIENCE       = 8
MODELS_TO_TUNE = ['ResNet50', 'VGG16']

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = ImageFolder(VAL_PATH, transform=val_transforms)
num_classes = len(val_dataset.classes)
class_names = val_dataset.classes
print(f"Classes: {num_classes}\n")

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.5, 0.5, 0.4, 0.15),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomPerspective(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# ============================================================
# BUILD MODEL WITH GIVEN CONFIG
# ============================================================
def build_resnet50(unfreeze, dropout, num_classes):
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for p in m.parameters():
        p.requires_grad = False

    if unfreeze == 'last1':
        for p in m.layer4.parameters(): p.requires_grad = True
        pg_backbone = [{'params': m.layer4.parameters(), 'lr_scale': 0.1}]
    elif unfreeze == 'last2':
        for p in m.layer3.parameters(): p.requires_grad = True
        for p in m.layer4.parameters(): p.requires_grad = True
        pg_backbone = [
            {'params': m.layer3.parameters(), 'lr_scale': 0.01},
            {'params': m.layer4.parameters(), 'lr_scale': 0.1},
        ]
    else:  # last3
        for p in m.layer2.parameters(): p.requires_grad = True
        for p in m.layer3.parameters(): p.requires_grad = True
        for p in m.layer4.parameters(): p.requires_grad = True
        pg_backbone = [
            {'params': m.layer2.parameters(), 'lr_scale': 0.001},
            {'params': m.layer3.parameters(), 'lr_scale': 0.01},
            {'params': m.layer4.parameters(), 'lr_scale': 0.1},
        ]

    if dropout > 0:
        m.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(m.fc.in_features, num_classes)
        )
    else:
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    return m, pg_backbone, list(m.fc.parameters())


def build_vgg16(unfreeze, dropout, num_classes):
    m = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    for p in m.parameters():
        p.requires_grad = False

    if unfreeze == 'last1':
        for p in m.features[24:].parameters(): p.requires_grad = True
        pg_backbone = [{'params': m.features[24:].parameters(), 'lr_scale': 0.1}]
    elif unfreeze == 'last2':
        for p in m.features[17:].parameters(): p.requires_grad = True
        pg_backbone = [
            {'params': m.features[17:24].parameters(), 'lr_scale': 0.01},
            {'params': m.features[24:].parameters(),   'lr_scale': 0.1},
        ]
    else:  # last3
        for p in m.features[10:].parameters(): p.requires_grad = True
        pg_backbone = [
            {'params': m.features[10:17].parameters(), 'lr_scale': 0.001},
            {'params': m.features[17:24].parameters(), 'lr_scale': 0.01},
            {'params': m.features[24:].parameters(),   'lr_scale': 0.1},
        ]

    in_feat = m.classifier[6].in_features
    if dropout > 0:
        m.classifier[6] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, num_classes)
        )
    else:
        m.classifier[6] = nn.Linear(in_feat, num_classes)

    return m, pg_backbone, list(m.classifier.parameters())


def build_model(model_name, unfreeze, dropout):
    if model_name == 'ResNet50':
        return build_resnet50(unfreeze, dropout, num_classes)
    else:
        return build_vgg16(unfreeze, dropout, num_classes)


# ============================================================
# TRAIN ONE COMBO (quick search)
# ============================================================
def train_combo(model_name, lr_head, batch_size, unfreeze, dropout, epochs):
    train_dataset = ImageFolder(TRAIN_PATH, transform=get_train_transforms())
    train_loader  = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=0)
    val_loader    = DataLoader(val_dataset, batch_size=16,
                               shuffle=False, num_workers=0)

    m, pg_backbone, pg_head = build_model(model_name, unfreeze, dropout)
    model = m.to(device)

    # Build param groups with scaled LRs
    param_groups = []
    for pg in pg_backbone:
        param_groups.append({
            'params': pg['params'],
            'lr':     lr_head * pg['lr_scale']
        })
    param_groups.append({'params': pg_head, 'lr': lr_head})

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            if hasattr(out, 'logits'): out = out.logits
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        v_correct = v_total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                _, pred   = torch.max(out, 1)
                v_total   += labels.size(0)
                v_correct += (pred == labels).sum().item()
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = 100 * v_correct / v_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return round(best_val_acc, 2), round(f1 * 100, 2)


# ============================================================
# GRID SEARCH
# ============================================================
keys   = list(GRID.keys())
values = list(GRID.values())
combos = list(itertools.product(*values))
total  = len(combos)

print(f"Grid search: {total} combinations × {len(MODELS_TO_TUNE)} models")
print(f"Search epochs per combo: {SEARCH_EPOCHS}")
print(f"Estimated time: ~{total * len(MODELS_TO_TUNE) * SEARCH_EPOCHS * 15 // 60} min\n")

all_rows = []

for model_name in MODELS_TO_TUNE:
    print(f"\n{'='*65}")
    print(f"  Grid Search: {model_name}")
    print(f"{'='*65}")
    print(f"  {'#':>3}  {'LR':>8}  {'BS':>4}  {'Unfreeze':>8}  "
          f"{'Drop':>5}  {'Val%':>7}  {'F1%':>7}")
    print(f"  {'-'*55}")

    model_rows = []

    for i, combo in enumerate(combos):
        config = dict(zip(keys, combo))
        t0     = time.time()

        val_acc, f1 = train_combo(
            model_name,
            lr_head    = config['lr_head'],
            batch_size = config['batch_size'],
            unfreeze   = config['unfreeze'],
            dropout    = config['dropout'],
            epochs     = SEARCH_EPOCHS
        )

        elapsed = time.time() - t0
        row = {
            'model':    model_name,
            'lr_head':  config['lr_head'],
            'batch_size': config['batch_size'],
            'unfreeze': config['unfreeze'],
            'dropout':  config['dropout'],
            'val_acc':  val_acc,
            'f1':       f1,
        }
        model_rows.append(row)
        all_rows.append(row)

        print(f"  {i+1:>3}  {config['lr_head']:>8}  {config['batch_size']:>4}  "
              f"{config['unfreeze']:>8}  {config['dropout']:>5}  "
              f"{val_acc:>6}%  {f1:>6}%  ({elapsed:.0f}s)")

    # Best config for this model
    best = max(model_rows, key=lambda x: x['val_acc'])
    print(f"\n  BEST {model_name}: "
          f"lr={best['lr_head']}  bs={best['batch_size']}  "
          f"unfreeze={best['unfreeze']}  dropout={best['dropout']}  "
          f"-> {best['val_acc']}% val")

    # Save per-model grid results
    grid_file = os.path.join(GRID_DIR, f"{model_name}_grid.csv")
    model_rows.sort(key=lambda x: x['val_acc'], reverse=True)
    with open(grid_file, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(model_rows[0].keys()))
        w.writeheader()
        w.writerows(model_rows)
    print(f"  Grid results saved -> {grid_file}")

# Save full grid results
full_file = os.path.join(GRID_DIR, "all_grid_results.csv")
all_rows.sort(key=lambda x: x['val_acc'], reverse=True)
with open(full_file, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    w.writeheader()
    w.writerows(all_rows)

# ============================================================
# PRINT TOP 5 CONFIGS PER MODEL
# ============================================================
print("\n" + "="*65)
print("TOP 5 CONFIGS PER MODEL")
print("="*65)
for model_name in MODELS_TO_TUNE:
    rows = [r for r in all_rows if r['model'] == model_name][:5]
    print(f"\n  {model_name}:")
    print(f"  {'LR':>8}  {'BS':>4}  {'Unfreeze':>8}  {'Drop':>5}  {'Val%':>7}  {'F1%':>7}")
    print(f"  {'-'*50}")
    for r in rows:
        print(f"  {r['lr_head']:>8}  {r['batch_size']:>4}  "
              f"{r['unfreeze']:>8}  {r['dropout']:>5}  "
              f"{r['val_acc']:>6}%  {r['f1']:>6}%")

print(f"\nFull results -> {full_file}")
print(f"\nNext: python retrain_best.py")