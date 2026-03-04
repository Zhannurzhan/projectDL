import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import time, os, csv
import numpy as np

TRAIN_PATH  = r'C:\Users\szhan\DeepLearning\projectDL\train'
VAL_PATH    = r'C:\Users\szhan\DeepLearning\projectDL\val'
MODEL_DIR   = r'C:\Users\szhan\DeepLearning\projectDL\models'
RESULTS_DIR = r'C:\Users\szhan\DeepLearning\projectDL\results'
GRID_DIR    = r'C:\Users\szhan\DeepLearning\projectDL\results\grid_search'

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

NUM_EPOCHS = 30
PATIENCE   = 10

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_dataset = ImageFolder(VAL_PATH, transform=val_transforms)
num_classes = len(val_dataset.classes)
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

def load_best_config(model_name):
    grid_file = os.path.join(GRID_DIR, f"{model_name}_grid.csv")
    with open(grid_file) as f:
        rows = list(csv.DictReader(f))
    best = max(rows, key=lambda x: float(x['val_acc']))
    print(f"  Best config for {model_name}:")
    print(f"    lr={best['lr_head']}  bs={best['batch_size']}  "
          f"unfreeze={best['unfreeze']}  dropout={best['dropout']}")
    print(f"    Grid search val: {best['val_acc']}%")
    return {
        'lr_head':    float(best['lr_head']),
        'batch_size': int(best['batch_size']),
        'unfreeze':   best['unfreeze'],
        'dropout':    float(best['dropout']),
    }

def build_resnet50(unfreeze, dropout, lr_head):
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for p in m.parameters(): p.requires_grad = False
    if unfreeze == 'last1':
        for p in m.layer4.parameters(): p.requires_grad = True
        pg = [{'params': m.layer4.parameters(), 'lr': lr_head * 0.1}]
    elif unfreeze == 'last2':
        for p in m.layer3.parameters(): p.requires_grad = True
        for p in m.layer4.parameters(): p.requires_grad = True
        pg = [{'params': m.layer3.parameters(), 'lr': lr_head * 0.01},
              {'params': m.layer4.parameters(), 'lr': lr_head * 0.1}]
    else:
        for p in m.layer2.parameters(): p.requires_grad = True
        for p in m.layer3.parameters(): p.requires_grad = True
        for p in m.layer4.parameters(): p.requires_grad = True
        pg = [{'params': m.layer2.parameters(), 'lr': lr_head * 0.001},
              {'params': m.layer3.parameters(), 'lr': lr_head * 0.01},
              {'params': m.layer4.parameters(), 'lr': lr_head * 0.1}]
    if dropout > 0:
        m.fc = nn.Sequential(nn.Dropout(dropout),
                             nn.Linear(m.fc.in_features, num_classes))
    else:
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    pg.append({'params': m.fc.parameters(), 'lr': lr_head})
    return m.to(device), pg

def build_vgg16(unfreeze, dropout, lr_head):
    m = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    for p in m.parameters(): p.requires_grad = False
    if unfreeze == 'last1':
        for p in m.features[24:].parameters(): p.requires_grad = True
        pg = [{'params': m.features[24:].parameters(), 'lr': lr_head * 0.1}]
    elif unfreeze == 'last2':
        for p in m.features[17:].parameters(): p.requires_grad = True
        pg = [{'params': m.features[17:24].parameters(), 'lr': lr_head * 0.01},
              {'params': m.features[24:].parameters(),   'lr': lr_head * 0.1}]
    else:
        for p in m.features[10:].parameters(): p.requires_grad = True
        pg = [{'params': m.features[10:17].parameters(), 'lr': lr_head * 0.001},
              {'params': m.features[17:24].parameters(), 'lr': lr_head * 0.01},
              {'params': m.features[24:].parameters(),   'lr': lr_head * 0.1}]
    in_feat = m.classifier[6].in_features
    if dropout > 0:
        m.classifier[6] = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(in_feat, num_classes))
    else:
        m.classifier[6] = nn.Linear(in_feat, num_classes)
    pg.append({'params': m.classifier.parameters(), 'lr': lr_head})
    return m.to(device), pg

def retrain(model_name, config):
    print(f"\n{'='*60}")
    print(f"  Retraining: {model_name} (full {NUM_EPOCHS} epochs)")
    print(f"{'='*60}")

    train_dataset = ImageFolder(TRAIN_PATH, transform=get_train_transforms())
    train_loader  = DataLoader(train_dataset, batch_size=config['batch_size'],
                               shuffle=True, num_workers=0)
    val_loader    = DataLoader(val_dataset, batch_size=16,
                               shuffle=False, num_workers=0)

    if model_name == 'ResNet50':
        model, pg = build_resnet50(config['unfreeze'], config['dropout'], config['lr_head'])
    else:
        model, pg = build_vgg16(config['unfreeze'], config['dropout'], config['lr_head'])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(pg)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

    best_val_acc      = 0.0
    epochs_no_improve = 0
    history           = []
    total_time        = 0.0

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        model.train()
        t_loss = t_correct = t_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item()
            _, pred    = torch.max(out, 1)
            t_total   += labels.size(0)
            t_correct += (pred == labels).sum().item()

        train_acc = 100 * t_correct / t_total
        avg_loss  = t_loss / len(train_loader)

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
        elapsed = time.time() - t0
        total_time += elapsed
        scheduler.step()

        print(f"  Epoch [{epoch+1:02d}/{NUM_EPOCHS}]  "
              f"Loss: {avg_loss:.4f}  "
              f"Train: {train_acc:.2f}%  "
              f"Val: {val_acc:.2f}%  "
              f"Time: {elapsed:.1f}s")

        history.append({'epoch': epoch+1, 'loss': round(avg_loss, 4),
                        'train_acc': round(train_acc, 2),
                        'val_acc':   round(val_acc,   2)})

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, f"{model_name}_tuned.pth"))
            print(f"  --> Best saved ({val_acc:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Final metrics on val
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, f"{model_name}_tuned.pth"),
        map_location=device, weights_only=True))
    model.eval()
    all_preds, all_labels, inf_times = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            t0  = time.time()
            out = model(images)
            inf_times.append((time.time() - t0) / images.size(0))
            _, pred = torch.max(out, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    np.save(os.path.join(RESULTS_DIR, f"{model_name}_tuned_cm.npy"), cm)

    model_mb = os.path.getsize(
        os.path.join(MODEL_DIR, f"{model_name}_tuned.pth")) / 1e6
    inf_ms   = np.mean(inf_times) * 1000

    hist_file = os.path.join(RESULTS_DIR, f"{model_name}_tuned_history.csv")
    with open(hist_file, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch','loss','train_acc','val_acc'])
        w.writeheader()
        w.writerows(history)

    result = {
        'model':          f"{model_name}_tuned",
        'best_val_acc':   round(best_val_acc,    2),
        'precision':      round(precision * 100, 2),
        'recall':         round(recall    * 100, 2),
        'f1_score':       round(f1        * 100, 2),
        'train_time_s':   round(total_time,      1),
        'inference_ms':   round(inf_ms,          2),
        'model_size_mb':  round(model_mb,        2),
        'epochs_trained': len(history),
    }

    print(f"\n  [{model_name}_tuned] Val={result['best_val_acc']}%  "
          f"F1={result['f1_score']}%  "
          f"Size={result['model_size_mb']}MB  "
          f"Inf={result['inference_ms']}ms")
    return result

# ─── MAIN ─────────────────────────────────────────────────────────────
results = []
for model_name in ['ResNet50', 'VGG16']:
    config = load_best_config(model_name)
    result = retrain(model_name, config)
    results.append(result)

# Update comparison.csv
cmp_file = os.path.join(RESULTS_DIR, "comparison.csv")
fields   = ['model','best_val_acc','precision','recall','f1_score',
            'train_time_s','inference_ms','model_size_mb','epochs_trained']
existing = []
if os.path.exists(cmp_file):
    tuned_names = [r['model'] for r in results]
    with open(cmp_file) as f:
        existing = [r for r in csv.DictReader(f) if r['model'] not in tuned_names]
for r in results:
    existing.append({k: r.get(k, '') for k in fields})
existing.sort(key=lambda x: float(x['best_val_acc']), reverse=True)
with open(cmp_file, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(existing)

print("\n" + "="*60)
print("FINAL COMPARISON — BEFORE vs AFTER TUNING")
print("="*60)
print(f"{'Model':<20} {'Val Acc':>8} {'F1':>8} {'Size MB':>9}")
print("-"*55)
for r in existing:
    print(f"{r['model']:<20} {r['best_val_acc']:>7}%  "
          f"{r['f1_score']:>7}%  "
          f"{r['model_size_mb']:>8}")
print("="*60)
print(f"\nNext: python evaluate_test.py")