import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import (precision_recall_fscore_support,
                             confusion_matrix, classification_report)
import os, csv, json
import numpy as np

TEST_PATH   = r'C:\Users\szhan\DeepLearning\projectDL\test'
MODEL_DIR   = r'C:\Users\szhan\DeepLearning\projectDL\models'
RESULTS_DIR = r'C:\Users\szhan\DeepLearning\projectDL\results'

os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = ImageFolder(TEST_PATH, transform=test_transforms)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
class_names  = test_dataset.classes
num_classes  = len(class_names)
print(f"Test: {len(test_dataset)} images | {num_classes} classes\n")

def load_model(name):
    if name == "AlexNet":
        m = models.alexnet(weights=None)
        m.classifier[6] = nn.Linear(4096, num_classes)
    elif name == "VGG16":
        m = models.vgg16(weights=None)
        m.classifier[6] = nn.Linear(4096, num_classes)
    elif name == "GoogLeNet":
        m = models.googlenet(weights=None, aux_logits=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "ResNet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "EfficientNet":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)

    pth = os.path.join(MODEL_DIR, f"{name}_best.pth")
    if not os.path.exists(pth):
        print(f"  WARNING: {pth} not found, skipping.")
        return None

    state = torch.load(pth, map_location=device, weights_only=True)
    # Handle dropout wrapper in fc/classifier
    try:
        m.load_state_dict(state)
    except RuntimeError:
        # Try stripping 'module.' prefix if saved with DataParallel
        new_state = {k.replace("module.", ""): v for k, v in state.items()}
        m.load_state_dict(new_state, strict=False)

    m.eval()
    return m.to(device)

MODEL_NAMES = ["AlexNet", "VGG16", "GoogLeNet", "ResNet50", "EfficientNet"]
all_results = []

print("=" * 65)
print("TEST SET EVALUATION")
print("=" * 65)

for name in MODEL_NAMES:
    model = load_model(name)
    if model is None:
        continue

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            _, pred = torch.max(out, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    np.save(os.path.join(RESULTS_DIR, f"{name}_test_cm.npy"), cm)

    # Top-5 accuracy
    top5_correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            _, top5 = torch.topk(out, min(5, num_classes), dim=1)
            for i, label in enumerate(labels):
                if label in top5[i]:
                    top5_correct += 1
    top5_acc = 100 * top5_correct / len(all_labels)

    result = {
        'model':      name,
        'test_acc':   round(acc,             2),
        'top5_acc':   round(top5_acc,        2),
        'precision':  round(precision * 100, 2),
        'recall':     round(recall    * 100, 2),
        'f1_score':   round(f1        * 100, 2),
    }
    all_results.append(result)

    print(f"\n  [{name}]")
    print(f"    Top-1 Accuracy : {acc:.2f}%")
    print(f"    Top-5 Accuracy : {top5_acc:.2f}%")
    print(f"    Precision      : {precision*100:.2f}%")
    print(f"    Recall         : {recall*100:.2f}%")
    print(f"    F1-Score       : {f1*100:.2f}%")

    # Per-class report
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, zero_division=0)
    report_path = os.path.join(RESULTS_DIR, f"{name}_test_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Model: {name}\n")
        f.write(f"Test Accuracy: {acc:.2f}%\n\n")
        f.write(report)
    print(f"    Per-class report saved -> {report_path}")

# Save test results CSV
test_csv = os.path.join(RESULTS_DIR, "test_results.csv")
fields   = ['model','test_acc','top5_acc','precision','recall','f1_score']
all_results.sort(key=lambda x: x['test_acc'], reverse=True)
with open(test_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(all_results)

print("\n" + "=" * 65)
print("FINAL TEST RESULTS")
print("=" * 65)
print(f"{'Model':<14} {'Top-1':>7} {'Top-5':>7} {'F1':>7} {'Prec':>7} {'Rec':>7}")
print("-" * 65)
for r in all_results:
    print(f"{r['model']:<14} {r['test_acc']:>6}%  {r['top5_acc']:>6}%  "
          f"{r['f1_score']:>6}%  {r['precision']:>6}%  {r['recall']:>6}%")
print("=" * 65)
print(f"\nResults saved -> {test_csv}")
print(f"Next: python compare_models.py") 