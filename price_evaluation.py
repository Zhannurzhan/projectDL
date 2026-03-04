import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import json, os, csv
import numpy as np

TEST_PATH   = r'C:\Users\szhan\DeepLearning\projectDL\test'
MODEL_DIR   = r'C:\Users\szhan\DeepLearning\projectDL\models'
PRICES_PATH = r'C:\Users\szhan\DeepLearning\projectDL\prices.json'
RESULTS_DIR = r'C:\Users\szhan\DeepLearning\projectDL\results'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load prices
with open(PRICES_PATH) as f:
    PRICES = json.load(f)

# Dataset
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dataset = ImageFolder(TEST_PATH, transform=test_transforms)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
class_names  = test_dataset.classes
num_classes  = len(class_names)

def load_resnet50():
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    state = torch.load(os.path.join(MODEL_DIR, "ResNet50_best.pth"),
                       map_location=device, weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m.to(device)

MODEL_NAMES = ["AlexNet", "VGG16", "GoogLeNet", "ResNet50", "EfficientNet"]

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
    try:
        m.load_state_dict(state)
    except RuntimeError:
        m.load_state_dict(state, strict=False)
    m.eval()
    return m.to(device)

print("\n" + "="*60)
print("PRICE ESTIMATION EVALUATION (MAE & RMSE)")
print("="*60)

all_results = []

for name in MODEL_NAMES:
    model = load_model(name)
    if model is None:
        continue

    true_prices = []
    pred_prices = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            out    = model(images)
            _, pred_idx = torch.max(out, 1)

            for true_label, pred_label in zip(labels.numpy(), pred_idx.cpu().numpy()):
                true_class = class_names[true_label]
                pred_class = class_names[pred_label]

                true_price = PRICES.get(true_class, None)
                pred_price = PRICES.get(pred_class, None)

                if true_price is not None and pred_price is not None:
                    # Extract numeric value from price string if needed
                    if isinstance(true_price, str):
                        true_price = float(''.join(c for c in true_price if c.isdigit() or c == '.'))
                    if isinstance(pred_price, str):
                        pred_price = float(''.join(c for c in pred_price if c.isdigit() or c == '.'))

                    true_prices.append(float(true_price))
                    pred_prices.append(float(pred_price))

    true_prices = np.array(true_prices)
    pred_prices = np.array(pred_prices)

    mae  = np.mean(np.abs(true_prices - pred_prices))
    rmse = np.sqrt(np.mean((true_prices - pred_prices) ** 2))
    mape = np.mean(np.abs((true_prices - pred_prices) / (true_prices + 1e-8))) * 100

    result = {
        'model': name,
        'mae':   round(mae,  2),
        'rmse':  round(rmse, 2),
        'mape':  round(mape, 2),
        'samples': len(true_prices),
    }
    all_results.append(result)

    print(f"\n  [{name}]")
    print(f"    MAE  (Mean Absolute Error)      : {mae:.2f}")
    print(f"    RMSE (Root Mean Squared Error)  : {rmse:.2f}")
    print(f"    MAPE (Mean Abs Percentage Error): {mape:.2f}%")
    print(f"    Samples evaluated               : {len(true_prices)}")

# Save results
price_csv = os.path.join(RESULTS_DIR, "price_evaluation.csv")
with open(price_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['model','mae','rmse','mape','samples'])
    w.writeheader()
    w.writerows(all_results)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Model':<14} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8}")
print("-"*42)
for r in sorted(all_results, key=lambda x: x['mae']):
    print(f"{r['model']:<14} {r['mae']:>8} {r['rmse']:>8} {r['mape']:>7}%")
print("="*60)
print(f"\nSaved -> {price_csv}")