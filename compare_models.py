import os, csv, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

RESULTS_DIR = r'C:\Users\szhan\DeepLearning\projectDL\results'
PLOTS_DIR   = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_NAMES = ["AlexNet", "VGG16", "GoogLeNet", "ResNet50", "EfficientNet"]
COLORS      = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

# ─── Load comparison CSV ─────────────────────────────────────────────
def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

val_rows  = load_csv(os.path.join(RESULTS_DIR, "comparison.csv"))
test_rows = load_csv(os.path.join(RESULTS_DIR, "test_results.csv"))

val_dict  = {r['model']: r for r in val_rows}
test_dict = {r['model']: r for r in test_rows}

print("Loaded models:")
print(f"  Val  : {list(val_dict.keys())}")
print(f"  Test : {list(test_dict.keys())}\n")

# ─── 1. BAR CHART: Val vs Test Accuracy ──────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
x       = np.arange(len(MODEL_NAMES))
width   = 0.35

val_accs  = [float(val_dict.get(m, {}).get('best_val_acc', 0))  for m in MODEL_NAMES]
test_accs = [float(test_dict.get(m, {}).get('test_acc', 0)) for m in MODEL_NAMES]

bars1 = ax.bar(x - width/2, val_accs,  width, label='Validation', color='#4C72B0', alpha=0.85)
bars2 = ax.bar(x + width/2, test_accs, width, label='Test',       color='#DD8452', alpha=0.85)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Validation vs Test Accuracy by Model', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(MODEL_NAMES)
ax.set_ylim(0, 105)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '1_val_vs_test_accuracy.png'), dpi=150)
plt.close()
print("Saved: 1_val_vs_test_accuracy.png")

# ─── 2. BAR CHART: F1, Precision, Recall ─────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
x     = np.arange(len(MODEL_NAMES))
width = 0.25

f1s   = [float(test_dict.get(m, {}).get('f1_score',  0)) for m in MODEL_NAMES]
precs = [float(test_dict.get(m, {}).get('precision', 0)) for m in MODEL_NAMES]
recs  = [float(test_dict.get(m, {}).get('recall',    0)) for m in MODEL_NAMES]

ax.bar(x - width, f1s,   width, label='F1-Score',  color='#55A868', alpha=0.85)
ax.bar(x,         precs, width, label='Precision',  color='#4C72B0', alpha=0.85)
ax.bar(x + width, recs,  width, label='Recall',     color='#C44E52', alpha=0.85)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('F1-Score, Precision and Recall by Model (Test Set)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(MODEL_NAMES)
ax.set_ylim(0, 105)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '2_f1_precision_recall.png'), dpi=150)
plt.close()
print("Saved: 2_f1_precision_recall.png")

# ─── 3. SCATTER: Accuracy vs Model Size ──────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
sizes = [float(val_dict.get(m, {}).get('model_size_mb', 0)) for m in MODEL_NAMES]

for i, m in enumerate(MODEL_NAMES):
    ax.scatter(sizes[i], val_accs[i], s=200, color=COLORS[i], zorder=5, label=m)
    ax.annotate(m, (sizes[i], val_accs[i]),
                textcoords="offset points", xytext=(8, 4), fontsize=10)

ax.set_xlabel('Model Size (MB)', fontsize=12)
ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax.set_title('Accuracy vs Model Size Trade-off', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '3_accuracy_vs_size.png'), dpi=150)
plt.close()
print("Saved: 3_accuracy_vs_size.png")

# ─── 4. TRAINING CURVES ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, m in enumerate(MODEL_NAMES):
    hist_path = os.path.join(RESULTS_DIR, f"{m}_history.csv")
    if not os.path.exists(hist_path):
        axes[i].text(0.5, 0.5, f'{m}\nNo history found',
                     ha='center', va='center', transform=axes[i].transAxes)
        continue

    with open(hist_path) as f:
        hist = list(csv.DictReader(f))

    epochs     = [int(r['epoch'])      for r in hist]
    train_accs = [float(r['train_acc']) for r in hist]
    val_accs_h = [float(r['val_acc'])   for r in hist]
    losses     = [float(r['loss'])      for r in hist]

    ax = axes[i]
    ax2 = ax.twinx()
    ax.plot(epochs, train_accs, 'b-o', markersize=3, label='Train Acc', linewidth=1.5)
    ax.plot(epochs, val_accs_h, 'r-o', markersize=3, label='Val Acc',   linewidth=1.5)
    ax2.plot(epochs, losses,    'g--', markersize=2, label='Loss',       linewidth=1, alpha=0.6)

    ax.set_title(f'{m}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)', color='black')
    ax2.set_ylabel('Loss', color='green')
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=8)

# Hide unused subplot
if len(MODEL_NAMES) < 6:
    axes[5].set_visible(False)

plt.suptitle('Training Curves — All Models', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '4_training_curves.png'), dpi=150)
plt.close()
print("Saved: 4_training_curves.png")

# ─── 5. CONFUSION MATRIX — Best Model (ResNet50) ─────────────────────
best_model = "ResNet50"
cm_path    = os.path.join(RESULTS_DIR, f"{best_model}_test_cm.npy")

if os.path.exists(cm_path):
    cm = np.load(cm_path)

    # Load class names from test folder
    from torchvision.datasets import ImageFolder
    TEST_PATH   = r'C:\Users\szhan\DeepLearning\projectDL\test'
    test_ds     = ImageFolder(TEST_PATH)
    class_names = test_ds.classes

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label',      fontsize=12)
    ax.set_title(f'Confusion Matrix — {best_model} (Test Set)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '5_confusion_matrix_resnet50.png'), dpi=150)
    plt.close()
    print("Saved: 5_confusion_matrix_resnet50.png")
else:
    print(f"WARNING: {cm_path} not found — run evaluate_test.py first")

# ─── 6. INFERENCE SPEED vs ACCURACY ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
inf_ms = [float(val_dict.get(m, {}).get('inference_ms', 0)) for m in MODEL_NAMES]

for i, m in enumerate(MODEL_NAMES):
    ax.scatter(inf_ms[i], val_accs[i], s=200, color=COLORS[i], zorder=5)
    ax.annotate(m, (inf_ms[i], val_accs[i]),
                textcoords="offset points", xytext=(6, 4), fontsize=10)

ax.set_xlabel('Inference Time (ms per image)', fontsize=12)
ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax.set_title('Accuracy vs Inference Speed', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '6_accuracy_vs_speed.png'), dpi=150)
plt.close()
print("Saved: 6_accuracy_vs_speed.png")

# ─── SUMMARY TABLE ───────────────────────────────────────────────────
print("\n" + "="*75)
print("COMPLETE RESULTS SUMMARY")
print("="*75)
print(f"{'Model':<14} {'Val%':>7} {'Test%':>7} {'Top5%':>7} {'F1%':>7} {'MB':>7} {'ms':>7}")
print("-"*75)
for m in MODEL_NAMES:
    v = val_dict.get(m,  {})
    t = test_dict.get(m, {})
    print(f"{m:<14} "
          f"{v.get('best_val_acc','N/A'):>7}  "
          f"{t.get('test_acc','N/A'):>7}  "
          f"{t.get('top5_acc','N/A'):>7}  "
          f"{t.get('f1_score','N/A'):>7}  "
          f"{v.get('model_size_mb','N/A'):>7}  "
          f"{v.get('inference_ms','N/A'):>7}")
print("="*75)
print(f"\nAll plots saved -> {PLOTS_DIR}")
print(f"Next: cd app && python app.py")