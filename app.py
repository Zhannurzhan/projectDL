from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json, os, io
import numpy as np

app = FastAPI(title="Tool Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── CONFIG ──────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "ResNet50_best.pth")
PRICES_PATH = os.path.join(BASE_DIR, "prices.json")
NUM_CLASSES = 23
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── LOAD PRICES ─────────────────────────────────────────────────────
with open(PRICES_PATH) as f:
    PRICES = json.load(f)

# ─── LOAD MODEL ──────────────────────────────────────────────────────
def load_model():
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m.to(DEVICE)

model      = load_model()
print(f"Model loaded on {DEVICE}")

# ─── CLASS NAMES (alphabetical — matches ImageFolder order) ──────────
TRAIN_DIR = os.path.join(BASE_DIR, "train")
CLASS_NAMES = sorted(os.listdir(TRAIN_DIR))
print(f"Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")

# ─── TRANSFORMS ──────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─── PREDICT ENDPOINT ────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top5_probs, top5_idx = torch.topk(probs, 5)

    predictions = []
    for prob, idx in zip(top5_probs.cpu().numpy(), top5_idx.cpu().numpy()):
        name  = CLASS_NAMES[idx]
        price = PRICES.get(name, "N/A")
        predictions.append({
            "rank":       len(predictions) + 1,
            "class_name": name,
            "label":      name.replace("_", " ").title(),
            "confidence": round(float(prob) * 100, 2),
            "price":      price,
        })

    top = predictions[0]
    return JSONResponse({
        "success":     True,
        "predicted":   top["label"],
        "class_name":  top["class_name"],
        "confidence":  top["confidence"],
        "price":       top["price"],
        "top5":        predictions,
        "device":      str(DEVICE),
    })

# ─── HEALTH CHECK ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "classes": len(CLASS_NAMES)}

# ─── SERVE FRONTEND ──────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(BASE_DIR, "static", "index.html")
    with open(html_path) as f:
        return f.read()

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)