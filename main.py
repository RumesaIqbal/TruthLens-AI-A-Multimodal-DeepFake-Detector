import os
import re
import numpy as np
import cv2
import uvicorn

import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tensorflow.keras.models import load_model

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TruthLens AI",
    description="Multimodal AI-generated content detector for images and text.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Config ─────────────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS  = {"png", "jpg", "jpeg", "webp"}
MAX_TEXT_LENGTH     = 128                              # matches training MAX_LENGTH
IMAGE_MODEL_PATH    = "models/ai_vs_human_cnn.h5"
TEXT_MODEL_PATH     = "models/ai_text_detector"       # folder with config, model.safetensors, tokenizer files

# Label map — must match training: 0 = Human Written, 1 = AI Generated
LABEL_MAP = {0: "Human Written", 1: "AI Generated"}

# ── Global model handles ───────────────────────────────────────────────────────
image_model  = None
text_model   = None
tokenizer    = None
text_device  = None


# ── Load Models at Startup ─────────────────────────────────────────────────────
@app.on_event("startup")
async def load_models():
    global image_model, text_model, tokenizer, text_device

    # ── Image model (TensorFlow / Keras CNN) ──────────────────────────────────
    print("Loading CNN image model...")
    image_model = load_model(IMAGE_MODEL_PATH, compile=False)
    print("✅ CNN image model loaded.")

    # ── Text model (PyTorch DistilBERT) ───────────────────────────────────────
    print("Loading DistilBERT text model...")
    text_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Text model device: {text_device}")

    tokenizer  = DistilBertTokenizerFast.from_pretrained(TEXT_MODEL_PATH)
    text_model = DistilBertForSequenceClassification.from_pretrained(TEXT_MODEL_PATH)
    text_model = text_model.to(text_device)
    text_model.eval()
    print("✅ DistilBERT text model loaded.")


# ── Pydantic Schema ────────────────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str


# ── Helpers ────────────────────────────────────────────────────────────────────
def is_allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_text(text: str) -> str:
    """
    Mirrors the preprocessing used during training:
      - Lowercase
      - Remove URLs
      - Remove special characters (keep letters, digits, basic punctuation)
      - Collapse extra whitespace
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r"[^a-z0-9\s.,!?'\"\-]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Image Prediction ───────────────────────────────────────────────────────────
def run_image_prediction(image_bytes: bytes) -> dict:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = float(image_model.predict(img)[0][0])

    if pred > 0.5:
        return {"label": "AI Generated",    "confidence": round(pred, 4)}
    else:
        return {"label": "Human Generated", "confidence": round(1 - pred, 4)}


# ── Text Prediction ────────────────────────────────────────────────────────────
def run_text_prediction(input_text: str) -> dict:
    """
    Uses the fine-tuned PyTorch DistilBERT model from ai_text_detector/.
    Label mapping: 0 = Human Written, 1 = AI Generated  (matches training).
    """
    cleaned = preprocess_text(input_text)

    encoding = tokenizer(
        cleaned,
        max_length=MAX_TEXT_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids      = encoding["input_ids"].to(text_device)
    attention_mask = encoding["attention_mask"].to(text_device)

    with torch.no_grad():
        outputs = text_model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]  # [human_prob, ai_prob]

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    label      = LABEL_MAP[pred_class]

    return {
        "label":      label,
        "confidence": round(confidence, 4),
        "human_prob": round(float(probs[0]), 4),
        "ai_prob":    round(float(probs[1]), 4),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("static/index.html", "r") as f:
        return f.read()


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if not is_allowed(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a JPG, PNG, or WEBP image."
        )

    image_bytes = await file.read()
    result = run_image_prediction(image_bytes)
    result["confidence_percent"] = f"{result['confidence'] * 100:.1f}%"
    return result


@app.post("/predict/text")
async def predict_text(request: TextRequest):
    text = request.text.strip()

    if len(text) < 10:
        raise HTTPException(
            status_code=400,
            detail="Text is too short. Please enter at least a sentence."
        )

    result = run_text_prediction(text)
    result["confidence_percent"] = f"{result['confidence'] * 100:.1f}%"
    return result


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)