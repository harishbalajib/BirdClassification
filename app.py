# app.py — BirdCLEF Inference API with real audio features

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from prometheus_fastapi_instrumentator import Instrumentator

import torchaudio
import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
import os
import io

# === Constants ===
NUM_CLASSES = 206
THRESHOLD = 0.5
SAMPLE_RATE = 16000
RAW_MODEL_INPUT_LEN = 320000  # 10s at 16kHz

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
MODEL_DIR = os.path.join(BASE_DIR, "Optimized_Model")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# === Device Selection (GPU fallback) ===
force_cpu = os.getenv("FORCE_CPU") == "1"
use_gpu = not force_cpu and ort.get_device() == "GPU"
PROVIDERS = ["CUDAExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

# === Load ONNX sessions ===
SESSIONS = {
    "embedding": ort.InferenceSession(os.path.join(MODEL_DIR, "embedding_classifier_opt.onnx"), providers=PROVIDERS),
    "resnet": ort.InferenceSession(os.path.join(MODEL_DIR, "resnet50_multilabel_opt.onnx"), providers=PROVIDERS),
    "effnet": ort.InferenceSession(os.path.join(MODEL_DIR, "efficientnet_b3_lora_opt.onnx"), providers=PROVIDERS),
    "raw": ort.InferenceSession(os.path.join(MODEL_DIR, "raw_audio_cnn_opt.onnx"), providers=PROVIDERS),
    "meta": ort.InferenceSession(os.path.join(MODEL_DIR, "meta_mlp_opt.onnx"), providers=PROVIDERS),
}

# === Dummy class labels ===
CLASSES = [f"class_{i}" for i in range(NUM_CLASSES)]

# === FastAPI App ===
app = FastAPI(
    title="BirdCLEF Inference API",
    description="Upload a .wav or .ogg file for top-1 bird sound classification",
    version="2.0.0"
)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # === Load and preprocess audio ===
        audio_bytes = io.BytesIO(await file.read())
        waveform, sr = torchaudio.load(audio_bytes)

        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)

        waveform = waveform.mean(dim=0) if waveform.dim() > 1 else waveform
        waveform = waveform.float()

        waveform = waveform[:RAW_MODEL_INPUT_LEN]
        if waveform.shape[0] < RAW_MODEL_INPUT_LEN:
            waveform = F.pad(waveform, (0, RAW_MODEL_INPUT_LEN - waveform.shape[0]))

        waveform = (waveform - waveform.mean()) / waveform.std().clamp_min(1e-6)
        wav_input = waveform.unsqueeze(0).numpy().astype(np.float32)

        # === Extract mel spectrogram ===
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=256,
            n_mels=64
        )
        mel = mel_transform(waveform.unsqueeze(0))  # Shape: [1, 64, T]
        mel = torch.log1p(mel)

        # === Resize mel to [1, 1, 64, 313]
        mel_padded = torch.zeros((1, 64, 313))
        mel_len = min(mel.shape[-1], 313)
        mel_padded[:, :, :mel_len] = mel[:, :, :mel_len]
        mel_input = mel_padded.unsqueeze(0).numpy().astype(np.float32)

        # === Placeholder embedding: flatten mel to 2048 vector
        emb_input = mel_padded.view(1, -1)[:, :2048]
        if emb_input.shape[1] < 2048:
            emb_input = F.pad(emb_input, (0, 2048 - emb_input.shape[1]))
        emb_input = emb_input.numpy().astype(np.float32)

        # === Use mel_input as both m and ma (can augment later)
        ma_input = mel_input.copy()
        m_input = mel_input.copy()

        # === Run ONNX inference
        p1 = SESSIONS["embedding"].run(None, {"embedding_input": emb_input})[0]
        p2 = SESSIONS["resnet"].run(None, {"mel_aug_input": ma_input})[0]
        p3 = SESSIONS["effnet"].run(None, {"mel_input": m_input})[0]
        p4 = SESSIONS["raw"].run(None, {"wav_input": wav_input})[0]
        fused = np.concatenate([p1, p2, p3, p4], axis=1)
        final_out = SESSIONS["meta"].run(None, {"fusion_input": fused})[0]

        # === Top-1 Prediction ===
        probs = 1 / (1 + np.exp(-final_out[0]))  # sigmoid
        top_index = int(np.argmax(probs))
        top_label = CLASSES[top_index]
        top_score = float(probs[top_index])

        return {"predicted_label": top_label, "probability": top_score}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Prometheus metrics ===
Instrumentator().instrument(app).expose(app)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
