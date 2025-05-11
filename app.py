# BirdCLEF Inference API (Multimodal Fusion with Internal Preprocessing)

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
import os
import io
from prometheus_fastapi_instrumentator import Instrumentator
import torch.nn as nn
import timm
from peft import get_peft_model, LoraConfig
import uvicorn

# Constants
NUM_CLASSES = 206
THRESHOLD = 0.5
SAMPLE_RATE = 16000
DURATION_SEC = 10
RAW_MODEL_INPUT_LEN = 320000  # <-- updated for correct shape
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HTML template path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Define models
class MetaMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout):
        super().__init__()
        layers, dims = [], [in_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        layers.append(nn.Linear(dims[-1], NUM_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DummyModel(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(np.prod(input_shape), output_dim)
    def forward(self, x): return self.fc(self.flatten(x))

# Load models (replace with real ones later)
CKPT_META = os.path.join(BASE_DIR, "best_meta_mlp.pt")

emb_model = DummyModel((2048,), NUM_CLASSES).to(DEVICE).eval()
res_model = DummyModel((1, 64, 313), NUM_CLASSES).to(DEVICE).eval()
eff_model = DummyModel((1, 64, 313), NUM_CLASSES).to(DEVICE).eval()
raw_model = DummyModel((RAW_MODEL_INPUT_LEN,), NUM_CLASSES).to(DEVICE).eval()
meta_model = MetaMLP(in_dim=NUM_CLASSES * 4, hidden_dims=[1024, 512], dropout=0.3).to(DEVICE)
meta_model.load_state_dict(torch.load(CKPT_META, map_location=DEVICE))
meta_model.eval()

# Dummy class labels
CLASSES = [f"class_{i}" for i in range(NUM_CLASSES)]

# FastAPI instance
app = FastAPI(title="BirdCLEF Inference API", description="Upload a .wav or .ogg file for prediction", version="1.0.0")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        audio_bytes = io.BytesIO(file.file.read())
        waveform, sr = torchaudio.load(audio_bytes)

        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        waveform = waveform.mean(dim=0) if waveform.dim() > 1 else waveform
        waveform = waveform.float()

        T = RAW_MODEL_INPUT_LEN  # <-- key fix
        if waveform.size(0) < T:
            waveform = F.pad(waveform, (0, T - waveform.size(0)))
        else:
            waveform = waveform[:T]

        waveform = (waveform - waveform.mean()) / waveform.std().clamp_min(1e-6)
        wav = waveform.unsqueeze(0).to(DEVICE)

        # Dummy input tensors for other models
        emb = torch.tensor(np.random.randn(1, 2048), dtype=torch.float32).to(DEVICE)
        ma  = torch.tensor(np.random.randn(1, 1, 64, 313), dtype=torch.float32).to(DEVICE)
        m   = torch.tensor(np.random.randn(1, 1, 64, 313), dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            p1 = torch.sigmoid(emb_model(emb))
            p2 = torch.sigmoid(res_model(ma))
            p3 = torch.sigmoid(eff_model(m))
            p4 = torch.sigmoid(raw_model(wav))
            features = torch.cat([p1, p2, p3, p4], dim=1)
            logits = meta_model(features)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        predicted = [CLASSES[i] for i, p in enumerate(probs) if p >= THRESHOLD]
        confidence = [float(p) for i, p in enumerate(probs) if p >= THRESHOLD]
        return {"predicted_labels": predicted, "probabilities": confidence}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

Instrumentator().instrument(app).expose(app)

if __name__ == '__main__':
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
