# BirdCLEF Inference API (Multimodal Fusion with Internal Preprocessing)

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
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

# Constants
NUM_CLASSES = 206
THRESHOLD = 0.5
SAMPLE_RATE = 16000
DURATION_SEC = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load dummy models (replace with real ones)
MODEL_DIR = "/mnt/BirdCLEF/Models"
CKPT_META = os.path.join(MODEL_DIR, "best_meta_mlp.pt")

emb_model = DummyModel((2048,), NUM_CLASSES).to(DEVICE).eval()
res_model = DummyModel((1, 64, 313), NUM_CLASSES).to(DEVICE).eval()
eff_model = DummyModel((1, 64, 313), NUM_CLASSES).to(DEVICE).eval()
raw_model = DummyModel((320000,), NUM_CLASSES).to(DEVICE).eval()
meta_model = MetaMLP(in_dim=NUM_CLASSES * 4, hidden_dims=[1024, 512], dropout=0.3).to(DEVICE)
meta_model.load_state_dict(torch.load(CKPT_META, map_location=DEVICE))
meta_model.eval()

# Placeholder class names (replace with real labels)
CLASSES = [f"class_{i}" for i in range(NUM_CLASSES)]

# API setup
app = FastAPI(title="BirdCLEF Inference API", description="Upload a .wav file for prediction", version="1.0.0")

import uvicorn

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        # Load audio
        waveform, sr = torchaudio.load(io.BytesIO(file.file.read()))
        waveform = waveform.squeeze(0).float()

        # Pad or trim
        T = SAMPLE_RATE * DURATION_SEC
        if waveform.size(0) < T:
            waveform = F.pad(waveform, (0, T - waveform.size(0)))
        else:
            waveform = waveform[:T]
        waveform = (waveform - waveform.mean()) / waveform.std().clamp_min(1e-6)
        wav = waveform.unsqueeze(0).to(DEVICE)

        # Simulate actual preprocessed inputs with correct shapes
        emb = torch.tensor(np.random.randn(1, 2048), dtype=torch.float32).to(DEVICE)
        ma  = torch.tensor(np.random.randn(1, 1, 64, 313), dtype=torch.float32).to(DEVICE)
        m   = torch.tensor(np.random.randn(1, 1, 64, 313), dtype=torch.float32).to(DEVICE)

        # Run models
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
    uvicorn.run("__main__:app", host="0.0.0.0", port=6540)
