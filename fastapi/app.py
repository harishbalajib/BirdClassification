from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torchaudio
import torch
import torch.nn.functional as F
import numpy as np
import os
import io
from tritonclient.http import InferenceServerClient, InferInput
from prometheus_fastapi_instrumentator import Instrumentator

# === Constants ===
SAMPLE_RATE = 16000
RAW_MODEL_INPUT_LEN = 320000
TRITON_URL = os.getenv("TRITON_SERVER_URL", "triton_server:8000")  # Docker internal hostname
CLASSES = [f"class_{i}" for i in range(206)]  # Dummy class names

# === Initialize FastAPI ===
app = FastAPI(
    title="BirdCLEF Triton API",
    description="Upload .wav file for bird classification using ONNX + Triton",
    version="1.0.0"
)

# === Triton Client ===
triton_client = InferenceServerClient(url=TRITON_URL)

# === Prometheus ===
Instrumentator().instrument(app).expose(app)

# === Inference Helper ===
def infer_model(model_name: str, input_name: str, input_array: np.ndarray, output_name: str):
    infer_input = InferInput(input_name, input_array.shape, "FP32")
    infer_input.set_data_from_numpy(input_array)
    result = triton_client.infer(model_name=model_name, inputs=[infer_input])
    return result.as_numpy(output_name)

# === Preprocessing ===
def preprocess_audio(wav_bytes: bytes):
    waveform, sr = torchaudio.load(io.BytesIO(wav_bytes))
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    waveform = waveform.mean(dim=0) if waveform.dim() > 1 else waveform
    waveform = F.pad(waveform, (0, max(0, RAW_MODEL_INPUT_LEN - waveform.shape[0])))
    waveform = (waveform - waveform.mean()) / waveform.std().clamp_min(1e-6)
    return waveform[:RAW_MODEL_INPUT_LEN].unsqueeze(0).numpy().astype(np.float32)

def get_mel_and_embedding(waveform_tensor):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=64
    )
    mel = torch.log1p(mel_transform(torch.tensor(waveform_tensor)))
    mel_padded = torch.zeros((1, 64, 313))
    mel_len = min(mel.shape[-1], 313)
    mel_padded[:, :, :mel_len] = mel[:, :, :mel_len]
    mel_np = mel_padded.unsqueeze(0).numpy().astype(np.float32)

    emb = mel_padded.view(1, -1)[:, :2048]
    if emb.shape[1] < 2048:
        emb = F.pad(emb, (0, 2048 - emb.shape[1]))
    emb_np = emb.numpy().astype(np.float32)
    return mel_np, emb_np

# === Route ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # === Load and preprocess ===
        audio_bytes = await file.read()
        wav_input = preprocess_audio(audio_bytes)
        mel_input, emb_input = get_mel_and_embedding(torch.tensor(wav_input[0]))

        # === Triton Inference ===
        p1 = infer_model("embedding_classifier_opt", "embedding_input", emb_input, "embedding_output")
        p2 = infer_model("resnet50_multilabel_opt", "mel_aug_input", mel_input, "resnet_output")
        p3 = infer_model("efficientnet_b3_lora_opt", "mel_input", mel_input, "effnet_output")
        p4 = infer_model("raw_audio_cnn_opt", "wav_input", wav_input, "raw_output")

        fused_input = np.concatenate([p1, p2, p3, p4], axis=1).astype(np.float32)
        final_pred = infer_model("meta_mlp_opt", "fusion_input", fused_input, "meta_output")

        probs = 1 / (1 + np.exp(-final_pred[0]))
        idx = int(np.argmax(probs))

        return {
            "predicted_label": CLASSES[idx],
            "probability": float(probs[idx])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
