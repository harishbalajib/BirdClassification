from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from prometheus_fastapi_instrumentator import Instrumentator

import torchaudio
import torch
import torch.nn.functional as F
import numpy as np
import tritonclient.http as httpclient
from pydub import AudioSegment
import io
import os

# === Constants ===
NUM_CLASSES = 206
THRESHOLD = 0.5
SAMPLE_RATE = 16000
RAW_MODEL_INPUT_LEN = 320000
TRITON_URL = os.getenv("TRITON_SERVER_URL", "triton_server:8000")

# === Setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
triton_client = httpclient.InferenceServerClient(url=TRITON_URL)

CLASSES = [f"class_{i}" for i in range(NUM_CLASSES)]
app = FastAPI(title="BirdCLEF Triton API", version="1.0")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # === Convert OGG to WAV ===
        audio_bytes = io.BytesIO(await file.read())
        audio = AudioSegment.from_file(audio_bytes, format="ogg")
        wav_bytes = io.BytesIO()
        audio.set_frame_rate(SAMPLE_RATE).set_channels(1).export(wav_bytes, format="wav")
        wav_bytes.seek(0)

        waveform, sr = torchaudio.load(wav_bytes)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)

        waveform = waveform.mean(dim=0) if waveform.ndim > 1 else waveform
        waveform = waveform.float()
        waveform = waveform[:RAW_MODEL_INPUT_LEN]
        if waveform.shape[0] < RAW_MODEL_INPUT_LEN:
            waveform = F.pad(waveform, (0, RAW_MODEL_INPUT_LEN - waveform.shape[0]))
        waveform = (waveform - waveform.mean()) / waveform.std().clamp_min(1e-6)
        wav_input = waveform.unsqueeze(0).numpy().astype(np.float32)

        # === Mel Spectrogram ===
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=256,
            n_mels=64
        )
        mel = torch.log1p(mel_transform(waveform.unsqueeze(0)))
        mel_padded = torch.zeros((1, 64, 313))
        mel_len = min(mel.shape[-1], 313)
        mel_padded[:, :, :mel_len] = mel[:, :, :mel_len]
        mel_input = mel_padded.unsqueeze(0).numpy().astype(np.float32)

        # === Embedding input
        emb_input = mel_padded.view(1, -1)[:, :2048]
        if emb_input.shape[1] < 2048:
            emb_input = F.pad(emb_input, (0, 2048 - emb_input.shape[1]))
        emb_input = emb_input.numpy().astype(np.float32)

        # === Inference Function with correct output name
        def triton_infer(model_name, input_name, input_data, output_name):
            input_tensor = httpclient.InferInput(input_name, input_data.shape, "FP32")
            input_tensor.set_data_from_numpy(input_data)
            result = triton_client.infer(
                model_name=model_name,
                inputs=[input_tensor],
                outputs=[httpclient.InferRequestedOutput(output_name)]
            )
            return result.as_numpy(output_name)

        # === Individual Model Calls
        p1 = triton_infer("embedding_classifier_opt", "embedding_input", emb_input, "embedding_output")
        p2 = triton_infer("resnet50_multilabel_opt", "mel_aug_input", mel_input, "resnet_output")
        p3 = triton_infer("efficientnet_b3_lora_opt", "mel_input", mel_input, "effnet_output")
        p4 = triton_infer("raw_audio_cnn_opt", "wav_input", wav_input, "raw_output")

        fused = np.concatenate([p1, p2, p3, p4], axis=1)
        p_fused = triton_infer("meta_mlp_opt", "fusion_input", fused, "meta_output")

        probs = 1 / (1 + np.exp(-p_fused[0]))
        top_index = int(np.argmax(probs))
        return {
            "predicted_label": CLASSES[top_index],
            "probability": float(probs[top_index])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Prometheus Monitoring ===
Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
