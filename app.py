from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import uvicorn
import io
from inference import wavs_to_fbank_tensors
from tester import Tester

app = FastAPI(title="STT API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
# TODO: load your model/decoder once at startup
@app.on_event("startup")
async def startup():
    # Example:
    # global asr
    # asr = YourASR.load_from_checkpoint("/models/ckpt.pt")
    MODEL = Tester(folder_path="./pseudo_data", checkpoint_path="./pretrained.pt", is_streaming=False, decoding_method="greedy_search", max_duration=300)

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> Dict[str, str]:
    # Read audio bytes
    audio_bytes = await file.read()
    wav = io.BytesIO(audio_bytes)
    # TODO: decode (e.g., torchaudio.load + model inference)
    # text = asr.decode(wav)
    input_paths = [wav]
    input_batch = wavs_to_fbank_tensors(input_paths, device="cuda")
    outputs = MODEL(input_batch, is_sherpa_format=False)
    text = outputs[0]
    # text = "dummy transcription"
    return {"text": text}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=True)
