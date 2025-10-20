from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import uvicorn
import io
from inference import wavs_to_fbank_tensors
from tester import Tester
import os 
import tempfile
import subprocess

app = FastAPI(title="STT API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
SAMPLE_RATE = 16000

# TODO: load your model/decoder once at startup
@app.on_event("startup")
def _load():
    # Example:
    # global asr
    # asr = YourASR.load_from_checkpoint("/models/ckpt.pt")
    global MODEL
    MODEL = Tester(folder_path="./pseudo_data", checkpoint_path="./pretrained.pt", is_streaming=False, decoding_method="greedy_search", max_duration=300)

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

def ffmpeg_to_wav16k(in_path: str, out_path: str):
    # Convert anything → 16k mono s16 WAV
    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-sample_fmt", "s16", out_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


@app.post("/inference")
def inference(file: UploadFile = File(...)):
    # Save upload to temp, convert to wav16k if needed
    with tempfile.TemporaryDirectory() as td:
        src_path = os.path.join(td, file.filename)
        with open(src_path, "wb") as f:
            f.write(file.file.read())

        wav_path = os.path.join(td, "pcm16k.wav")
        print("Converting input to 16k WAV...")
        try:
            ffmpeg_to_wav16k(src_path, wav_path)
        except Exception as e:
            raise HTTPException(400, f"ffmpeg failed to decode input: {e}")
        print("Converted input to 16k WAV")
        input_paths = [wav_path]
        input_batch = wavs_to_fbank_tensors(input_paths, device="cuda")
        outputs = MODEL(input_batch, is_sherpa_format=False)
        text = outputs[0]
        # text = "dummy transcription"
        return {"text": text}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=True)
