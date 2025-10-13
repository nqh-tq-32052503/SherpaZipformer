# app.py
import os
import tempfile
import subprocess
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import soundfile as sf

from test_hieunq10 import inference_one_file
app = FastAPI(title="Sherpa-ONNX HTTP ASR")



SAMPLE_RATE = 16000  # target sample rate for recognition
RECOGNIZER = None


@app.get("/healthz")
def healthz():
    ok = RECOGNIZER is not None
    return {"ok": ok}

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
        # Read as float32 and feed to recognizer
        results = inference_one_file(wav_path)
        return JSONResponse({
            "transcript": results,
        })
