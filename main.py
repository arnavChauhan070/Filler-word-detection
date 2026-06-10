from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from faster_whisper import WhisperModel
from collections import Counter
import os, time, requests, uuid, shutil, re, subprocess
import torch

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE = "float16" if DEVICE == "cuda" else "int8"

MODEL_PATH = os.environ.get("MODEL_PATH", "nyrahealth/faster_CrisperWhisper")

if MODEL_PATH.startswith("/mnt/"):
    LOCAL_MODEL = "/tmp/model"
    print(f"Checking mount path: {MODEL_PATH}")
    print(f"Mount exists: {os.path.exists(MODEL_PATH)}")
    if os.path.exists(MODEL_PATH):
        print(f"Files in mount: {os.listdir(MODEL_PATH)}")
        if not os.path.exists(LOCAL_MODEL):
            os.makedirs(LOCAL_MODEL, exist_ok=True)
            print(f"Copying model files...")
            result = subprocess.run(
                ["cp", "-r", f"{MODEL_PATH}/.", LOCAL_MODEL],
                capture_output=True, text=True
            )
            print(f"Copy stdout: {result.stdout}")
            print(f"Copy stderr: {result.stderr}")
            print(f"Copy returncode: {result.returncode}")
            print(f"Files in /tmp/model: {os.listdir(LOCAL_MODEL)}")
        MODEL_PATH = LOCAL_MODEL
    else:
        print("MOUNT PATH DOES NOT EXIST - falling back to HuggingFace download")
        MODEL_PATH = "nyrahealth/faster_CrisperWhisper"

print(f"Loading model from: {MODEL_PATH} on {DEVICE}")
model = WhisperModel(MODEL_PATH, device=DEVICE, compute_type=COMPUTE)
print("Model loaded.")


def is_hallucination(text: str) -> bool:
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    if len(words) < 5:
        return False
    top_count = Counter(words).most_common(1)[0][1]
    return (top_count / len(words)) > 0.4


class TranscribeRequest(BaseModel):
    audio_url: str


@app.get("/")
def root():
    return {"message": "CrisperWhisper API"}

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.get("/warm")
def warm():
    return {"status": "warm", "device": DEVICE}

@app.post("/transcribe")
async def transcribe(request: TranscribeRequest):
    start    = time.time()
    uid      = str(uuid.uuid4())
    ext      = request.audio_url.split(".")[-1].split("?")[0]
    tmp_path = f"/tmp/{uid}.{ext}"
    wav_path = f"/tmp/{uid}.wav"

    try:
        r = requests.get(request.audio_url, stream=True, timeout=30)
        if r.status_code != 200:
            raise HTTPException(400, "Failed to download audio.")
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)

        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path,
             "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            capture_output=True, check=True
        )

        segments, _ = model.transcribe(
            wav_path,
            language="en",
            word_timestamps=True,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=800,
            ),
            condition_on_previous_text=True,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            temperature=[0.0, 0.2, 0.4],
            prepend_punctuations="",
            append_punctuations="",
        )

        transcript    = ""
        all_words     = []
        filler_events = []

        for segment in segments:
            if is_hallucination(segment.text):
                continue
            for word in segment.words:
                raw = word.word.strip()
                duration_ms = (word.end - word.start) * 1000
                if duration_ms < 50 and "[" not in raw:
                    continue
                filler_match = re.search(r'\[([^\]]+)\]', raw)
                if filler_match:
                    filler_events.append({
                        "word":  filler_match.group(0),
                        "start": round(word.start, 3),
                        "end":   round(word.end, 3),
                    })
                    continue
                transcript += raw.lstrip(".,") + " "
                clean = re.sub(r"[^\w\s]", "", raw).strip().lower()
                if clean:
                    all_words.append(clean)

    except HTTPException:
        raise
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"ffmpeg failed: {e.stderr.decode()}")
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        for p in [tmp_path, wav_path]:
            if os.path.exists(p):
                os.remove(p)

    return JSONResponse({
        "transcript":      transcript.strip(),
        "filler_events":   filler_events,
        "filler_count":    len(filler_events),
        "word_frequency":  dict(Counter(all_words).most_common()),
        "latency_seconds": round(time.time() - start, 2),
    })