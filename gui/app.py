#!/usr/bin/env python3
"""
Faster Qwen3-TTS desktop GUI (local web app).

Usage:
    python gui/app.py                    # opens http://127.0.0.1:7870
    python gui/app.py --port 8000 --no-browser
    python gui/app.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base   # preload a model

Models that are not cached yet are downloaded from Hugging Face on first load.
"""

import argparse
import gc
import io
import json
import re
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from faster_qwen3_tts import FasterQwen3TTS  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
VOICES_DIR = DATA_DIR / "voices"
OUTPUT_DIR = DATA_DIR / "outputs"
HISTORY_FILE = DATA_DIR / "history.json"
VOICES_FILE = DATA_DIR / "voices.json"
for _d in (VOICES_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MODELS = [
    {"id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "label": "0.6B Base", "type": "base", "note": "Fast voice cloning"},
    {"id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "label": "1.7B Base", "type": "base", "note": "Best-quality voice cloning"},
    {"id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "label": "0.6B CustomVoice", "type": "custom_voice", "note": "Built-in speakers"},
    {"id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "label": "1.7B CustomVoice", "type": "custom_voice", "note": "Built-in speakers + style instructions"},
    {"id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "label": "1.7B VoiceDesign", "type": "voice_design", "note": "Describe a voice in words"},
]
LANGUAGES = ["Auto", "English", "Chinese", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# Built-in demo reference voice shipped in the repo (text matches the README example).
BUILTIN_VOICES = []
_ref = REPO_DIR / "ref_audio.wav"
if _ref.exists():
    BUILTIN_VOICES.append({
        "id": "builtin-ref",
        "name": "Repo sample (ref_audio.wav)",
        "path": str(_ref),
        "ref_text": (
            "I'm confused why some people have super short timelines, yet at the same time are bullish on "
            "scaling up reinforcement learning atop LLMs. If we're actually close to a human-like learner, "
            "then this whole approach of training on verifiable outcomes is doomed."
        ),
        "builtin": True,
    })

app = FastAPI(title="Faster Qwen3-TTS GUI")

_state = {"model": None, "model_id": None, "model_type": None, "dtype": None, "loading": None, "error": None}
_gen_lock = threading.Lock()
_load_lock = threading.Lock()
_stop_flags: dict[str, bool] = {}
_file_lock = threading.Lock()


# ---------------------------------------------------------------- helpers
def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _is_cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        # config.json alone can be left over from a partial download, so check for weights
        return any(isinstance(try_to_load_from_cache(model_id, f), str)
                   for f in ("model.safetensors", "model.safetensors.index.json"))
    except Exception:
        return False


def _gpu_info():
    if not torch.cuda.is_available():
        return {"available": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "used_gb": round((total - free) / 1e9, 1),
        "total_gb": round(total / 1e9, 1),
    }


def _speakers(model) -> list[str]:
    try:
        if hasattr(model, "get_supported_speakers"):
            return list(model.get_supported_speakers() or [])
        return list(model.model.get_supported_speakers() or [])
    except Exception:
        return []


def _languages(model) -> list[str]:
    try:
        langs = model.model.get_supported_languages() or []
        if langs:
            return ["Auto"] + [str(l).title() for l in langs if str(l).lower() != "auto"]
    except Exception:
        pass
    return LANGUAGES


def _all_voices() -> list[dict]:
    saved = _read_json(VOICES_FILE, [])
    return BUILTIN_VOICES + [v for v in saved if Path(v["path"]).exists()]


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")[:40] or "voice"


# ---------------------------------------------------------------- pages / status
@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    model = _state["model"]
    return {
        "loaded": model is not None,
        "model_id": _state["model_id"],
        "model_type": _state["model_type"],
        "dtype": _state["dtype"],
        "loading": _state["loading"],
        "error": _state["error"],
        "gpu": _gpu_info(),
        "speakers": _speakers(model) if model else [],
        "languages": _languages(model) if model else LANGUAGES,
        "busy": _gen_lock.locked(),
        "models": [{**m, "cached": _is_cached(m["id"])} for m in MODELS],
    }


# ---------------------------------------------------------------- model load / unload
class LoadReq(BaseModel):
    model_id: str
    dtype: str = "bf16"


def _unload_locked():
    _state.update(model=None, model_id=None, model_type=None, dtype=None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _do_load(model_id: str, dtype: str):
    try:
        with _gen_lock:  # never swap models mid-generation
            _unload_locked()
            m = FasterQwen3TTS.from_pretrained(
                model_id,
                device="cuda" if torch.cuda.is_available() else "cpu",
                dtype=DTYPES[dtype],
                attn_implementation="sdpa",
                max_seq_len=2048,
            )
            _state.update(model=m, model_id=model_id, dtype=dtype, model_type=m.model.model.tts_model_type)
            try:
                m.warmup(prefill_len=100)
            except Exception as exc:  # best-effort; generation prepares lazily anyway
                print(f"warmup skipped: {exc}")
    except Exception as exc:
        _state.update(model=None, model_id=None, model_type=None, error=f"{type(exc).__name__}: {exc}")
        print(f"Model load failed: {exc}")
    finally:
        _state["loading"] = None


@app.post("/api/load")
def load(req: LoadReq):
    if req.dtype not in DTYPES:
        raise HTTPException(400, "Bad dtype")
    if req.model_id not in {m["id"] for m in MODELS}:
        raise HTTPException(400, "Unknown model")
    with _load_lock:
        if _state["loading"]:
            raise HTTPException(409, "A model is already loading")
        _state.update(loading=req.model_id, error=None)
    threading.Thread(target=_do_load, args=(req.model_id, req.dtype), daemon=True).start()
    return {"status": "loading"}


@app.post("/api/unload")
def unload():
    if _state["loading"]:
        raise HTTPException(409, "Model is loading")
    with _gen_lock:
        _unload_locked()
    return {"status": "unloaded"}


# ---------------------------------------------------------------- voices (reference library)
@app.get("/api/voices")
def voices():
    return [{k: v for k, v in x.items() if k != "path"} for x in _all_voices()]


@app.post("/api/voices")
async def add_voice(name: str = Form(...), ref_text: str = Form(""), audio: UploadFile = File(...)):
    data = await audio.read()
    if not data:
        raise HTTPException(400, "Empty audio")
    vid = uuid.uuid4().hex[:10]
    suffix = Path(audio.filename or "").suffix.lower() or ".wav"
    path = VOICES_DIR / f"{_slug(name)}_{vid}{suffix}"
    path.write_bytes(data)
    try:
        info = sf.info(str(path))
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"Unreadable audio file: {exc}")
    entry = {"id": vid, "name": name.strip() or "Voice", "path": str(path), "ref_text": ref_text.strip(),
             "duration": round(info.duration, 1), "builtin": False}
    with _file_lock:
        saved = _read_json(VOICES_FILE, [])
        saved.append(entry)
        _write_json(VOICES_FILE, saved)
    return {k: v for k, v in entry.items() if k != "path"}


@app.delete("/api/voices/{vid}")
def delete_voice(vid: str):
    with _file_lock:
        saved = _read_json(VOICES_FILE, [])
        for v in saved:
            if v["id"] == vid:
                Path(v["path"]).unlink(missing_ok=True)
        _write_json(VOICES_FILE, [v for v in saved if v["id"] != vid])
    return {"status": "deleted"}


@app.get("/api/voices/{vid}/audio")
def voice_audio(vid: str):
    for v in _all_voices():
        if v["id"] == vid:
            return FileResponse(v["path"])
    raise HTTPException(404)


# ---------------------------------------------------------------- generation
class GenReq(BaseModel):
    text: str
    language: str = "English"
    mode: str  # clone | custom | design (must match the loaded model)
    voice_id: str | None = None
    ref_text: str | None = None
    xvec_only: bool = False
    speaker: str | None = None
    instruct: str | None = None
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    do_sample: bool = True
    max_new_tokens: int = 2048
    chunk_size: int = 8
    client_id: str = ""


MODE_FOR_TYPE = {"base": "clone", "custom_voice": "custom", "voice_design": "design"}


def _make_stream(model, req: GenReq):
    common = dict(
        text=req.text,
        language=req.language,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        do_sample=req.do_sample,
        repetition_penalty=req.repetition_penalty,
        chunk_size=req.chunk_size,
    )
    if req.mode == "clone":
        voice = next((v for v in _all_voices() if v["id"] == req.voice_id), None)
        if not voice:
            raise HTTPException(400, "Pick or add a reference voice first")
        ref_text = req.ref_text if req.ref_text else voice["ref_text"]
        if not req.xvec_only and not (ref_text or "").strip():
            raise HTTPException(400, "Reference transcript is required (or enable 'Voice only' mode)")
        return model.generate_voice_clone_streaming(
            ref_audio=voice["path"], ref_text=ref_text or "", xvec_only=req.xvec_only, **common)
    if req.mode == "custom":
        if not req.speaker:
            raise HTTPException(400, "Pick a speaker")
        return model.generate_custom_voice_streaming(
            speaker=req.speaker, instruct=req.instruct or None, **common)
    if req.mode == "design":
        if not (req.instruct or "").strip():
            raise HTTPException(400, "Describe the voice you want")
        return model.generate_voice_design_streaming(instruct=req.instruct, **common)
    raise HTTPException(400, "Unknown mode")


@app.post("/api/generate")
async def generate(req: GenReq):
    model = _state["model"]
    if model is None:
        raise HTTPException(409, "No model loaded")
    if not req.text.strip():
        raise HTTPException(400, "Type some text first")
    if MODE_FOR_TYPE.get(_state["model_type"]) != req.mode:
        raise HTTPException(400, f"Loaded model ({_state['model_type']}) can't do '{req.mode}'")
    if not _gen_lock.acquire(blocking=False):
        raise HTTPException(409, "Another generation is running")

    gen_id = uuid.uuid4().hex[:12]
    flag = req.client_id or gen_id
    _stop_flags[flag] = False
    try:
        stream = _make_stream(model, req)
        t0 = time.perf_counter()
        first = await run_in_threadpool(next, stream, None)  # raises early on bad args
    except BaseException:
        _stop_flags.pop(flag, None)
        _gen_lock.release()
        raise
    if first is None:
        _stop_flags.pop(flag, None)
        _gen_lock.release()
        raise HTTPException(500, "Model produced no audio")
    sr = int(first[1])
    ttfa = time.perf_counter() - t0

    def pcm(chunk: np.ndarray) -> bytes:
        return (np.clip(chunk.astype(np.float32), -1, 1) * 32767).astype("<i2").tobytes()

    def body():
        chunks = [first[0]]
        stopped = False
        try:
            yield pcm(first[0])
            for chunk, _sr, _timing in stream:
                if _stop_flags.get(flag):
                    stopped = True
                    break
                chunks.append(chunk)
                yield pcm(chunk)
        finally:
            total = time.perf_counter() - t0
            try:
                stream.close()
            except Exception:
                pass
            _stop_flags.pop(flag, None)
            _gen_lock.release()
            audio = np.concatenate(chunks).astype(np.float32)
            dur = len(audio) / sr
            fname = f"{time.strftime('%Y%m%d_%H%M%S')}_{gen_id[:6]}.wav"
            sf.write(OUTPUT_DIR / fname, audio, sr)
            entry = {
                "id": gen_id, "file": fname, "text": req.text, "mode": req.mode, "model": _state["model_id"],
                "language": req.language, "speaker": req.speaker, "instruct": req.instruct,
                "voice": req.voice_id, "duration": round(dur, 2), "ttfa": round(ttfa, 3),
                "total": round(total, 2), "rtf": round(dur / total, 2) if total else 0,
                "stopped": stopped, "created": time.time(),
            }
            with _file_lock:
                hist = _read_json(HISTORY_FILE, [])
                hist.insert(0, entry)
                _write_json(HISTORY_FILE, hist[:500])

    return StreamingResponse(
        body(), media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(sr), "X-Gen-Id": gen_id,
                 "Access-Control-Expose-Headers": "X-Sample-Rate, X-Gen-Id"},
    )


@app.post("/api/stop/{client_id}")
def stop(client_id: str):
    if client_id in _stop_flags:
        _stop_flags[client_id] = True
    return {"status": "ok"}


# ---------------------------------------------------------------- history
@app.get("/api/history")
def history():
    return [h for h in _read_json(HISTORY_FILE, []) if (OUTPUT_DIR / h["file"]).exists()]


@app.get("/api/history/{hid}/audio")
def history_audio(hid: str, download: bool = False, format: str = "wav"):
    for h in _read_json(HISTORY_FILE, []):
        if h["id"] == hid and (OUTPUT_DIR / h["file"]).exists():
            if format == "mp3":
                audio, sr = sf.read(OUTPUT_DIR / h["file"], dtype="float32")
                buf = io.BytesIO()
                sf.write(buf, audio, sr, format="MP3")
                name = Path(h["file"]).with_suffix(".mp3").name
                return Response(buf.getvalue(), media_type="audio/mpeg",
                                headers={"Content-Disposition": f'attachment; filename="{name}"'})
            return FileResponse(OUTPUT_DIR / h["file"], media_type="audio/wav",
                                filename=h["file"] if download else None)
    raise HTTPException(404)


@app.delete("/api/history/{hid}")
def delete_history(hid: str):
    with _file_lock:
        hist = _read_json(HISTORY_FILE, [])
        for h in hist:
            if h["id"] == hid:
                (OUTPUT_DIR / h["file"]).unlink(missing_ok=True)
        _write_json(HISTORY_FILE, [h for h in hist if h["id"] != hid])
    return {"status": "deleted"}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7870)
    ap.add_argument("--model", help="Preload this model id at startup")
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if args.model:
        _state["loading"] = args.model
        threading.Thread(target=_do_load, args=(args.model, args.dtype), daemon=True).start()
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Faster Qwen3-TTS GUI running at {url}  (Ctrl+C to quit)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
