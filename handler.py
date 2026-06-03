"""
RunPod serverless handler for OpenAI Whisper large-v3 — audio → text transcription.

Architecture:
  - Whisper large-v3 (multilingual, 100+ languages)
  - Supports word-level timestamps, diarization hints, multiple output formats
  - Runs on any GPU with >=4GB VRAM (T4, L4, RTX 4090, etc.)
  - ~30s real-time factor on a T4 for large-v3

Environment (set by RunPod template):
  - RUNPOD_POD_ID       — auto
  - RUNPOD_AI_API_KEY   — auto
  - WHISPER_MODEL       — model size (default: "large-v3")

Input schema (via RunPod serverless job):
  {
    "input": {
      "audio_base64": "<base64-encoded audio bytes>",  // REQUIRED — audio file bytes
      "language": "en",                                // optional — language code (auto-detect if omitted)
      "response_format": "json",                       // optional — "text", "json", "srt", "vtt"
      "temperature": 0.0,                              // optional — sampling temperature (0.0 = greedy)
      "word_timestamps": false,                        // optional — word-level timestamps
      "model": "large-v3"                              // optional — model size override
    }
  }

Output:
  {
    "text": "transcribed text",
    "segments": [...],
    "language": "en",
    "wall_time_s": 1.2
  }
"""

import base64
import io
import os
import time
import traceback

# ── Environment setup ─────────────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import whisper
import numpy as np

# ── Model path / size (can be overridden per job) ────────────────────────────
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")

# ── Global model (loaded once, reused across jobs) ───────────────────────────
_model = None
_model_name = None


def load_model(model_size: str = DEFAULT_MODEL):
    """Load Whisper model once and cache globally. Reloads if model size changes."""
    global _model, _model_name
    if _model is not None and _model_name == model_size:
        return _model

    print(f"[Cold Start] Loading Whisper {model_size} model...", flush=True)
    t0 = time.time()

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {_device}", flush=True)

    _model = whisper.load_model(model_size, device=_device)
    _model_name = model_size

    print(f"[Cold Start] Model ready in {time.time() - t0:.1f}s", flush=True)
    return _model


def decode_audio(audio_base64: str) -> np.ndarray:
    """Decode base64 audio bytes into a numpy array (float32, mono)."""
    audio_bytes = base64.b64decode(audio_base64)

    # Use ffmpeg via whisper's built-in audio decoding
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        audio = whisper.load_audio(tmp_path)
        audio = whisper.pad_or_trim(audio)
    finally:
        os.unlink(tmp_path)

    return audio


def run_inference(
    audio_base64: str,
    language: str | None = None,
    response_format: str = "json",
    temperature: float = 0.0,
    word_timestamps: bool = False,
    model_size: str = DEFAULT_MODEL,
) -> dict:
    """
    Run Whisper transcription.
    Returns dict with text, segments, language, wall_time_s.
    """
    model = load_model(model_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Decode audio
    print(f"[Inference] Decoding audio ({len(audio_base64)} base64 chars)...", flush=True)
    t_start = time.time()

    audio = decode_audio(audio_base64)

    decode_time = time.time() - t_start
    print(f"[Inference] Audio decoded in {decode_time:.1f}s, shape={audio.shape}", flush=True)

    # Prepare transcription options
    decode_options = {
        "language": language,
        "temperature": temperature,
        "word_timestamps": word_timestamps,
    }

    # Remove None values
    decode_options = {k: v for k, v in decode_options.items() if v is not None}

    print(f"[Inference] Transcribing with options: {decode_options}", flush=True)

    t_transcribe = time.time()

    with torch.inference_mode():
        result = model.transcribe(
            audio,
            **decode_options,
        )

    transcription_time = time.time() - t_transcribe
    wall_time = time.time() - t_start
    print(f"[Done] Transcription took {transcription_time:.1f}s (total: {wall_time:.1f}s)", flush=True)

    # Format output based on response_format
    text = result.get("text", "").strip()
    segments = result.get("segments", [])
    detected_language = result.get("language", language or "unknown")

    if response_format == "text":
        return {
            "text": text,
            "language": detected_language,
            "wall_time_s": round(wall_time, 1),
        }
    elif response_format == "srt":
        srt_output = _segments_to_srt(segments)
        return {
            "text": srt_output,
            "language": detected_language,
            "wall_time_s": round(wall_time, 1),
        }
    elif response_format == "vtt":
        vtt_output = _segments_to_vtt(segments)
        return {
            "text": vtt_output,
            "language": detected_language,
            "wall_time_s": round(wall_time, 1),
        }
    else:  # json
        # Serialize segments with timestamps
        serializable_segments = []
        for seg in segments:
            seg_dict = {
                "id": seg.get("id", 0),
                "start": round(seg.get("start", 0.0), 2),
                "end": round(seg.get("end", 0.0), 2),
                "text": seg.get("text", "").strip(),
            }
            if "words" in seg:
                seg_dict["words"] = [
                    {
                        "word": w.get("word", ""),
                        "start": round(w.get("start", 0.0), 2),
                        "end": round(w.get("end", 0.0), 2),
                        "probability": round(w.get("probability", 0.0), 3),
                    }
                    for w in seg["words"]
                ]
            serializable_segments.append(seg_dict)

        return {
            "text": text,
            "segments": serializable_segments,
            "language": detected_language,
            "wall_time_s": round(wall_time, 1),
        }


def _format_timestamp(seconds: float, fmt: str = "srt") -> str:
    """Format seconds to SRT or VTT timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)

    if fmt == "vtt":
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _segments_to_srt(segments: list) -> str:
    """Convert segments to SubRip (SRT) format."""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_timestamp(seg.get("start", 0.0), "srt")
        end = _format_timestamp(seg.get("end", 0.0), "srt")
        text = seg.get("text", "").strip()
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _segments_to_vtt(segments: list) -> str:
    """Convert segments to WebVTT format."""
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = _format_timestamp(seg.get("start", 0.0), "vtt")
        end = _format_timestamp(seg.get("end", 0.0), "vtt")
        text = seg.get("text", "").strip()
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# RunPod Serverless Handler
# ═══════════════════════════════════════════════════════════════════════════════


def handler(job):
    """
    RunPod serverless handler: base64 audio → transcription.
    Called once per job. The model stays loaded across jobs (global).
    """
    job_input = job.get("input", {})

    audio_base64 = job_input.get("audio_base64", "")
    if not audio_base64:
        return {"error": "Missing required field: audio_base64"}

    language = job_input.get("language", None)
    response_format = job_input.get("response_format", "json")
    temperature = float(job_input.get("temperature", 0.0))
    word_timestamps = bool(job_input.get("word_timestamps", False))
    model_size = str(job_input.get("model", DEFAULT_MODEL))

    # Validate response_format
    allowed_formats = ["text", "json", "srt", "vtt"]
    if response_format not in allowed_formats:
        return {
            "error": f"Invalid response_format: '{response_format}'. Allowed: {allowed_formats}"
        }

    try:
        result = run_inference(
            audio_base64=audio_base64,
            language=language,
            response_format=response_format,
            temperature=temperature,
            word_timestamps=word_timestamps,
            model_size=model_size,
        )
        return result

    except Exception as exc:
        traceback.print_exc()
        return {
            "error": f"Whisper transcription failed: {str(exc)}",
            "traceback": traceback.format_exc(),
        }


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
