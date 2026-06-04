# elabs / Whisper STT

[![Deploy on RunPod](https://img.shields.io/badge/RunPod-Deploy-orange?logo=runpod)](https://console.runpod.io/hub)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green)](https://developer.nvidia.com/cuda-toolkit)
[![MIT](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/licenses/MIT)

OpenAI **Whisper large-v3** speech-to-text. Transcribe audio in 100+ languages with word-level timestamps and multiple output formats (text, SRT, VTT, JSON).

![Whisper STT](https://pub-796a08821c1c483aaf5e274e0d03e350.r2.dev/hub-icons/whisper.svg)

## Highlights

- Whisper large-v3 -- OpenAI's best transcription model
- 100+ languages -- automatic language detection
- Word-level timestamps -- precise timing for captions
- Multiple output formats -- text, SRT, VTT, JSON
- URL or base64 input -- accepts MP3, WAV, M4A, FLAC, OGG

## Quick Start

```bash
curl -X POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"audio_url": "https://example.com/audio.mp3", "language": "en"}}'
```

## API

### Input (URL)

```json
{
  "input": {
    "audio_url": "https://example.com/audio.mp3",
    "language": "en",
    "response_format": "text",
    "temperature": 0.0
  }
}
```

### Input (base64)

```json
{
  "input": {
    "audio_base64": "<base64 encoded audio>",
    "language": "auto",
    "response_format": "json"
  }
}
```

### Output (text format)

```json
{
  "text": "Transcribed text content here.",
  "language": "en",
  "wall_time_s": 3.2
}
```

### Output (json format)

```json
{
  "text": "Full transcription text.",
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.5, "text": "First sentence."},
    {"id": 1, "start": 2.5, "end": 5.0, "text": "Second sentence."}
  ],
  "language": "en",
  "wall_time_s": 3.2
}
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio_url` | string | optional | URL to audio file |
| `audio_base64` | string | optional | Base64-encoded audio |
| `language` | string | `"auto"` | Language code or "auto" |
| `response_format` | string | `"text"` | "text", "json", "srt", "vtt" |
| `temperature` | float | `0.0` | Sampling temperature |
| `whisper_model` | string | `"large-v3"` | "base", "small", "medium", "large-v3" |

## GPU Requirements

- Minimum: >=4GB VRAM
- Recommended: RTX 4090, L4, T4 (>=8GB for large-v3)
- CUDA: 12.4+

## Benchmarks

| GPU | 60s audio | 10min audio |
|---|---|---|
| RTX 4090 | ~5s | ~45s |
| L4 | ~8s | ~70s |
| T4 | ~15s | ~130s |

## License

MIT. Based on [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3).
