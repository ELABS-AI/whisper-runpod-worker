# elabs / Whisper STT

[![Run on RunPod](https://runpod.io/badge/runpod-hub)](https://runpod.io/console/hub)

OpenAI **Whisper large-v3** speech-to-text. Supports 100+ languages, word-level timestamps, and multiple output formats (plain text, SRT, VTT, JSON). Runs on any GPU with ≥4GB VRAM.

## Highlights

- **100+ languages** — multilingual transcription with auto-detect
- **Word-level timestamps** — precise word alignment for subtitling
- **Multiple output formats** — text, JSON (with segments), SRT, VTT
- **Configurable model size** — base (fast) to large-v3 (accuracy)
- **GPU efficient** — runs on T4, L4, RTX 4090, and any GPU with ≥4GB VRAM

## API

### Input

```json
{
  "input": {
    "audio_base64": "<base64-encoded WAV/MP3/FLAC/OGG bytes>",
    "language": "en",
    "response_format": "json",
    "temperature": 0.0,
    "word_timestamps": false,
    "model": "large-v3"
  }
}
```

### Output

```json
{
  "text": "The transcribed text content goes here.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "The transcribed text"
    }
  ],
  "language": "en",
  "wall_time_s": 1.2
}
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio_base64` | string | **required** | Base64-encoded audio bytes (WAV, MP3, FLAC, OGG) |
| `language` | string | `null` | Language code (`en`, `fr`, `de`, etc.) or `null` for auto-detect |
| `response_format` | string | `"json"` | Output format: `text`, `json`, `srt`, `vtt` |
| `temperature` | float | `0.0` | Sampling temperature (0.0 = greedy/ deterministic) |
| `word_timestamps` | bool | `false` | Include word-level timestamps in segments |
| `model` | string | `"large-v3"` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` |

## GPU Requirements

- **Recommended**: RTX 4090 / RTX 6000 Ada / L40S
- **Minimum**: Any GPU with ≥4GB VRAM (T4, L4, RTX 3080, A5000, etc.)
- **CUDA**: 12.0+

## Benchmark

| GPU | Model | Audio Duration | Wall Time |
|---|---|---|---|
| RTX 4090 | large-v3 | 60s | ~3.5s |
| RTX 4090 | base | 60s | ~0.8s |
| T4 | large-v3 | 60s | ~12s |
| T4 | base | 60s | ~2.0s |

## License

Apache-2.0 — OpenAI Whisper (MIT licensed model weights).
