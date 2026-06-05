# elabs / Whisper STT

OpenAI Whisper large-v3 speech-to-text transcription. Supports 100+ languages with word-level timestamps.

[![Docker Build](https://github.com/ELABS-AI/whisper-runpod-worker/actions/workflows/build.yml/badge.svg)](https://github.com/ELABS-AI/whisper-runpod-worker/actions/workflows/build.yml)

---

## Quick Start

Deploy this worker on [RunPod Serverless](https://www.runpod.io/serverless) using the **Deploy on RunPod** button in the Hub, or manually with the Docker image:

```
ghcr.io/elabs-ai/whisper-runpod-worker:latest
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `openai/whisper-large-v3` | Whisper model variant (large-v3, medium, small) |
| `HF_HOME` | `/runpod-volume/models/huggingface` | HuggingFace cache directory |
| `HUGGINGFACE_HUB_CACHE` | `/runpod-volume/models/huggingface/hub` | HuggingFace hub cache |

> **Note:** `HF_HOME` and `HUGGINGFACE_HUB_CACHE` should point to a RunPod Network Volume mount path for model caching between runs.

---

## API Reference

### Input

```json
{"input": {"audio_b64": "<base64 audio>", "language": "en"}}
```

### Output

```json
{"text": "Transcribed text here.", "segments": [{"start": 0.0, "end": 2.5, "text": "..."}], "language": "en", "wall_time_s": 4.1}
```

---

## Usage Examples

### Python (runpod SDK)

```python
import runpod
import base64

client = runpod.AsyncioEndpointClient("whisper-runpod-worker")
result = await client.run({"input": {"audio_b64": "<base64 audio>", "language": "en"}})
print(result)
```

### cURL

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"audio_b64": "<base64 audio>", "language": "en"}}'

```

---

## GPU Requirements

RTX 3090+ (24GB VRAM) | ~2-5s for 30s audio | MIT license

---

## License

Apache 2.0 — See [LICENSE](LICENSE)

---

## Built by [E-Labs AI](https://www.elabsai.com)

Part of the E-Labs AI Studio serverless model fleet. Visit [elabsai.com](https://www.elabsai.com) to use these models in a hosted UI.
