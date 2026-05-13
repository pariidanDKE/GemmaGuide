# Quickstart: SpatialSense

**Date**: 2026-04-30

## Prerequisites

- NVIDIA GPU with ≥16GB VRAM (T4 baseline; RTX 3090 confirmed in Phase 0)
- CUDA 12.x, Python 3.11
- HuggingFace account — accept the Gemma license at `huggingface.co/google/gemma-4-4b-it` before downloading weights

## Install Dependencies

```bash
pip install vllm[audio] transformers torch torchvision \
            gradio gtts numpy pillow matplotlib openai
```

## 1. Start the vLLM Server

```bash
bash scripts/start_gemma4.sh
```

Current default profile is the known-good hackathon demo profile for T4-fit work:
- Gemma served as `google/gemma-4-E4B-it`
- served name: `gemma-4-e4b-it`
- quantization: `bitsandbytes`
- conservative single-sequence vLLM config

The script uses the pyenv virtualenv named `vllm`. Override the binary path if needed:

```bash
VLLM_BIN=/path/to/vllm bash scripts/start_gemma4.sh
```

Key memory knobs:

```bash
GPU_MEM_UTIL=0.4 MAX_MODEL_LEN=4096 MAX_NUM_SEQS=1 MAX_SOFT_TOKENS=280 bash scripts/start_gemma4.sh
```

Notes:
- `GPU_MEM_UTIL` must still be high enough for vLLM to allocate KV cache. If set too low, the engine fails to start.
- `MAX_MODEL_LEN` is the cleanest quality-preserving lever for reducing VRAM.
- `MAX_NUM_SEQS=1` should remain fixed for T4-class runs.
- `MAX_SOFT_TOKENS` reduces multimodal vision-token budget if more headroom is needed.

Verify it's up: `curl http://localhost:8000/v1/models`

Expected: JSON response listing `gemma-4-e4b-it`.

## 2. Load TIPSv2

TIPSv2 b14-dpt weights download from HuggingFace automatically on first import (`google/tipsv2-b14-dpt`). No manual download needed.

After both models are loaded, monitor VRAM: `watch -n2 nvidia-smi`  
Current default short-side is `672`, which is the T4-oriented setting. If more detail is needed on larger hardware, override with:

```bash
SPATIALSENSE_TIPSV2_SHORT_SIDE=896 python app_blind.py
```

If OOM still occurs on T4, reduce further to `448`.

## 3. Start the Demo

```bash
python app.py
```

Open the Gradio URL printed to the terminal (default: `http://localhost:7860`).

**Basic interaction**:
1. Upload a photo using the image panel
2. Record a question using the microphone button, or type in the text box
3. Click Submit
4. Read the response text and listen to the spoken answer

Expected end-to-end latency: 10–30 seconds on T4 16GB.

## 4. Run Tests

```bash
# Unit tests — no GPU required
pytest tests/unit/ -v

# Integration tests — requires GPU + running vLLM server (step 1 above)
pytest tests/integration/ -v --gpu
```

Integration tests use real model instances. Do not mock TIPSv2 or Gemma outputs — mock/live divergence has been a confirmed risk in this project.

## Known Issues and Workarounds

**VRAM OOM on T4**: Apply the following in order:
```bash
# 1. Keep single-sequence serving and lower context first
GPU_MEM_UTIL=0.4 MAX_MODEL_LEN=2048 MAX_NUM_SEQS=1 bash scripts/start_gemma4.sh

# 2. Lower TIPS input size if needed
SPATIALSENSE_TIPSV2_SHORT_SIDE=448 python app_blind.py

# 3. Lower multimodal token budget only if still needed
GPU_MEM_UTIL=0.4 MAX_MODEL_LEN=2048 MAX_NUM_SEQS=1 MAX_SOFT_TOKENS=140 bash scripts/start_gemma4.sh
```

Do not set `GPU_MEM_UTIL` arbitrarily low. If it is too small, vLLM may fail to start with `No available memory for the cache blocks`.

**Audio input not working via vLLM**: If direct audio fails (vLLM audio support not yet validated on this path as of Phase 0), the pipeline falls back to Whisper base for STT. Set `AUDIO_FALLBACK=whisper` in your environment and install `openai-whisper`:
```bash
pip install openai-whisper
```

**Gemma tool calling errors**: If Gemma does not call tools correctly, check that `--enable-auto-tool-choice` and `--tool-call-parser gemma` flags are set on the vLLM server.

## Hardware Targets

| Hardware | VRAM | Status |
|----------|------|--------|
| Colab T4 | 16GB | Baseline target — known-good demo profile is E4B + bitsandbytes + conservative vLLM + TIPS short-side 672 |
| RTX 3090 | 24GB | Confirmed working in Phase 0 (~13GB combined) |
| Vast.ai RTX 3090 | 24GB | Fallback cloud option |
