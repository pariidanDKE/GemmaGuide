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

## 1. Start the vLLM Server (Gemma 4 4B-IT)

```bash
bash scripts/start_gemma4.sh
```

The script uses the pyenv virtualenv named `vllm`. Override the binary path if needed:

```bash
VLLM_BIN=/path/to/vllm bash scripts/start_gemma4.sh
```

Verify it's up: `curl http://localhost:8000/v1/models`

Expected: JSON response listing `gemma-4-e4b-it`.

## 2. Load TIPSv2

TIPSv2 b14-dpt weights download from HuggingFace automatically on first import (`google/tipsv2-b14-dpt`). No manual download needed.

After both models are loaded, monitor VRAM: `watch -n2 nvidia-smi`  
Expected combined allocation: ~13GB on RTX 3090. If OOM occurs on T4, reduce TIPSv2 input resolution in `pipeline/tips_runner.py` from 896px to 672px short side.

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

**VRAM OOM on T4**: Reduce TIPSv2 input size:
```python
# pipeline/tips_runner.py
TIPSV2_SHORT_SIDE = 672  # reduce from 896 if OOM on T4
```

**Audio input not working via vLLM**: If direct audio fails (vLLM audio support not yet validated on this path as of Phase 0), the pipeline falls back to Whisper base for STT. Set `AUDIO_FALLBACK=whisper` in your environment and install `openai-whisper`:
```bash
pip install openai-whisper
```

**Gemma tool calling errors**: If Gemma does not call tools correctly, check that `--enable-auto-tool-choice` and `--tool-call-parser gemma` flags are set on the vLLM server.

## Hardware Targets

| Hardware | VRAM | Status |
|----------|------|--------|
| Colab T4 | 16GB | Baseline target — validate in Phase 1 item 1.5 |
| RTX 3090 | 24GB | Confirmed working in Phase 0 (~13GB combined) |
| Vast.ai RTX 3090 | 24GB | Fallback cloud option |
