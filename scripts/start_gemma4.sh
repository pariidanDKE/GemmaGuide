#!/bin/bash
set -e

# ============================================================================
# Gemma 4 vLLM Server
# ============================================================================
# Serves Gemma 4 via OpenAI-compatible API on port 8000.
# Optimised for vision requests + tool calling.
#
# Supported models (set MODEL below):
#   google/gemma-4-E2B-it   — effective 2B,  ~16GB VRAM  ← default
#   google/gemma-4-E4B-it   — effective 4B,  ~24GB VRAM
#   google/gemma-4-26b-it   — MoE 26B/4B,   ~80GB VRAM
#   google/gemma-4-31b-it   — dense 31B,    ~80GB VRAM (2×GPU)
#
# Endpoints once running:
#   POST http://localhost:8000/v1/chat/completions   — text + vision + tools
#   GET  http://localhost:8000/health
#   GET  http://localhost:8000/v1/models
#
# Vision usage (image_url content block):
#   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
#   {"type": "image_url", "image_url": {"url": "https://..."}}
#
# Tool calling: standard OpenAI tools schema — parser handles Gemma's protocol.
# Thinking mode: pass extra_body={"chat_template_kwargs": {"enable_thinking": true}}
#
# Audio: requires vllm[audio] — pip install "vllm[audio]" in the venv.
#
# venv: ~/.pyenv/versions/3.12.0/envs/vllm  (Python 3.12, vLLM 0.18.0)
#   Override: VLLM_BIN=/path/to/vllm bash scripts/start_gemma4.sh
# ============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Path to the vllm executable inside the "vllm" pyenv virtualenv.
# Override at runtime if your path differs: VLLM_BIN=/path/to/vllm bash scripts/start_gemma4.sh
VENV_VLLM="${VLLM_BIN:-$HOME/.pyenv/versions/3.12.0/envs/vllm/bin/vllm}"

if [ ! -x "$VENV_VLLM" ]; then
    echo "ERROR: vllm not found at $VENV_VLLM"
    echo "Set VLLM_BIN=/path/to/vllm to override."
    exit 1
fi

# ============================================================================
# Configuration
# ============================================================================
MODEL="${MODEL:-google/gemma-4-E4B-it}"
SERVED_NAME="${SERVED_NAME:-gemma-4-e4b-it}"
PORT="${PORT:-8000}"

# Gemma 4 supports up to 128k tokens, but KV cache scales with context length.
# For T4/shared-GPU setups, stay conservative by default.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# GPU memory fraction (0.0–1.0)
# Keep this lower when Gemma shares the GPU with TIPSv2.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"

# Tensor parallelism — set to 2+ for 31B on multi-GPU
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"

# Lower concurrency reduces KV cache pressure.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

# Vision token budget per image — valid values: 70 140 280 560 1120
# Higher = more detail, more memory; lower = faster
MAX_SOFT_TOKENS="${MAX_SOFT_TOKENS:-280}"

# Quantization mode should match the selected checkpoint or serving strategy.
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-bitsandbytes}"

# Max images / audio per request (audio requires vllm[audio])
MM_LIMITS='{"image": 4, "audio": 5}'

# ============================================= `   `===============================
# Start vLLM Server (foreground — Ctrl+C to stop)
# ============================================================================
echo "Starting Gemma 4 server: $MODEL on port $PORT"
echo "Using vllm binary: $VENV_VLLM"
echo "Memory profile: gpu_mem_util=$GPU_MEM_UTIL max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS max_soft_tokens=$MAX_SOFT_TOKENS"

exec "$VENV_VLLM" serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --quantization "$VLLM_QUANTIZATION" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --tensor-parallel-size "$TENSOR_PARALLEL" \
  --tool-call-parser gemma4 \
  --enable-auto-tool-choice \
  --reasoning-parser gemma4 \
  --mm-processor-kwargs "{\"max_soft_tokens\": $MAX_SOFT_TOKENS}" \
  --limit-mm-per-prompt "$MM_LIMITS" \
  --async-scheduling
