#!/bin/bash
set -e

# ============================================================================
# Gemma 4 vLLM Server
# ============================================================================
# Serves Gemma 4 via OpenAI-compatible API on port 8000.
# Optimised for vision requests + tool calling.
#
# Supported models (set MODEL below):
#   google/gemma-4-E2B-it   — effective 2B,  ~16GB VRAM
#   google/gemma-4-E4B-it   — effective 4B,  ~24GB VRAM
#   google/gemma-4-26b-it   — MoE 26B/4B,   ~80GB VRAM
#   google/gemma-4-31b-it   — dense 31B,    ~80GB VRAM (2×GPU)
#
# Endpoints once running:
#   POST http://localhost:8000/v1/chat/completions
#   GET  http://localhost:8000/health
#   GET  http://localhost:8000/v1/models
#
# Audio: requires vllm[audio] — pip install "vllm[audio]".
# ============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VENV_VLLM="${VLLM_BIN:-$HOME/.pyenv/versions/3.12.0/envs/vllm/bin/vllm}"

if [ ! -x "$VENV_VLLM" ]; then
    echo "ERROR: vllm not found at $VENV_VLLM"
    echo "Set VLLM_BIN=/path/to/vllm to override."
    exit 1
fi

MODEL="${MODEL:-google/gemma-4-E4B-it}"
SERVED_NAME="${SERVED_NAME:-gemma-4-e4b-it}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.60}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_SOFT_TOKENS="${MAX_SOFT_TOKENS:-560}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-fp8}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
MM_LIMITS='{"image": 4, "audio": 5}'

CMD=(
  "$VENV_VLLM" serve "$MODEL"
  --served-model-name "$SERVED_NAME"
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-num-seqs "$MAX_NUM_SEQS"
  --tensor-parallel-size "$TENSOR_PARALLEL"
  --tool-call-parser gemma4
  --enable-auto-tool-choice
  --reasoning-parser gemma4
  --mm-processor-kwargs "{\"max_soft_tokens\": $MAX_SOFT_TOKENS}"
  --limit-mm-per-prompt "$MM_LIMITS"
  --async-scheduling
)

case "${VLLM_QUANTIZATION,,}" in
  ""|none|off)
    ;;
  *)
    CMD+=(--quantization "$VLLM_QUANTIZATION")
    ;;
esac

if [ -n "$VLLM_EXTRA_ARGS" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( $VLLM_EXTRA_ARGS )
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Starting Gemma 4 server: $MODEL on port $PORT"
echo "Using vllm binary: $VENV_VLLM"
echo "Memory profile: gpu_mem_util=$GPU_MEM_UTIL max_model_len=$MAX_MODEL_LEN max_num_seqs=$MAX_NUM_SEQS max_soft_tokens=$MAX_SOFT_TOKENS"
echo "Quantization: ${VLLM_QUANTIZATION:-<unset>}"
echo "Extra args: ${VLLM_EXTRA_ARGS:-<none>}"

exec "${CMD[@]}"
