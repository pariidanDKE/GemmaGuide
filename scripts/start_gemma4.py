#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _resolve_vllm_bin() -> str:
    configured = os.getenv("VLLM_BIN")
    if configured:
        if os.path.isfile(configured) and os.access(configured, os.X_OK):
            return configured
        raise RuntimeError(
            f"vllm not found at {configured}. Set VLLM_BIN=/path/to/vllm to override."
        )

    detected = shutil.which("vllm")
    if detected:
        return detected

    fallback = os.path.expanduser("~/.pyenv/versions/3.12.0/envs/vllm/bin/vllm")
    if os.path.isfile(fallback) and os.access(fallback, os.X_OK):
        return fallback

    raise RuntimeError("Could not find `vllm` on PATH. Set VLLM_BIN=/path/to/vllm to override.")


def build_vllm_command() -> list[str]:
    model = _env("MODEL", "google/gemma-4-E4B-it")
    served_name = _env("SERVED_NAME", "gemma-4-e4b-it")
    port = _env("PORT", "8000")
    max_model_len = _env("MAX_MODEL_LEN", "8192")
    gpu_mem_util = _env("GPU_MEM_UTIL", "0.60")
    tensor_parallel = _env("TENSOR_PARALLEL", "1")
    max_num_seqs = _env("MAX_NUM_SEQS", "1")
    max_soft_tokens = _env("MAX_SOFT_TOKENS", "560")
    quantization = _env("VLLM_QUANTIZATION", "fp8").strip()
    dtype = _env("VLLM_DTYPE", "").strip()
    extra_args = shlex.split(os.getenv("VLLM_EXTRA_ARGS", ""))
    mm_limits = '{"image": 4, "audio": 5}'
    mm_processor_kwargs = f'{{"max_soft_tokens": {max_soft_tokens}}}'

    cmd = [
        _resolve_vllm_bin(),
        "serve",
        model,
        "--served-model-name",
        served_name,
        "--port",
        port,
        "--max-model-len",
        max_model_len,
        "--gpu-memory-utilization",
        gpu_mem_util,
        "--max-num-seqs",
        max_num_seqs,
        "--tensor-parallel-size",
        tensor_parallel,
        "--tool-call-parser",
        "gemma4",
        "--enable-auto-tool-choice",
        "--reasoning-parser",
        "gemma4",
        "--mm-processor-kwargs",
        mm_processor_kwargs,
        "--limit-mm-per-prompt",
        mm_limits,
        "--async-scheduling",
    ]

    if quantization.lower() not in {"", "none", "off"}:
        cmd.extend(["--quantization", quantization])

    if dtype:
        cmd.extend(["--dtype", dtype])

    cmd.extend(extra_args)
    return cmd


def main() -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    try:
        cmd = build_vllm_command()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Starting Gemma 4 server: {cmd[2]} on port {_env('PORT', '8000')}", flush=True)
    print(f"Using vllm binary: {cmd[0]}", flush=True)
    print(
        "Memory profile: "
        f"gpu_mem_util={_env('GPU_MEM_UTIL', '0.60')} "
        f"max_model_len={_env('MAX_MODEL_LEN', '8192')} "
        f"max_num_seqs={_env('MAX_NUM_SEQS', '1')} "
        f"max_soft_tokens={_env('MAX_SOFT_TOKENS', '560')}",
        flush=True,
    )
    print(f"Quantization: {_env('VLLM_QUANTIZATION', 'fp8') or '<unset>'}", flush=True)
    print(f"dtype: {_env('VLLM_DTYPE', '') or '<default>'}", flush=True)
    print(f"Extra args: {os.getenv('VLLM_EXTRA_ARGS', '') or '<none>'}", flush=True)

    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
