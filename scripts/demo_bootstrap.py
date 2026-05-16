#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import signal
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


PUBLIC_URL_RE = re.compile(r"https://[-a-zA-Z0-9.]+trycloudflare\.com")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_log_dir() -> Path:
    return Path(os.getenv("DEMO_BOOTSTRAP_LOG_DIR", "/tmp/gemma_demo_logs"))


def _default_state_path() -> Path:
    return _default_log_dir() / "demo_state.json"


def _is_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(pid: int, timeout: float = 20.0) -> str:
    if not _is_alive(pid):
        return "not_running"

    os.killpg(pid, signal.SIGTERM)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _is_alive(pid):
            return "terminated"
        time.sleep(0.5)

    os.killpg(pid, signal.SIGKILL)
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if not _is_alive(pid):
            return "killed"
        time.sleep(0.2)
    return "unknown"


def _tail_text(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _emit(message: str) -> None:
    print(message, flush=True)


def _read_new_text(path: Path, offset: int) -> tuple[str, int]:
    if not path.exists():
        return "", offset
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        handle.seek(offset)
        text = handle.read()
        return text, handle.tell()


def _emit_log_chunk(service_name: str, text: str) -> None:
    for line in text.splitlines():
        _emit(f"[{service_name}] {line}")


def _wait_for_url(
    url: str,
    *,
    timeout_s: float,
    interval_s: float,
    accept_status: set[int] | None = None,
    pid: int | None = None,
    service_name: str | None = None,
    log_path: Path | None = None,
    verbose: bool = False,
) -> None:
    deadline = time.time() + timeout_s
    started_at = time.time()
    last_status_at = 0.0
    log_offset = 0
    while time.time() < deadline:
        if verbose and log_path:
            new_text, log_offset = _read_new_text(log_path, log_offset)
            if new_text:
                _emit_log_chunk(service_name or "service", new_text)
        if pid is not None and not _is_alive(pid):
            details = f"{service_name or 'service'} exited before becoming ready"
            if log_path:
                log_tail = _tail_text(log_path)
                if log_tail:
                    details += f". Log tail from {log_path}:\n{log_tail}"
            raise RuntimeError(details)
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                if accept_status is None or status in accept_status:
                    if verbose and log_path:
                        new_text, log_offset = _read_new_text(log_path, log_offset)
                        if new_text:
                            _emit_log_chunk(service_name or "service", new_text)
                    return
        except Exception:
            pass
        now = time.time()
        if now - last_status_at >= 10.0:
            elapsed = int(now - started_at)
            _emit(f"Waiting for {service_name or url} ({elapsed}s elapsed)...")
            last_status_at = now
        time.sleep(interval_s)
    if verbose and log_path:
        new_text, log_offset = _read_new_text(log_path, log_offset)
        if new_text:
            _emit_log_chunk(service_name or "service", new_text)
    details = f"Timed out waiting for {url}"
    if log_path:
        log_tail = _tail_text(log_path)
        if log_tail:
            details += f". Log tail from {log_path}:\n{log_tail}"
    raise TimeoutError(details)


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def _download_cloudflared(binary_path: Path) -> None:
    system = platform.system()
    machine = platform.machine().lower()
    if system != "Linux" or machine not in {"x86_64", "amd64"}:
        raise RuntimeError(f"cloudflared helper expects Linux x86_64, got {system} {machine}")
    if binary_path.exists():
        return
    _download_file(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        binary_path,
    )
    binary_path.chmod(binary_path.stat().st_mode | 0o111)


def _spawn_process(
    *,
    cmd: list[str],
    env: dict[str, str],
    cwd: Path,
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc.pid


def _load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _write_state(state_path: Path, payload: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _service_status(pid: int | None, log_path: str | None = None) -> dict[str, Any]:
    path = Path(log_path) if log_path else None
    return {
        "pid": pid,
        "running": _is_alive(pid),
        "log_path": log_path,
        "log_tail": _tail_text(path) if path else "",
    }


def _ensure_not_running(state_path: Path) -> None:
    state = _load_state(state_path)
    if not state:
        return
    services = state.get("services", {})
    running = [
        name
        for name, info in services.items()
        if _is_alive(info.get("pid"))
    ]
    if running:
        raise RuntimeError(
            "Existing demo services are still running: "
            + ", ".join(sorted(running))
            + f". Stop them first with: {sys.executable} scripts/demo_bootstrap.py stop"
        )


def _parse_tunnel_url(log_path: Path, timeout_s: float, *, verbose: bool = False) -> str:
    deadline = time.time() + timeout_s
    started_at = time.time()
    last_status_at = 0.0
    log_offset = 0
    while time.time() < deadline:
        if verbose:
            new_text, log_offset = _read_new_text(log_path, log_offset)
            if new_text:
                _emit_log_chunk("cloudflared", new_text)
        match = PUBLIC_URL_RE.search(_tail_text(log_path, max_chars=12000))
        if match:
            if verbose:
                new_text, log_offset = _read_new_text(log_path, log_offset)
                if new_text:
                    _emit_log_chunk("cloudflared", new_text)
            return match.group(0)
        now = time.time()
        if now - last_status_at >= 10.0:
            elapsed = int(now - started_at)
            _emit(f"Waiting for cloudflared public URL ({elapsed}s elapsed)...")
            last_status_at = now
        time.sleep(1.0)
    if verbose:
        new_text, log_offset = _read_new_text(log_path, log_offset)
        if new_text:
            _emit_log_chunk("cloudflared", new_text)
    raise TimeoutError("Timed out waiting for cloudflared to print a public URL")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start or stop the demo stack.")
    parser.add_argument("--state-path", default=str(_default_state_path()))

    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Start vLLM, the app, and optionally the tunnel.")
    start.add_argument("--repo-root", default=str(_repo_root()))
    start.add_argument("--log-dir", default=str(_default_log_dir()))
    start.add_argument("--with-tunnel", action="store_true")
    start.add_argument("--app-host", default="127.0.0.1")
    start.add_argument("--app-port", type=int, default=7862)
    start.add_argument("--vllm-port", type=int, default=8000)
    start.add_argument("--model", default=os.getenv("MODEL", "google/gemma-4-E4B-it"))
    start.add_argument("--vllm-served-name", default=os.getenv("VLLM_SERVED_NAME", "gemma-4-e4b-it"))
    start.add_argument("--quantization", default=os.getenv("VLLM_QUANTIZATION", "fp8"))
    start.add_argument("--max-model-len", default=os.getenv("MAX_MODEL_LEN", "8192"))
    start.add_argument("--gpu-mem-util", default=os.getenv("GPU_MEM_UTIL", "0.60"))
    start.add_argument("--max-num-seqs", default=os.getenv("MAX_NUM_SEQS", "1"))
    start.add_argument("--tensor-parallel", default=os.getenv("TENSOR_PARALLEL", "1"))
    start.add_argument("--max-soft-tokens", default=os.getenv("MAX_SOFT_TOKENS", "560"))
    start.add_argument("--tips-short-side", default=os.getenv("SPATIALSENSE_TIPSV2_SHORT_SIDE", "672"))
    start.add_argument("--vllm-extra-args", default=os.getenv("VLLM_EXTRA_ARGS", ""))
    start.add_argument("--dtype", default=os.getenv("VLLM_DTYPE", ""))
    start.add_argument("--vllm-timeout", type=float, default=900.0)
    start.add_argument("--app-timeout", type=float, default=180.0)
    start.add_argument("--tunnel-timeout", type=float, default=120.0)
    start.add_argument("--verbose", action="store_true", help="Print startup progress and stream new log lines while waiting.")

    subparsers.add_parser("status", help="Print current service status as JSON.")
    subparsers.add_parser("stop", help="Stop any services recorded in the state file.")
    return parser


def _start(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    log_dir = Path(args.log_dir).resolve()
    state_path = Path(args.state_path).resolve()
    _ensure_not_running(state_path)

    log_dir.mkdir(parents=True, exist_ok=True)
    vllm_log = log_dir / "vllm.log"
    app_log = log_dir / "app.log"
    tunnel_log = log_dir / "cloudflared.log"
    cloudflared_path = log_dir / "cloudflared"

    state: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(repo_root),
        "log_dir": str(log_dir),
        "state_path": str(state_path),
        "config": {
            "app_host": args.app_host,
            "app_port": args.app_port,
            "vllm_port": args.vllm_port,
            "model": args.model,
            "vllm_served_name": args.vllm_served_name,
            "quantization": args.quantization,
            "max_model_len": args.max_model_len,
            "gpu_mem_util": args.gpu_mem_util,
            "max_num_seqs": args.max_num_seqs,
            "tensor_parallel": args.tensor_parallel,
            "max_soft_tokens": args.max_soft_tokens,
            "tips_short_side": args.tips_short_side,
            "vllm_extra_args": args.vllm_extra_args,
            "dtype": args.dtype,
            "with_tunnel": args.with_tunnel,
        },
        "services": {},
        "urls": {
            "vllm_health": f"http://127.0.0.1:{args.vllm_port}/health",
            "vllm_models": f"http://127.0.0.1:{args.vllm_port}/v1/models",
            "app_local": f"http://127.0.0.1:{args.app_port}/",
            "public": None,
        },
    }
    _write_state(state_path, state)

    vllm_env = os.environ.copy()
    vllm_env.update(
        {
            "PORT": str(args.vllm_port),
            "MODEL": args.model,
            "SERVED_NAME": args.vllm_served_name,
            "VLLM_QUANTIZATION": args.quantization,
            "MAX_MODEL_LEN": str(args.max_model_len),
            "GPU_MEM_UTIL": str(args.gpu_mem_util),
            "MAX_NUM_SEQS": str(args.max_num_seqs),
            "TENSOR_PARALLEL": str(args.tensor_parallel),
            "MAX_SOFT_TOKENS": str(args.max_soft_tokens),
            "VLLM_EXTRA_ARGS": args.vllm_extra_args,
            "VLLM_DTYPE": args.dtype,
        }
    )
    if "VLLM_BIN" not in vllm_env:
        detected_vllm = shutil.which("vllm")
        if not detected_vllm:
            raise RuntimeError("Could not find `vllm` on PATH. Install it before starting the demo stack.")
        vllm_env["VLLM_BIN"] = detected_vllm

    app_env = os.environ.copy()
    app_env.update(
        {
            "APP_HOST": args.app_host,
            "APP_PORT": str(args.app_port),
            "APP_DISABLE_SSL": "1",
            "VLLM_BASE_URL": f"http://127.0.0.1:{args.vllm_port}/v1",
            "VLLM_MODEL_ID": args.vllm_served_name,
            "SPATIALSENSE_TIPSV2_SHORT_SIDE": str(args.tips_short_side),
        }
    )

    launched: list[int] = []
    try:
        _emit(f"Starting vllm with model {args.model} on port {args.vllm_port}...")
        vllm_pid = _spawn_process(
            cmd=["bash", "scripts/start_gemma4.sh"],
            env=vllm_env,
            cwd=repo_root,
            log_path=vllm_log,
        )
        _emit(f"vllm pid={vllm_pid}; log={vllm_log}")
        launched.append(vllm_pid)
        state["services"]["vllm"] = {"pid": vllm_pid, "log_path": str(vllm_log)}
        _write_state(state_path, state)
        _emit(f"Waiting for vllm health at {state['urls']['vllm_health']}")
        _wait_for_url(
            state["urls"]["vllm_health"],
            timeout_s=args.vllm_timeout,
            interval_s=5.0,
            accept_status={200},
            pid=vllm_pid,
            service_name="vllm",
            log_path=vllm_log,
            verbose=args.verbose,
        )
        _emit("vllm is ready")

        _emit(f"Starting app on {args.app_host}:{args.app_port}...")
        app_pid = _spawn_process(
            cmd=[sys.executable, "app.py"],
            env=app_env,
            cwd=repo_root,
            log_path=app_log,
        )
        _emit(f"app pid={app_pid}; log={app_log}")
        launched.append(app_pid)
        state["services"]["app"] = {"pid": app_pid, "log_path": str(app_log)}
        _write_state(state_path, state)
        _emit(f"Waiting for app health at {state['urls']['app_local']}")
        _wait_for_url(
            state["urls"]["app_local"],
            timeout_s=args.app_timeout,
            interval_s=2.0,
            accept_status={200},
            pid=app_pid,
            service_name="app",
            log_path=app_log,
            verbose=args.verbose,
        )
        _emit("app is ready")

        if args.with_tunnel:
            _emit("Starting cloudflared tunnel...")
            _download_cloudflared(cloudflared_path)
            tunnel_pid = _spawn_process(
                cmd=[str(cloudflared_path), "tunnel", "--url", state["urls"]["app_local"]],
                env=os.environ.copy(),
                cwd=repo_root,
                log_path=tunnel_log,
            )
            _emit(f"cloudflared pid={tunnel_pid}; log={tunnel_log}")
            launched.append(tunnel_pid)
            state["services"]["cloudflared"] = {"pid": tunnel_pid, "log_path": str(tunnel_log)}
            _write_state(state_path, state)
            state["urls"]["public"] = _parse_tunnel_url(tunnel_log, args.tunnel_timeout, verbose=args.verbose)
            _write_state(state_path, state)
            _emit(f"cloudflared public URL: {state['urls']['public']}")

        _emit("Startup complete")
        print(json.dumps(_status_payload(state_path), indent=2, sort_keys=True))
        return 0
    except Exception:
        for pid in reversed(launched):
            try:
                _terminate_process_group(pid)
            except Exception:
                pass
        raise


def _status_payload(state_path: Path) -> dict[str, Any]:
    state = _load_state(state_path)
    if not state:
        return {
            "state_path": str(state_path),
            "services": {},
            "urls": {},
            "running": False,
        }

    services = {
        name: _service_status(info.get("pid"), info.get("log_path"))
        for name, info in state.get("services", {}).items()
    }
    return {
        "state_path": str(state_path),
        "started_at": state.get("started_at"),
        "repo_root": state.get("repo_root"),
        "log_dir": state.get("log_dir"),
        "config": state.get("config", {}),
        "services": services,
        "urls": state.get("urls", {}),
        "running": any(info["running"] for info in services.values()),
    }


def _status(args: argparse.Namespace) -> int:
    state_path = Path(args.state_path).resolve()
    print(json.dumps(_status_payload(state_path), indent=2, sort_keys=True))
    return 0


def _stop(args: argparse.Namespace) -> int:
    state_path = Path(args.state_path).resolve()
    state = _load_state(state_path)
    if not state:
        print(json.dumps({"state_path": str(state_path), "stopped": {}}, indent=2, sort_keys=True))
        return 0

    stopped: dict[str, str] = {}
    for name in ("cloudflared", "app", "vllm"):
        info = state.get("services", {}).get(name)
        if not info:
            continue
        pid = info.get("pid")
        stopped[name] = _terminate_process_group(pid) if pid else "missing_pid"

    print(json.dumps({"state_path": str(state_path), "stopped": stopped}, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "start":
        return _start(args)
    if args.command == "status":
        return _status(args)
    if args.command == "stop":
        return _stop(args)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
