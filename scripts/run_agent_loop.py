#!/usr/bin/env python3
from __future__ import annotations

# Usage examples:
#   Basic run:
#     python scripts/run_agent_loop.py \
#       --question "How far is the desk?"
#
#   Audio question from WAV file:
#     python scripts/run_agent_loop.py \
#       --audio data/media/audio/question.wav
#
#   Use a minimal system prompt override:
#     python scripts/run_agent_loop.py \
#       --question "How far is the desk?" \
#       --system-prompt "Call call_dpt_head once using structured tool_calls. No plain text."
#
#   Restrict tools to call_dpt_head only:
#     python scripts/run_agent_loop.py \
#       --question "How far is the desk?" \
#       --tool-mode dpt_only
#
#   Dump request/response JSONL for debugging:
#     python scripts/run_agent_loop.py \
#       --question "How far is the desk?" \
#       --dump-requests /tmp/spatialsense_req.jsonl \
#       --dump-responses /tmp/spatialsense_resp.jsonl
#
#   Show newly generated box debug images:
#     python scripts/run_agent_loop.py \
#       --question "How far is the desk?" \
#       --show-new-box-images
#
#   Print each tool input/output payload inline:
#     python scripts/run_agent_loop.py \
#       --question "How far is the desk?" \
#       --print-tool-io

import argparse
import logging
import os
from pathlib import Path
import sys
import warnings

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from pipeline.session import create_session
from server.agent import run_agent_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one SpatialSense agent-loop turn against a local vLLM server."
    )
    parser.add_argument(
        "--image",
        default="data/media/images/original_highres.jpeg",
        help="Path to input image (default: data/media/images/original_highres.jpeg)",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="User question text to run through the agent loop.",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Path to WAV audio question. If provided, audio is sent instead of text.",
    )
    parser.add_argument(
        "--tool-mode",
        choices=["full", "dpt_only"],
        default=None,
        help="Override SPATIALSENSE_TOOL_MODE for this run.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Override SPATIALSENSE_SYSTEM_PROMPT for this run.",
    )
    parser.add_argument(
        "--dump-requests",
        default=None,
        help="Path to write JSONL request dumps (SPATIALSENSE_DUMP_REQUEST_PATH).",
    )
    parser.add_argument(
        "--dump-responses",
        default=None,
        help="Path to write JSONL response dumps (SPATIALSENSE_DUMP_RESPONSE_PATH).",
    )
    parser.add_argument(
        "--show-new-box-images",
        action="store_true",
        help="Print newly created files in data/media/debug_boxes after the run.",
    )
    parser.add_argument(
        "--print-tool-io",
        action="store_true",
        help="Print each tool input and output payload during the agent loop.",
    )
    parser.add_argument(
        "--print-tool-returns",
        action="store_true",
        help="Deprecated alias for --print-tool-io.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level for latency and debug logs (default: INFO).",
    )
    return parser.parse_args()


def set_env_if(name: str, value: str | None) -> None:
    if value is not None:
        os.environ[name] = value


def _suppress_noisy_third_party_logs() -> None:
    # Keep SpatialSense logs visible while reducing noisy dependency output.
    for logger_name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "urllib3",
        "transformers",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    warnings.filterwarnings(
        "ignore",
        message="xFormers is not available.*",
        category=UserWarning,
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _suppress_noisy_third_party_logs()

    set_env_if("SPATIALSENSE_TOOL_MODE", args.tool_mode)
    set_env_if("SPATIALSENSE_SYSTEM_PROMPT", args.system_prompt)
    set_env_if("SPATIALSENSE_DUMP_REQUEST_PATH", args.dump_requests)
    set_env_if("SPATIALSENSE_DUMP_RESPONSE_PATH", args.dump_responses)
    if args.print_tool_io or args.print_tool_returns:
        os.environ["SPATIALSENSE_PRINT_TOOL_IO"] = "1"

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}")
        return 1

    audio_bytes: bytes | None = None
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"ERROR: audio not found: {audio_path}")
            return 1
        audio_bytes = audio_path.read_bytes()

    if audio_bytes is None and not args.question:
        print("ERROR: provide either --question or --audio")
        return 1

    debug_dir = Path("data/media/debug_boxes")
    before = set(p.name for p in debug_dir.glob("*.png")) if debug_dir.exists() else set()

    image = Image.open(image_path).convert("RGB")
    question: str | bytes = audio_bytes if audio_bytes is not None else args.question
    session = create_session(image, question)

    try:
        response_text = run_agent_loop(session)
    finally:
        session.release()

    print("AGENT_RESPONSE_START")
    print(response_text)
    print("AGENT_RESPONSE_END")

    if args.show_new_box_images:
        after = set(p.name for p in debug_dir.glob("*.png")) if debug_dir.exists() else set()
        new_files = sorted(after - before)
        print(f"NEW_DEBUG_BOX_FILES={len(new_files)}")
        for name in new_files:
            print(f"NEW_DEBUG_BOX_FILE={debug_dir / name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
