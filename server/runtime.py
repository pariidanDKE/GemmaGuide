from __future__ import annotations

import os
import re

from openai import OpenAI


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")


def strip_markdown(text: str) -> str:
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"`{1,3}.*?`{1,3}", "", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.strip()


def patch_httpx_localhost_verify() -> None:
    import httpx

    orig_get = httpx.get

    def _httpx_no_verify_localhost(url, **kwargs):
        if isinstance(url, str) and any(host in url for host in ("localhost", "127.0.0.1", "0.0.0.0")):
            kwargs["verify"] = False
        return orig_get(url, **kwargs)

    httpx.get = _httpx_no_verify_localhost


def create_vllm_client() -> OpenAI:
    return OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)
