"""Blind-first FastAPI server bootstrap."""
from __future__ import annotations

import logging
import os

import uvicorn

from server.app_factory import create_app
from server.runtime import patch_httpx_localhost_verify


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = create_app()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _run_server() -> None:
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "7862"))
    ssl_certfile = os.getenv("APP_SSL_CERTFILE")
    ssl_keyfile = os.getenv("APP_SSL_KEYFILE")

    if not ssl_certfile:
        ssl_certfile = "/home/dan-parii/ml-workstation.tail9deb72.ts.net.crt"
    if not ssl_keyfile:
        ssl_keyfile = "/home/dan-parii/ml-workstation.tail9deb72.ts.net.key"
    if _env_flag("APP_DISABLE_SSL"):
        ssl_certfile = None
        ssl_keyfile = None

    logger.info(
        "Starting blind app on %s:%s ssl=%s",
        host,
        port,
        bool(ssl_certfile and ssl_keyfile),
    )

    uvicorn.run(
        app,
        host=host,
        port=port,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )


if __name__ == "__main__":
    patch_httpx_localhost_verify()
    _run_server()
