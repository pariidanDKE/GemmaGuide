"""
Text-to-speech synthesis for SpatialSense.

Default provider: gTTS (no API key required).
Fallback: Whisper-based STT is handled upstream in the app layer, not here.
  Set AUDIO_FALLBACK=whisper in your environment and install openai-whisper
  if direct audio-to-Gemma fails on vLLM — see quickstart.md Known Issues.
"""

from __future__ import annotations

import io
import os


def synthesize(text: str) -> bytes:
    """Convert text to WAV/MP3 audio bytes using gTTS."""
    from gtts import gTTS

    tts = gTTS(text=text, lang="en", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()
