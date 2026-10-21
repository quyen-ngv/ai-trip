"""Speech-to-text for social videos: Groq Whisper, else OpenAI Whisper, else faster-whisper here.

The first provider with a key wins; local Whisper is the fallback when neither key is set. Each
provider call is reported to the AI call log as TRANSCRIBE_AUDIO.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from . import ai_log, config

logger = logging.getLogger(__name__)
_LOCAL_WHISPER = None
_LOCAL_LOCK = threading.Lock()


async def transcribe(audio: bytes, filename: str = "audio.mp3") -> tuple[str, str]:
    """(transcript, "provider:model"). Raises when a Whisper API answers with an error."""
    if config.GROQ_API_KEY:
        return await _api("groq", "https://api.groq.com/openai/v1/audio/transcriptions", config.GROQ_API_KEY,
                          config.GROQ_WHISPER_MODEL, audio, filename)
    if config.OPENAI_API_KEY:
        return await _api("openai", "https://api.openai.com/v1/audio/transcriptions", config.OPENAI_API_KEY,
                          config.OPENAI_WHISPER_MODEL, audio, filename)
    if config.LOCAL_WHISPER_ENABLED:
        return await asyncio.to_thread(_local, audio, filename)
    logger.warning("Audio transcription skipped: no GROQ_API_KEY, OPENAI_API_KEY, and LOCAL_WHISPER_ENABLED=false")
    return "", "skipped:no_transcriber"


async def _api(provider: str, url: str, key: str, model: str, audio: bytes, filename: str) -> tuple[str, str]:
    logger.info("Transcribing audio with %s: model=%s", provider, model)
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(url, headers={"Authorization": f"Bearer {key}"}, data={"model": model},
                                     files={"file": (filename, audio, "audio/mpeg")})
    body: Any = None
    try:
        body = response.json() if response.is_success else None
    except ValueError:
        pass
    await ai_log.report(operation="TRANSCRIBE_AUDIO", provider=provider, model=model, ok=response.is_success,
                        request={"model": model, "file": filename, "bytes": len(audio)}, response=body,
                        error=None if response.is_success else {"httpStatus": response.status_code,
                                                                "body": response.text[:2000]},
                        latency_ms=int((time.monotonic() - started) * 1000))
    response.raise_for_status()
    text = (body or {}).get("text", "").strip()
    logger.info("%s transcript complete: chars=%s", provider, len(text))
    return text, f"{provider}:{model}"


def warm_up() -> None:
    """Load local Whisper at startup when it is the transcriber jobs will use, so the first social
    video job does not wait for the model to load."""
    if config.GROQ_API_KEY or config.OPENAI_API_KEY or not config.LOCAL_WHISPER_ENABLED:
        return
    try:
        with _LOCAL_LOCK:
            _load_local()
    except Exception:
        logger.warning("Local Whisper warm-up failed; the first job will retry the load", exc_info=True)


def _load_local():
    global _LOCAL_WHISPER
    if _LOCAL_WHISPER is not None:
        return _LOCAL_WHISPER
    logger.info("Loading local Whisper model: model=%s device=%s compute_type=%s cache_dir=%s",
                config.LOCAL_WHISPER_MODEL, config.LOCAL_WHISPER_DEVICE, config.LOCAL_WHISPER_COMPUTE_TYPE,
                config.LOCAL_WHISPER_CACHE_DIR or "default")
    from faster_whisper import WhisperModel

    kwargs: dict[str, Any] = {"device": config.LOCAL_WHISPER_DEVICE, "compute_type": config.LOCAL_WHISPER_COMPUTE_TYPE,
                              "cpu_threads": config.LOCAL_WHISPER_CPU_THREADS}
    if config.LOCAL_WHISPER_CACHE_DIR:
        kwargs["download_root"] = config.LOCAL_WHISPER_CACHE_DIR
    _LOCAL_WHISPER = WhisperModel(config.LOCAL_WHISPER_MODEL, **kwargs)
    logger.info("Local Whisper model loaded")
    return _LOCAL_WHISPER


def _local(audio: bytes, filename: str) -> tuple[str, str]:
    # One transcription at a time: a second one would only split the same CPU quota.
    with _LOCAL_LOCK:
        try:
            model = _load_local()
            with tempfile.TemporaryDirectory(prefix="social_audio_") as tmp:
                path = Path(tmp) / (Path(filename).name or "audio.mp3")
                path.write_bytes(audio)
                logger.info("Transcribing audio with local faster-whisper: language=%s",
                            config.LOCAL_WHISPER_LANGUAGE or "auto")
                kwargs: dict[str, Any] = {"vad_filter": True, "beam_size": config.LOCAL_WHISPER_BEAM_SIZE}
                if config.LOCAL_WHISPER_LANGUAGE:
                    kwargs["language"] = config.LOCAL_WHISPER_LANGUAGE
                segments, info = model.transcribe(str(path), **kwargs)
                text = " ".join(segment.text.strip() for segment in segments if segment.text).strip()
            logger.info("Local Whisper transcript complete: chars=%s detected_language=%s language_probability=%s",
                        len(text), getattr(info, "language", ""), getattr(info, "language_probability", ""))
            return text, f"local-faster-whisper:{config.LOCAL_WHISPER_MODEL}"
        except Exception:
            logger.exception("Local Whisper transcription failed")
            return "", "failed:local-faster-whisper"
