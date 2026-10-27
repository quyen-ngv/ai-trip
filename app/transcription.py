"""Speech-to-text for social videos: always faster-whisper on this server, never a paid API.

The model is baked into the image; LOCAL_WHISPER_ENABLED=false turns transcription off and the
social pipeline then works from the caption and on-screen text alone.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger(__name__)
_LOCAL_WHISPER = None
_LOCAL_LOCK = threading.Lock()


async def transcribe(audio: bytes, filename: str = "audio.mp3") -> tuple[str, str]:
    """(transcript, "provider:model"), transcribed by local Whisper whatever API keys are set."""
    if config.LOCAL_WHISPER_ENABLED:
        return await asyncio.to_thread(_local, audio, filename)
    logger.warning("Audio transcription skipped: LOCAL_WHISPER_ENABLED=false")
    return "", "skipped:no_transcriber"


def warm_up() -> None:
    """Load local Whisper at startup when it is the transcriber jobs will use, so the first social
    video job does not wait for the model to load."""
    if not config.LOCAL_WHISPER_ENABLED:
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
