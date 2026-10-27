import asyncio, logging, os, threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException, Request
from . import ai_log, social, transcription
from .assistant.graph import run_turn
from .models import SocialExtract, SocialStep, StartJob, StartTurn
from .workflow import run_job

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=transcription.warm_up, name="whisper-warm-up", daemon=True).start()
    yield


app = FastAPI(title="Pengoo AI service: trip worker, Pengoo assistant, social video AI", lifespan=lifespan)
TOKEN = os.getenv("AI_TRIP_INTERNAL_TOKEN", "")
tasks: set[asyncio.Task] = set()


def _authorize(token: str | None, what: str) -> None:
    if TOKEN and token != TOKEN:
        logger.warning("rejected %s: bad internal token", what)
        raise HTTPException(401, "Invalid internal token")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/ai-trip/jobs", status_code=202)
async def start(job: StartJob, x_internal_token: str | None = Header(default=None)):
    _authorize(x_internal_token, f"job {job.jobId}")
    logger.info("accepted job %s locale=%s destinations=%d", job.jobId, job.locale, len(job.request.get("destinations") or []))
    task = asyncio.create_task(run_job(job.model_dump(), TOKEN))
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return {"jobId": job.jobId, "accepted": True}


@app.post("/v1/assistant/turns", status_code=202)
async def start_turn(turn: StartTurn, x_internal_token: str | None = Header(default=None)):
    """One question to Pengoo. Accepted at once; the answer goes back through Java's callbacks."""
    _authorize(x_internal_token, f"assistant turn {turn.turnId}")
    logger.info("accepted assistant turn %s type=%s locale=%s", turn.turnId, turn.conversationType, turn.locale)
    task = asyncio.create_task(run_turn(turn.model_dump(), TOKEN))
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return {"turnId": turn.turnId, "accepted": True}


# Social video steps, called synchronously by google-maps-bot. X-Ai-Log-Url / X-Ai-Log-Token /
# X-Trace-Id say where the call log goes (Java's /v1/api/internal/ai-calls, the job's token and id).

@app.post("/v1/social/transcribe")
async def social_transcribe(request: Request, x_internal_token: str | None = Header(default=None),
                            x_ai_log_url: str | None = Header(default=None),
                            x_ai_log_token: str | None = Header(default=None),
                            x_trace_id: str | None = Header(default=None),
                            x_file_name: str | None = Header(default=None)):
    """Body: the audio file (MP3). Returns {transcript, provider}."""
    _authorize(x_internal_token, "social transcription")
    audio = await request.body()
    binding = social.bind_log(x_ai_log_url, x_ai_log_token, x_trace_id)
    try:
        text, provider = await transcription.transcribe(audio, x_file_name or "audio.mp3")
    except Exception as exc:
        logger.exception("social transcription failed")
        raise HTTPException(502, f"{type(exc).__name__}: {exc}")
    finally:
        ai_log.unbind(binding)
    return {"transcript": text, "provider": provider}


@app.post("/v1/social/prefilter")
async def social_prefilter(step: SocialStep, x_internal_token: str | None = Header(default=None),
                           x_ai_log_url: str | None = Header(default=None),
                           x_ai_log_token: str | None = Header(default=None),
                           x_trace_id: str | None = Header(default=None)):
    """{rejection: null | {reason, provider, model, usage}}; null also when the call failed."""
    _authorize(x_internal_token, "social prefilter")
    binding = social.bind_log(x_ai_log_url, x_ai_log_token, x_trace_id)
    try:
        rejection = await social.prefilter(metadata=step.metadata, transcript=step.transcript,
                                           ai_provider=step.aiProvider, ai_model=step.aiModel,
                                           ai_base_url=step.aiBaseUrl)
    finally:
        ai_log.unbind(binding)
    return {"rejection": rejection}


@app.post("/v1/social/extract")
async def social_extract(step: SocialExtract, x_internal_token: str | None = Header(default=None),
                         x_ai_log_url: str | None = Header(default=None),
                         x_ai_log_token: str | None = Header(default=None),
                         x_trace_id: str | None = Header(default=None)):
    """{extraction, provider, model, usage}. 422 for an unknown provider, 502 when the provider is not
    configured or its answer is unusable; `detail` says why."""
    _authorize(x_internal_token, "social extraction")
    binding = social.bind_log(x_ai_log_url, x_ai_log_token, x_trace_id)
    try:
        return await social.extract(metadata=step.metadata, transcript=step.transcript,
                                    transcript_provider=step.transcriptProvider, images=step.images,
                                    language=step.language, ai_provider=step.aiProvider, ai_model=step.aiModel,
                                    ai_base_url=step.aiBaseUrl)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        logger.exception("social extraction failed")
        raise HTTPException(502, f"{type(exc).__name__}: {exc}")
    finally:
        ai_log.unbind(binding)
