import asyncio, logging, os
from fastapi import FastAPI, Header, HTTPException
from .assistant.graph import run_turn
from .models import StartJob, StartTurn
from .workflow import run_job

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI(title="GoRoute AI Trip Worker and Pengoo assistant")
TOKEN = os.getenv("AI_TRIP_INTERNAL_TOKEN", "")
tasks: set[asyncio.Task] = set()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/ai-trip/jobs", status_code=202)
async def start(job: StartJob, x_internal_token: str | None = Header(default=None)):
    if TOKEN and x_internal_token != TOKEN:
        logger.warning("rejected job %s: bad internal token", job.jobId)
        raise HTTPException(401, "Invalid internal token")
    logger.info("accepted job %s locale=%s destinations=%d", job.jobId, job.locale, len(job.request.get("destinations") or []))
    task = asyncio.create_task(run_job(job.model_dump(), TOKEN))
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return {"jobId": job.jobId, "accepted": True}


@app.post("/v1/assistant/turns", status_code=202)
async def start_turn(turn: StartTurn, x_internal_token: str | None = Header(default=None)):
    """One question to Pengoo. Accepted at once; the answer goes back through Java's callbacks."""
    if TOKEN and x_internal_token != TOKEN:
        logger.warning("rejected assistant turn %s: bad internal token", turn.turnId)
        raise HTTPException(401, "Invalid internal token")
    logger.info("accepted assistant turn %s type=%s locale=%s", turn.turnId, turn.conversationType, turn.locale)
    task = asyncio.create_task(run_turn(turn.model_dump(), TOKEN))
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return {"turnId": turn.turnId, "accepted": True}
