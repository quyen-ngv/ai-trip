import asyncio, logging, os
from fastapi import FastAPI, Header, HTTPException
from .models import StartJob
from .workflow import run_job

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI(title="GoRoute AI Trip Worker")
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
