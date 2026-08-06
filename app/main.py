import asyncio, os, logging
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from .models import StartJob
from .workflow import run_job

# Set log level to DEBUG to see detailed request/response
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app=FastAPI(title="GoRoute AI Trip Worker")
TOKEN=os.getenv("AI_TRIP_INTERNAL_TOKEN","")
tasks:set[asyncio.Task]=set()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()
    logger.info(f"Incoming {request.method} {request.url.path} | Headers: {dict(request.headers)} | Body: {body.decode('utf-8') if body else 'EMPTY'}")
    
    # Recreate request with body for downstream
    async def receive():
        return {"type": "http.request", "body": body}
    
    request._receive = receive
    response = await call_next(request)
    return response

@app.get("/health")
async def health():return {"status":"ok"}

@app.post("/v1/ai-trip/jobs",status_code=202)
async def start(job:StartJob,x_internal_token:str|None=Header(default=None)):
    logger.info(f"Received AI trip job: jobId={job.jobId}, locale={job.locale}")
    if TOKEN and x_internal_token!=TOKEN:
        logger.error(f"Invalid token: expected={TOKEN[:10]}..., got={x_internal_token[:10] if x_internal_token else 'None'}...")
        raise HTTPException(401,"Invalid internal token")
    task=asyncio.create_task(run_job(job.model_dump(),TOKEN));tasks.add(task);task.add_done_callback(tasks.discard)
    return {"jobId":job.jobId,"accepted":True}
