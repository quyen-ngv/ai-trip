import hashlib, json, os, sqlite3, logging
from pathlib import Path
import httpx

logger = logging.getLogger(__name__)

class CachedModel:
    def __init__(self, prefix: str):
        self.url=os.getenv(f"{prefix}_BASE_URL","").rstrip("/")
        self.key=os.getenv(f"{prefix}_API_KEY","")
        self.model=os.getenv(f"{prefix}_MODEL","")
        logger.info(f"Initialized {prefix} LLM: url={self.url}, model={self.model}, key={'***' + self.key[-4:] if self.key else 'MISSING'}")
        path=Path(os.getenv("LLM_CACHE_PATH","/tmp/ai_trip_llm_cache.sqlite3"))
        path.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path,check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY,value TEXT NOT NULL)")

    async def json(self, system: str, payload: dict, max_tokens: int = 4096):
        """Call LLM with JSON output."""
        if not (self.url and self.key and self.model): 
            logger.error(f"LLM config incomplete")
            return None
        
        raw=json.dumps({"model":self.model,"system":system,"payload":payload,"max_tokens":max_tokens},ensure_ascii=False,sort_keys=True)
        key=hashlib.sha256(raw.encode()).hexdigest()
        row=self.db.execute("SELECT value FROM cache WHERE key=?",(key,)).fetchone()
        if row: 
            logger.info(f"Cache hit")
            return json.loads(row[0])
        
        body={
            "model": self.model,
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ]
        }
        
        logger.info(f"Calling {self.model}, payload: {len(json.dumps(payload))} bytes, max_tokens: {max_tokens}")
        
        async with httpx.AsyncClient(timeout=120) as client:
            try:
                response=await client.post(f"{self.url}/chat/completions",headers={"Authorization":f"Bearer {self.key}","Content-Type":"application/json"},json=body)
                
                # Fallback: if response_format rejected by API, retry without it
                if response.status_code == 400 and "response_format" in response.text.lower():
                    logger.warning("response_format rejected by API, retrying without it")
                    body.pop("response_format", None)
                    response = await client.post(f"{self.url}/chat/completions", headers={"Authorization":f"Bearer {self.key}","Content-Type":"application/json"}, json=body)
                
                if response.status_code != 200:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    response.raise_for_status()
                
                response_json = response.json()
                choices = response_json.get("choices") or []
                if not choices:
                    logger.warning("Empty choices from LLM, returning None")
                    return None
                content = choices[0].get("message",{}).get("content","")
                
                if not content or content.strip() == "":
                    logger.warning("Empty response from LLM, returning None")
                    return None
                
                logger.info(f"Got response: {len(content)} chars")
                value = json.loads(content)
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error: {e.response.status_code} - {e.response.text[:500]}")
                return None  # Fallback gracefully instead of crashing
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {str(e)}, content: {content[:500] if 'content' in locals() else 'N/A'}")
                return None
            except Exception as e:
                logger.error(f"Unexpected: {type(e).__name__}: {str(e)}")
                return None
        
        self.db.execute("INSERT OR REPLACE INTO cache(key,value) VALUES(?,?)",(key,json.dumps(value,ensure_ascii=False)))
        self.db.commit()
        return value
