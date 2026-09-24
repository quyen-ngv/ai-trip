"""Small, persistent TTL cache for the two cheap-model Web-research nodes."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class ResearchCache:
    def __init__(self, path: str):
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(file, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS research_cache (key TEXT PRIMARY KEY, expires_at REAL NOT NULL, value TEXT NOT NULL)")

    @staticmethod
    def key(namespace: str, value: Any) -> str:
        payload = json.dumps({"namespace": namespace, "value": value}, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        row = self.db.execute("SELECT expires_at, value FROM research_cache WHERE key=?", (key,)).fetchone()
        if not row or row[0] < time.time():
            if row:
                self.db.execute("DELETE FROM research_cache WHERE key=?", (key,))
                self.db.commit()
            return None
        return json.loads(row[1])

    def put(self, key: str, value: Any, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        self.db.execute(
            "INSERT OR REPLACE INTO research_cache(key, expires_at, value) VALUES(?,?,?)",
            (key, time.time() + ttl_seconds, json.dumps(value, ensure_ascii=False)),
        )
        self.db.commit()
