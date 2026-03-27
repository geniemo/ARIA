"""ARIA Agent Server 엔트리포인트 — uvicorn :8001."""

import uvicorn
from fastapi import FastAPI

from contracts.api_specs import AGENT_HOST, AGENT_PORT

app = FastAPI(title="ARIA Agent Server")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host=AGENT_HOST, port=AGENT_PORT)
