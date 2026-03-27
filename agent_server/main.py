"""ARIA Agent Server 엔트리포인트 — uvicorn :8001."""

import uvicorn
from fastapi import FastAPI

from contracts.api_specs import AGENT_HOST, AGENT_PORT
from contracts.schemas import AnomalyReport
from agent_server.server.agent_api import run_recovery

app = FastAPI(title="ARIA Agent Server")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/anomaly")
def anomaly(report: AnomalyReport):
    """Isaac Sim에서 anomaly report를 수신하고 ReAct 복구 루프를 실행."""
    result = run_recovery(report)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host=AGENT_HOST, port=AGENT_PORT)
