"""Isaac Sim HTTP 서버 — Agent의 execute_action 요청을 수신하고 실행."""

import base64
import io
import threading
from queue import Queue, Empty

import numpy as np
import uvicorn
from fastapi import FastAPI
from PIL import Image

from contracts.schemas import (
    AnomalyReport,
    ExecuteActionRequest,
    ExecuteActionResponse,
    ExecutionLog,
    RobotState,
)
from contracts.api_specs import SIM_HOST, SIM_PORT


app = FastAPI(title="ARIA Isaac Sim API")

# 시뮬레이션 루프와의 통신 큐
_action_request_queue: Queue = Queue()
_action_response_queue: Queue = Queue()


@app.post("/execute_action", response_model=ExecuteActionResponse)
def execute_action(request: ExecuteActionRequest):
    """Agent로부터 action 요청을 받아 시뮬레이션 루프에 전달하고 결과를 반환."""
    _action_request_queue.put(request)
    # 시뮬레이션 루프가 처리할 때까지 대기
    response = _action_response_queue.get(timeout=120)
    return response


def start_server():
    """별도 스레드에서 HTTP 서버 시작."""
    thread = threading.Thread(
        target=uvicorn.run,
        args=(app,),
        kwargs={"host": SIM_HOST, "port": SIM_PORT, "log_level": "warning"},
        daemon=True,
    )
    thread.start()


def get_pending_request() -> ExecuteActionRequest | None:
    """시뮬레이션 루프에서 호출 — 대기 중인 action 요청을 가져옴."""
    try:
        return _action_request_queue.get_nowait()
    except Empty:
        return None


def send_response(response: ExecuteActionResponse) -> None:
    """시뮬레이션 루프에서 호출 — action 실행 결과를 HTTP 응답으로 반환."""
    _action_response_queue.put(response)


def rgba_to_base64_png(rgba: np.ndarray) -> str:
    """RGBA numpy array → base64 PNG 문자열."""
    img = Image.fromarray(rgba[:, :, :3])
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def send_anomaly_report(report: AnomalyReport, agent_url: str) -> None:
    """Agent Server에 anomaly report를 전송."""
    import httpx
    httpx.post(f"{agent_url}/anomaly", json=report.model_dump(), timeout=10)
