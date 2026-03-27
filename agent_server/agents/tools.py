"""ARIA Agent Tools — extract_coordinates + execute_action."""

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from contracts.api_specs import SIM_BASE_URL, ENDPOINT_EXECUTE_ACTION

# --- 현재 anomaly report 저장 (extract_coordinates가 접근) ---

_current_report = None


def set_current_report(report) -> None:
    global _current_report
    _current_report = report


# --- extract_coordinates ---

@tool("extract_coordinates")
def extract_coordinates() -> dict:
    """Extract precise (x, y, z) world coordinates of the red cube from the overhead camera image.

    Automatically uses the overhead image from the current anomaly report.
    No input parameters needed.
    """
    from agent_server.vision.object_detector import detect_cube_from_overhead

    if _current_report is None:
        return {"error": "No anomaly report available"}

    try:
        coords = detect_cube_from_overhead(_current_report.overhead_image)
        return coords
    except Exception as e:
        return {"error": str(e)}


# --- execute_action ---

class ExecuteActionInput(BaseModel):
    action: str = Field(description="Action type: 'grasp' or 'move'")
    coords: dict = Field(description="Target coordinates {'x': float, 'y': float, 'z': float}")
    intent: str = Field(description="'explore' (move + observe) or 'recover' (full pick-and-place)")


@tool("execute_action", args_schema=ExecuteActionInput)
def execute_action(action: str, coords: dict, intent: str) -> dict:
    """Command the robot to perform an action in Isaac Sim.

    Sends an HTTP request to the Isaac Sim server and returns the result
    including success/failure, camera images, robot state, and execution log.
    """
    url = f"{SIM_BASE_URL}{ENDPOINT_EXECUTE_ACTION}"

    payload = {
        "action": action,
        "coords": coords,
        "intent": intent,
    }

    try:
        response = httpx.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException:
        return {"error": "Isaac Sim did not respond within 120 seconds"}
    except httpx.ConnectError:
        return {"error": "Cannot connect to Isaac Sim server. Is it running on :8000?"}
    except Exception as e:
        return {"error": str(e)}
