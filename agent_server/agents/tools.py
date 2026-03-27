"""ARIA Agent Tools — extract_coordinates + execute_action."""

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from contracts.api_specs import SIM_BASE_URL, ENDPOINT_EXECUTE_ACTION

# 최신 overhead 이미지 (extract_coordinates가 사용)
_latest_overhead_image: str | None = None


def set_current_report(report) -> None:
    """anomaly report에서 overhead 이미지를 설정."""
    global _latest_overhead_image
    _latest_overhead_image = report.overhead_image


def _update_overhead_image(image_b64: str) -> None:
    """execute_action 응답에서 최신 overhead 이미지로 업데이트."""
    global _latest_overhead_image
    _latest_overhead_image = image_b64


@tool("extract_coordinates")
def extract_coordinates() -> dict:
    """Extract precise (x, y, z) world coordinates of the red cube
    from the latest overhead camera image. No input parameters needed."""
    from agent_server.vision.object_detector import detect_cube_from_overhead

    if _latest_overhead_image is None:
        return {"error": "No overhead image available"}

    try:
        return detect_cube_from_overhead(_latest_overhead_image)
    except Exception as e:
        return {"error": str(e)}


class ExecuteActionInput(BaseModel):
    action: str = Field(description="Action type: 'grasp' or 'move'")
    coords: dict = Field(description="Target coordinates {'x': float, 'y': float, 'z': float}")
    intent: str = Field(description="'explore' (move + observe) or 'recover' (full pick-and-place)")


@tool("execute_action", args_schema=ExecuteActionInput)
def execute_action(action: str, coords: dict, intent: str) -> dict:
    """Command the robot to perform an action in Isaac Sim.
    After execution, the overhead image is updated for extract_coordinates."""
    url = f"{SIM_BASE_URL}{ENDPOINT_EXECUTE_ACTION}"
    payload = {"action": action, "coords": coords, "intent": intent}

    try:
        response = httpx.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        if data.get("overhead_image"):
            _update_overhead_image(data["overhead_image"])

        return {
            "success": data.get("success"),
            "gripper_width": data.get("gripper_width"),
            "robot_state": data.get("robot_state"),
            "execution_log": data.get("execution_log"),
            "note": "New overhead image captured. Call extract_coordinates to detect the cube.",
        }
    except httpx.TimeoutException:
        return {"error": "Isaac Sim did not respond within 120 seconds"}
    except httpx.ConnectError:
        return {"error": "Cannot connect to Isaac Sim server. Is it running on :8000?"}
    except Exception as e:
        return {"error": str(e)}
