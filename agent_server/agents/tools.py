"""ARIA Agent Tools — extract_coordinates + execute_action."""

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from contracts.api_specs import SIM_BASE_URL, ENDPOINT_EXECUTE_ACTION


# --- extract_coordinates ---

class ExtractCoordinatesInput(BaseModel):
    image_base64: str = Field(description="Base64 encoded overhead camera image (PNG)")


@tool("extract_coordinates", args_schema=ExtractCoordinatesInput)
def extract_coordinates(image_base64: str) -> dict:
    """Extract precise (x, y, z) world coordinates of the red cube from the overhead camera image.

    Uses OpenCV to detect the red cube and convert pixel coordinates
    to world coordinates based on known camera parameters.
    Only works with the overhead camera image.
    """
    from agent_server.vision.object_detector import detect_cube_from_overhead

    try:
        coords = detect_cube_from_overhead(image_base64)
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
