"""Overhead 카메라 이미지에서 빨간 큐브의 월드 좌표를 추출."""

import base64
import io

import cv2
import numpy as np
from PIL import Image

# --- 카메라 파라미터 (overhead 카메라 기준) ---
# 카메라 위치: [0.5, 0.0, 2.5], 아래를 봄
# 해상도: 640x480
# Isaac Sim Camera: euler [0, 90, 0] → Y축 90도 회전으로 아래를 봄

CAMERA_POSITION = np.array([0.5, 0.0, 2.5])
CAMERA_HEIGHT = CAMERA_POSITION[2]  # 2.5m
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# 작업대 상단 Z (큐브가 놓이는 높이)
TABLE_TOP_Z = 0.40  # table position[2] + scale[2]/2 = 0.2 + 0.2
CUBE_Z = 0.44  # 큐브 중심 높이 (0.42 + 0.04/2), scene_builder 기준

# 실측 기반 픽셀 → 월드 변환 스케일 계수
# 여러 큐브 위치에서 실제 좌표와 픽셀 좌표를 비교하여 도출
# px_norm (이미지 가로) → world_y, py_norm (이미지 세로) → world_x
SCALE_PX_TO_WORLD_Y = 0.865  # px_norm 1.0 → 0.865m
SCALE_PY_TO_WORLD_X = 0.644  # py_norm 1.0 → 0.644m


def _base64_to_cv2(image_base64: str) -> np.ndarray:
    """base64 PNG → OpenCV BGR 이미지."""
    image_bytes = base64.b64decode(image_base64)
    pil_image = Image.open(io.BytesIO(image_bytes))
    rgb = np.array(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _detect_red_cube(bgr_image: np.ndarray) -> tuple[float, float] | None:
    """BGR 이미지에서 빨간 큐브의 중심 픽셀 좌표를 반환.

    Returns:
        (cx, cy) 픽셀 좌표 또는 None (감지 실패)
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # 빨간색 마스크 (HSV에서 빨간색은 0 근처와 180 근처)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    # 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 컨투어 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 가장 큰 컨투어의 중심
    largest = max(contours, key=cv2.contourArea)

    # 너무 작은 컨투어 무시 (노이즈)
    if cv2.contourArea(largest) < 50:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return cx, cy


def _pixel_to_world(cx: float, cy: float) -> dict:
    """overhead 카메라의 픽셀 좌표 → 월드 좌표 변환.

    실측 데이터 기반 선형 매핑:
    - 이미지 가로(px) → 월드 Y (반비례)
    - 이미지 세로(py) → 월드 X (반비례)
    - 이미지 중심 → 카메라 위치 [0.5, 0]
    """
    px_norm = (cx / IMAGE_WIDTH) - 0.5
    py_norm = (cy / IMAGE_HEIGHT) - 0.5

    world_x = CAMERA_POSITION[0] - py_norm * SCALE_PY_TO_WORLD_X
    world_y = CAMERA_POSITION[1] - px_norm * SCALE_PX_TO_WORLD_Y

    return {
        "x": round(float(world_x), 4),
        "y": round(float(world_y), 4),
        "z": round(float(CUBE_Z), 4),
    }


def detect_cube_from_overhead(image_base64: str) -> dict:
    """overhead 카메라 이미지에서 빨간 큐브의 월드 좌표를 추출.

    Args:
        image_base64: overhead 카메라 이미지 (base64 PNG)

    Returns:
        {"x": float, "y": float, "z": float} 또는 {"error": str}
    """
    bgr = _base64_to_cv2(image_base64)
    pixel = _detect_red_cube(bgr)

    if pixel is None:
        return {"error": "object not found in overhead image"}

    cx, cy = pixel
    return _pixel_to_world(cx, cy)
