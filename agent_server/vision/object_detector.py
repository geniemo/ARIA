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
TABLE_TOP_Z = 0.42
CUBE_Z = 0.44  # 큐브 중심 높이

# FOV 기반 픽셀 → 월드 변환 계수
# Isaac Sim Camera focal_length=18, horizontal_aperture=36 → FOV ≈ 90°
# 높이 2.5m에서 작업대(z=0.4)까지 거리 ≈ 2.1m
# 이 거리에서 화면이 커버하는 실제 영역을 기반으로 계수 계산
CAMERA_TO_TABLE_DIST = CAMERA_HEIGHT - TABLE_TOP_Z  # ~2.08m


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

    overhead 카메라가 [0.5, 0, 2.5]에서 아래를 보고 있으므로,
    픽셀 중심이 월드 [0.5, 0]에 대응한다.
    """
    # Isaac Sim Camera with euler [0, 90, 0]:
    # 이미지 X축 → 월드 -Y 방향
    # 이미지 Y축 → 월드 +X 방향
    # (카메라가 Y축 90도 회전되어 있으므로)

    # 픽셀 중심으로부터의 오프셋 (정규화: -0.5 ~ 0.5)
    px_norm = (cx / IMAGE_WIDTH) - 0.5
    py_norm = (cy / IMAGE_HEIGHT) - 0.5

    # FOV 기반 실제 거리 계산
    # focal_length=18, horizontal_aperture=36 → tan(hfov/2) = 36/(2*18) = 1.0
    half_width_world = CAMERA_TO_TABLE_DIST * 1.0  # tan(45°) = 1.0
    half_height_world = half_width_world * (IMAGE_HEIGHT / IMAGE_WIDTH)

    # 픽셀 → 월드 매핑
    world_x = CAMERA_POSITION[0] + py_norm * 2 * half_height_world
    world_y = CAMERA_POSITION[1] - px_norm * 2 * half_width_world

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
