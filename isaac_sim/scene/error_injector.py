"""시나리오별 에러 주입."""

import numpy as np
from isaacsim.core.api.objects import DynamicCuboid


def inject_offset(cube: DynamicCuboid, offset_xy: np.ndarray | None = None) -> np.ndarray:
    """시나리오 A — 큐브를 원래 위치에서 XY 방향으로 오프셋.

    Args:
        cube: 오프셋할 큐브 객체.
        offset_xy: [dx, dy] 오프셋 (m). None이면 랜덤 생성 (2~5cm).

    Returns:
        적용된 오프셋 [dx, dy].
    """
    if offset_xy is None:
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0.02, 0.05)
        offset_xy = np.array([np.cos(angle) * distance, np.sin(angle) * distance])

    current_pos, current_ori = cube.get_world_pose()
    new_pos = current_pos.copy()
    new_pos[0] += offset_xy[0]
    new_pos[1] += offset_xy[1]
    cube.set_world_pose(position=new_pos, orientation=current_ori)

    return offset_xy


def inject_absence(cube: DynamicCuboid, displaced_pos: np.ndarray | None = None) -> np.ndarray:
    """시나리오 B — 큐브를 작업대 위이지만 카메라에서 보이지 않는 위치로 이동.

    로봇 팔 뒤쪽(반대편)이나 작업대 가장자리로 이동시켜,
    현재 카메라 시야에서는 안 보이지만 explore로 찾을 수 있는 위치.

    Args:
        cube: 이동할 큐브 객체.
        displaced_pos: 이동 위치. None이면 작업대 반대편으로 이동.

    Returns:
        큐브가 이동된 위치.
    """
    if displaced_pos is None:
        # 작업대 위, 원래 위치 반대편 (로봇 팔 뒤쪽)
        # 작업대: x=[0.2, 0.8], y=[-0.4, 0.4], 상단 z=0.42
        displaced_pos = np.array([0.3, -0.15, 0.44])

    cube.set_world_pose(position=displaced_pos)
    return displaced_pos
