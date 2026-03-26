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


def inject_absence(cube: DynamicCuboid) -> np.ndarray:
    """시나리오 B — 큐브를 작업 영역 밖으로 이동.

    Args:
        cube: 이동할 큐브 객체.

    Returns:
        큐브가 이동된 위치.
    """
    hidden_pos = np.array([2.0, 2.0, 0.0])
    cube.set_world_pose(position=hidden_pos)
    return hidden_pos
