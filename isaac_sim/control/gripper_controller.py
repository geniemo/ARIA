"""Gripper 제어 — ArticulationAction 기반 open/close."""

from isaacsim.robot.manipulators.grippers import ParallelGripper


class GripperController:
    """Franka ParallelGripper의 open/close 제어.

    gripper width를 읽어 동작 완료 여부를 판단한다.
    """

    # gripper가 더 이상 움직이지 않음을 판단하는 임계값
    SETTLE_THRESHOLD = 0.0005  # m
    SETTLE_STEPS = 5  # 연속 N스텝 동안 변화 없으면 완료

    def __init__(self, gripper: ParallelGripper) -> None:
        self._gripper = gripper
        self._prev_width = None
        self._settle_count = 0

    def open(self) -> None:
        """gripper 열기 명령."""
        self._gripper.open()
        self.reset_settle()

    def close(self) -> None:
        """gripper 닫기 명령."""
        self._gripper.close()
        self.reset_settle()

    def get_width(self) -> float:
        """현재 gripper width (두 finger 간격 합)."""
        positions = self._gripper.get_joint_positions()
        return float(positions[0] + positions[1])

    def is_done(self) -> bool:
        """gripper 동작이 완료되었는지 (더 이상 움직이지 않음)."""
        current_width = self.get_width()
        if self._prev_width is not None:
            if abs(current_width - self._prev_width) < self.SETTLE_THRESHOLD:
                self._settle_count += 1
            else:
                self._settle_count = 0
        self._prev_width = current_width
        return self._settle_count >= self.SETTLE_STEPS

    def reset_settle(self) -> None:
        """settle 판정 상태 초기화."""
        self._prev_width = None
        self._settle_count = 0
