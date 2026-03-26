"""로봇 스킬 및 상태 열거형 정의."""

from enum import Enum


class SkillName(str, Enum):
    """로봇이 수행할 수 있는 동작."""
    GRASP = "grasp"
    MOVE = "move"


class ActionIntent(str, Enum):
    """execute_action의 intent 구분."""
    EXPLORE = "explore"   # 이동 후 상태만 반환, P&P 미수행
    RECOVER = "recover"   # grasp 성공 시 full P&P 수행


class PhaseStatus(str, Enum):
    """phase 실행 상태."""
    COMPLETED = "completed"
    ABORTED = "aborted"


class PhaseName(str, Enum):
    """Pick-and-Place phase."""
    APPROACH = "approach"
    CLOSE_GRIPPER = "close_gripper"
    LIFT = "lift"
    MOVE = "move"
    OPEN_GRIPPER = "open_gripper"
