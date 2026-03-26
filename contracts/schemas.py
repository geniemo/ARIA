"""Isaac Sim <-> Agent Server 간 공유 Pydantic 모델."""

from pydantic import BaseModel, Field

from contracts.skill_primitives import ActionIntent, PhaseName, PhaseStatus


# --- 센서 / 상태 ---

class RobotState(BaseModel):
    """로봇의 현재 상태."""
    gripper_width: float = Field(description="현재 gripper 간격 (m)")
    end_effector_position: list[float] = Field(description="end-effector [x, y, z] (m)")
    joint_positions: list[float] = Field(description="7-DOF 관절 각도 (rad)")


# --- 실행 로그 ---

class ExecutionLog(BaseModel):
    """단일 phase의 실행 기록."""
    phase: PhaseName
    status: PhaseStatus
    duration_steps: int = Field(default=0, description="phase 소요 스텝 수")
    gripper_width_final: float | None = Field(default=None, description="phase 종료 시 gripper width")
    reason: str | None = Field(default=None, description="aborted 시 사유")


# --- Anomaly Report (Isaac Sim -> Agent) ---

class AnomalyReport(BaseModel):
    """Isaac Sim이 이상 감지 시 Agent에 전송하는 보고."""
    overhead_image: str = Field(description="overhead 카메라 이미지 (base64 PNG)")
    wrist_image: str = Field(description="wrist 카메라 이미지 (base64 PNG)")
    robot_state: RobotState
    execution_log: list[ExecutionLog] = Field(description="이상 발생까지의 phase별 실행 로그")


# --- Execute Action (Agent -> Isaac Sim -> Agent) ---

class ExecuteActionRequest(BaseModel):
    """Agent가 Isaac Sim에 보내는 동작 요청."""
    action: str = Field(description="동작 유형 (grasp, move 등)")
    coords: dict[str, float] = Field(description="목표 좌표 {x, y, z}")
    intent: ActionIntent = Field(description="explore (탐색) 또는 recover (복구)")


class ExecuteActionResponse(BaseModel):
    """Isaac Sim이 동작 실행 후 반환하는 결과."""
    success: bool
    gripper_width: float
    robot_state: RobotState
    overhead_image: str = Field(description="실행 후 overhead 카메라 이미지 (base64 PNG)")
    wrist_image: str = Field(description="실행 후 wrist 카메라 이미지 (base64 PNG)")
    execution_log: list[ExecutionLog] = Field(description="실행 과정의 phase별 로그")
