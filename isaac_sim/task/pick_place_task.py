"""Pick-and-Place phase 관리 + 실행 로그 기록."""

import numpy as np
from enum import Enum, auto

from contracts.schemas import ExecutionLog, RobotState
from contracts.skill_primitives import PhaseName, PhaseStatus
from control.rrt_controller import RRTController
from control.gripper_controller import GripperController


class TaskState(Enum):
    """P&P 태스크 상태."""
    IDLE = auto()
    RUNNING = auto()
    SUCCESS = auto()
    ANOMALY = auto()


# grasp 실패 판정 임계값 — 이 이하면 물체를 못 잡은 것
GRIPPER_GRASP_THRESHOLD = 0.005  # m

# approach 높이 — 큐브 위 이 높이에서 먼저 정렬 후 내려감
APPROACH_HEIGHT = 0.15  # m

# lift 높이
LIFT_HEIGHT = 0.15  # m


class PickPlaceTask:
    """RRT trajectory planner 기반 Pick-and-Place 태스크.

    phase별로 실행하고, 완료 조건을 판정하고, 실행 로그를 기록한다.
    close_gripper 완료 후 gripper width를 체크하여 anomaly를 판정한다.
    """

    def __init__(
        self,
        franka,
        rrt: RRTController,
        gripper_ctrl: GripperController,
        grasp_position: np.ndarray,
        place_position: np.ndarray,
        grasp_orientation: np.ndarray | None = None,
    ) -> None:
        self._franka = franka
        self._rrt = rrt
        self._gripper_ctrl = gripper_ctrl
        self._articulation_controller = franka.get_articulation_controller()

        # 좌표
        self._grasp_position = grasp_position.copy()

        self._approach_position = self._grasp_position.copy()
        self._approach_position[2] += APPROACH_HEIGHT

        self._lift_position = self._grasp_position.copy()
        self._lift_position[2] += LIFT_HEIGHT

        self._place_position = place_position.copy()

        self._place_above_position = self._place_position.copy()
        self._place_above_position[2] += APPROACH_HEIGHT

        self._orientation = grasp_orientation.copy() if grasp_orientation is not None else None

        # 상태
        self._state = TaskState.IDLE
        self._current_phase: PhaseName | None = None
        self._step_count = 0
        self._phase_start_step = 0
        self._execution_log: list[ExecutionLog] = []
        self._trajectory_planned = False

    @property
    def state(self) -> TaskState:
        return self._state

    @property
    def execution_log(self) -> list[ExecutionLog]:
        return self._execution_log

    def start(self) -> None:
        """태스크 시작."""
        self._state = TaskState.RUNNING
        self._execution_log = []
        self._step_count = 0
        self._enter_phase(PhaseName.APPROACH)

    def step(self) -> TaskState:
        """매 시뮬레이션 스텝마다 호출."""
        if self._state != TaskState.RUNNING:
            return self._state

        self._step_count += 1

        if self._current_phase == PhaseName.APPROACH:
            self._step_approach()
        elif self._current_phase == PhaseName.CLOSE_GRIPPER:
            self._step_close_gripper()
        elif self._current_phase == PhaseName.LIFT:
            self._step_lift()
        elif self._current_phase == PhaseName.MOVE:
            self._step_move()
        elif self._current_phase == PhaseName.OPEN_GRIPPER:
            self._step_open_gripper()

        return self._state

    def get_robot_state(self) -> RobotState:
        """현재 로봇 상태."""
        ee_pos, _ = self._franka.end_effector.get_world_pose()
        joint_positions = self._franka.get_joint_positions().tolist()
        return RobotState(
            gripper_width=self._gripper_ctrl.get_width(),
            end_effector_position=ee_pos.tolist(),
            joint_positions=joint_positions[:7],
        )

    # --- Phase 실행 (RRT trajectory 기반) ---

    def _move_to(self, target_position: np.ndarray) -> bool | None:
        """RRT로 목표까지 이동. Returns: True=도달, False=계획실패, None=진행중."""
        if not self._trajectory_planned:
            success = self._rrt.compute_plan(target_position, self._orientation)
            if not success:
                return False
            self._trajectory_planned = True

        action = self._rrt.step()
        if action is not None:
            self._articulation_controller.apply_action(action)

        if self._rrt.is_done():
            self._trajectory_planned = False
            return True

        return None  # 진행 중

    def _step_approach(self) -> None:
        """approach 위치 → grasp 위치로 2단계 이동."""
        if not hasattr(self, '_approach_phase'):
            self._approach_phase = 0  # 0: 높은 위치, 1: grasp 위치

        if self._approach_phase == 0:
            result = self._move_to(self._approach_position)
            if result is True:
                self._approach_phase = 1
            elif result is False:
                self._state = TaskState.ANOMALY
                self._complete_phase(reason="approach_plan_failed")
        elif self._approach_phase == 1:
            result = self._move_to(self._grasp_position)
            if result is True:
                self._complete_phase()
                self._enter_phase(PhaseName.CLOSE_GRIPPER)
                self._gripper_ctrl.close()
            elif result is False:
                self._state = TaskState.ANOMALY
                self._complete_phase(reason="descend_plan_failed")

    def _step_close_gripper(self) -> None:
        """gripper 닫기."""
        self._franka.gripper.close()
        if self._gripper_ctrl.is_done():
            width = self._gripper_ctrl.get_width()
            self._complete_phase(gripper_width_final=width)

            if width < GRIPPER_GRASP_THRESHOLD:
                self._state = TaskState.ANOMALY
                return

            self._enter_phase(PhaseName.LIFT)

    def _step_lift(self) -> None:
        """물체를 들어올림."""
        result = self._move_to(self._lift_position)
        if result is True:
            self._complete_phase()
            self._enter_phase(PhaseName.MOVE)
        elif result is False:
            self._state = TaskState.ANOMALY
            self._complete_phase(reason="lift_plan_failed")

    def _step_move(self) -> None:
        """place 위 높은 위치 → place 위치로 2단계 이동."""
        if not hasattr(self, '_move_phase'):
            self._move_phase = 0

        if self._move_phase == 0:
            result = self._move_to(self._place_above_position)
            if result is True:
                self._move_phase = 1
            elif result is False:
                self._state = TaskState.ANOMALY
                self._complete_phase(reason="move_plan_failed")
        elif self._move_phase == 1:
            result = self._move_to(self._place_position)
            if result is True:
                self._complete_phase()
                self._enter_phase(PhaseName.OPEN_GRIPPER)
                self._gripper_ctrl.open()
            elif result is False:
                self._state = TaskState.ANOMALY
                self._complete_phase(reason="place_descend_plan_failed")

    def _step_open_gripper(self) -> None:
        """gripper 열기."""
        self._franka.gripper.open()
        if self._gripper_ctrl.is_done():
            width = self._gripper_ctrl.get_width()
            self._complete_phase(gripper_width_final=width)
            self._state = TaskState.SUCCESS

    # --- Phase 관리 ---

    def _enter_phase(self, phase: PhaseName) -> None:
        """새 phase 진입."""
        self._current_phase = phase
        self._phase_start_step = self._step_count
        self._trajectory_planned = False
        self._gripper_ctrl._reset_settle()
        # sub-phase 상태 초기화
        if hasattr(self, '_approach_phase'):
            del self._approach_phase
        if hasattr(self, '_move_phase'):
            del self._move_phase

    def _complete_phase(self, gripper_width_final: float | None = None, reason: str | None = None) -> None:
        """현재 phase 완료 기록."""
        duration = self._step_count - self._phase_start_step
        self._execution_log.append(
            ExecutionLog(
                phase=self._current_phase,
                status=PhaseStatus.COMPLETED if reason is None else PhaseStatus.ABORTED,
                duration_steps=duration,
                gripper_width_final=gripper_width_final,
                reason=reason,
            )
        )
