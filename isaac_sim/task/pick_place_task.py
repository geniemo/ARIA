"""Pick-and-Place phase 관리 + 실행 로그 기록."""

import numpy as np
from enum import auto, Enum

from contracts.schemas import ExecutionLog, RobotState
from contracts.skill_primitives import PhaseName, PhaseStatus
from control.rrt_controller import RRTController
from control.gripper_controller import GripperController
from isaacsim.core.utils.types import ArticulationAction


class TaskState(Enum):
    """P&P 태스크 상태."""
    IDLE = auto()
    RUNNING = auto()
    SUCCESS = auto()
    ANOMALY = auto()


GRIPPER_GRASP_THRESHOLD = 0.005  # m — 이 이하면 물체를 못 잡은 것
APPROACH_HEIGHT = 0.15  # m — 큐브 위 이 높이에서 먼저 정렬 후 내려감
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

        self._grasp_position = grasp_position.copy()
        self._approach_position = self._grasp_position.copy()
        self._approach_position[2] += APPROACH_HEIGHT
        self._lift_position = self._grasp_position.copy()
        self._lift_position[2] += LIFT_HEIGHT
        self._place_position = place_position.copy()
        self._place_above_position = self._place_position.copy()
        self._place_above_position[2] += APPROACH_HEIGHT
        self._orientation = grasp_orientation.copy() if grasp_orientation is not None else None

        self._state = TaskState.IDLE
        self._current_phase: PhaseName | None = None
        self._step_count = 0
        self._phase_start_step = 0
        self._execution_log: list[ExecutionLog] = []
        self._trajectory_planned = False
        self._sub_phase = 0  # approach/move의 2단계 (0: high, 1: low)

    @property
    def state(self) -> TaskState:
        return self._state

    @property
    def execution_log(self) -> list[ExecutionLog]:
        return self._execution_log

    def start(self) -> None:
        self._state = TaskState.RUNNING
        self._execution_log = []
        self._step_count = 0
        self._enter_phase(PhaseName.APPROACH)

    def step(self) -> TaskState:
        if self._state != TaskState.RUNNING:
            return self._state

        self._step_count += 1
        phase_handlers = {
            PhaseName.APPROACH: self._step_approach,
            PhaseName.CLOSE_GRIPPER: self._step_close_gripper,
            PhaseName.LIFT: self._step_lift,
            PhaseName.MOVE: self._step_move,
            PhaseName.OPEN_GRIPPER: self._step_open_gripper,
        }
        handler = phase_handlers.get(self._current_phase)
        if handler:
            handler()
        return self._state

    def get_robot_state(self) -> RobotState:
        ee_pos, _ = self._franka.end_effector.get_world_pose()
        joint_positions = self._franka.get_joint_positions().tolist()
        return RobotState(
            gripper_width=self._gripper_ctrl.get_width(),
            end_effector_position=ee_pos.tolist(),
            joint_positions=joint_positions[:7],
        )

    # --- 로봇 제어 ---

    def _stop_robot(self) -> None:
        joint_positions = self._franka.get_joint_positions()
        self._articulation_controller.apply_action(
            ArticulationAction(
                joint_positions=joint_positions,
                joint_velocities=np.zeros_like(joint_positions),
            )
        )

    def _move_to(self, target: np.ndarray) -> bool | None:
        """RRT로 이동. True=도달, False=계획실패, None=진행중."""
        if not self._trajectory_planned:
            if not self._rrt.compute_plan(target, self._orientation):
                return False
            self._trajectory_planned = True

        action = self._rrt.step()
        if action is not None:
            self._articulation_controller.apply_action(action)

        if self._rrt.is_done():
            self._trajectory_planned = False
            return True
        return None

    def _set_anomaly(self, reason: str) -> None:
        self._state = TaskState.ANOMALY
        self._stop_robot()
        self._complete_phase(reason=reason)

    # --- Phase 실행 ---

    def _step_two_stage_move(self, high_target: np.ndarray, low_target: np.ndarray,
                              on_complete, fail_reason_high: str, fail_reason_low: str) -> None:
        """2단계 이동 (high → low) 공통 로직."""
        target = high_target if self._sub_phase == 0 else low_target
        result = self._move_to(target)

        if result is True:
            if self._sub_phase == 0:
                self._sub_phase = 1
            else:
                on_complete()
        elif result is False:
            reason = fail_reason_high if self._sub_phase == 0 else fail_reason_low
            self._set_anomaly(reason)

    def _step_approach(self) -> None:
        def on_complete():
            self._complete_phase()
            self._enter_phase(PhaseName.CLOSE_GRIPPER)
            self._gripper_ctrl.close()

        self._step_two_stage_move(
            self._approach_position, self._grasp_position,
            on_complete, "approach_plan_failed", "descend_plan_failed",
        )

    def _step_close_gripper(self) -> None:
        self._franka.gripper.close()
        if self._gripper_ctrl.is_done():
            width = self._gripper_ctrl.get_width()
            self._complete_phase(gripper_width_final=width)
            if width < GRIPPER_GRASP_THRESHOLD:
                self._state = TaskState.ANOMALY
                self._stop_robot()
            else:
                self._enter_phase(PhaseName.LIFT)

    def _step_lift(self) -> None:
        result = self._move_to(self._lift_position)
        if result is True:
            self._complete_phase()
            self._enter_phase(PhaseName.MOVE)
        elif result is False:
            self._set_anomaly("lift_plan_failed")

    def _step_move(self) -> None:
        def on_complete():
            self._complete_phase()
            self._enter_phase(PhaseName.OPEN_GRIPPER)
            self._gripper_ctrl.open()

        self._step_two_stage_move(
            self._place_above_position, self._place_position,
            on_complete, "move_plan_failed", "place_descend_plan_failed",
        )

    def _step_open_gripper(self) -> None:
        self._franka.gripper.open()
        if self._gripper_ctrl.is_done():
            width = self._gripper_ctrl.get_width()
            self._complete_phase(gripper_width_final=width)
            self._state = TaskState.SUCCESS

    # --- Phase 관리 ---

    def _enter_phase(self, phase: PhaseName) -> None:
        self._current_phase = phase
        self._phase_start_step = self._step_count
        self._trajectory_planned = False
        self._sub_phase = 0
        self._gripper_ctrl.reset_settle()

    def _complete_phase(self, gripper_width_final: float | None = None, reason: str | None = None) -> None:
        self._execution_log.append(
            ExecutionLog(
                phase=self._current_phase,
                status=PhaseStatus.COMPLETED if reason is None else PhaseStatus.ABORTED,
                duration_steps=self._step_count - self._phase_start_step,
                gripper_width_final=gripper_width_final,
                reason=reason,
            )
        )
