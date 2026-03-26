"""RMPFlow 기반 모션 제어 — Franka 내장 config 사용."""

import numpy as np
import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import SingleArticulation


class RMPFlowController(mg.MotionPolicyController):
    """Franka용 RMPFlow 컨트롤러.

    매 스텝 목표 좌표를 받아 joint action을 계산한다.
    Isaac Sim 내장 Franka RMPFlow config를 사용하므로 별도 config 불필요.
    """

    def __init__(
        self,
        name: str,
        robot_articulation: SingleArticulation,
        physics_dt: float = 1.0 / 60.0,
    ) -> None:
        self.rmp_flow_config = mg.interface_config_loader.load_supported_motion_policy_config(
            "Franka", "RMPflow"
        )
        self.rmp_flow = mg.lula.motion_policies.RmpFlow(**self.rmp_flow_config)
        self.articulation_rmp = mg.ArticulationMotionPolicy(
            robot_articulation, self.rmp_flow, physics_dt
        )

        mg.MotionPolicyController.__init__(
            self, name=name, articulation_motion_policy=self.articulation_rmp
        )

        self._default_position, self._default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position,
            robot_orientation=self._default_orientation,
        )

    # 수렴 판정용 상태
    _prev_ee_pos = None
    _converge_count = 0
    CONVERGE_THRESHOLD = 0.001  # m — 이 이하로 움직이면 정지로 판정
    CONVERGE_STEPS = 10  # 연속 N스텝

    def has_converged(self, current_position: np.ndarray) -> bool:
        """EE가 더 이상 움직이지 않으면 수렴으로 판정."""
        if self._prev_ee_pos is not None:
            delta = np.linalg.norm(current_position - self._prev_ee_pos)
            if delta < self.CONVERGE_THRESHOLD:
                self._converge_count += 1
            else:
                self._converge_count = 0
        self._prev_ee_pos = current_position.copy()
        return self._converge_count >= self.CONVERGE_STEPS

    def reset_convergence(self):
        """수렴 판정 상태 초기화."""
        self._prev_ee_pos = None
        self._converge_count = 0
