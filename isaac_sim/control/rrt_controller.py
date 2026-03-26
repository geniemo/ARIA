"""RRT 기반 trajectory planning + 실행 컨트롤러.

Isaac Sim 공식 PathPlannerController 패턴을 따름:
1. RRT로 경로 계획
2. PathPlannerVisualizer로 보간
3. LulaCSpaceTrajectoryGenerator로 trajectory 변환
4. ArticulationTrajectory로 action sequence 생성
"""

import numpy as np
from isaacsim.robot_motion.motion_generation import (
    ArticulationTrajectory,
    interface_config_loader,
)
from isaacsim.robot_motion.motion_generation.lula import RRT
from isaacsim.robot_motion.motion_generation.lula.trajectory_generator import (
    LulaCSpaceTrajectoryGenerator,
)
from isaacsim.robot_motion.motion_generation.path_planner_visualizer import (
    PathPlannerVisualizer,
)
from isaacsim.core.utils.types import ArticulationAction


class RRTController:
    """Franka용 RRT trajectory planner.

    compute_plan(): 현재 관절 → 목표 위치로 경로 계획 + action sequence 생성
    step(): 매 physics step마다 다음 action 반환
    is_done(): trajectory 실행 완료 여부
    """

    INTERPOLATION_MAX_DIST = 0.01  # 보간 시 waypoint 간 최대 거리 (rad)

    def __init__(self, robot_articulation, physics_dt: float = 1.0 / 60.0) -> None:
        self._robot = robot_articulation
        self._physics_dt = physics_dt

        # RRT planner 초기화
        rrt_config = interface_config_loader.load_supported_path_planner_config(
            "Franka", "RRT"
        )
        self._rrt = RRT(**rrt_config)
        self._rrt.set_max_iterations(10000)

        # Trajectory generator
        self._traj_gen = LulaCSpaceTrajectoryGenerator(
            rrt_config["robot_description_path"],
            rrt_config["urdf_path"],
        )

        # 로봇 base pose 설정 (시뮬레이션 상 위치와 일치시킴)
        robot_position, robot_orientation = robot_articulation.get_world_pose()
        self._rrt.set_robot_base_pose(
            robot_position=robot_position, robot_orientation=robot_orientation
        )

        # PathPlannerVisualizer (보간용)
        self._visualizer = PathPlannerVisualizer(robot_articulation, self._rrt)

        # 실행 상태
        self._action_sequence: list[ArticulationAction] = []
        self._done = True

    def compute_plan(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray | None = None,
    ) -> bool:
        """현재 관절에서 목표까지 action sequence를 생성한다.

        Returns:
            True: 성공, False: 경로 계획 실패
        """
        self._rrt.set_end_effector_target(target_position, target_orientation)
        self._rrt.update_world()

        active_joints = self._visualizer.get_active_joints_subset()
        start_pos = active_joints.get_joint_positions()

        path = self._rrt.compute_path(start_pos, np.array([]))

        if path is None or len(path) <= 1:
            # 재시도
            self._rrt.set_random_seed(np.random.randint(0, 100000))
            path = self._rrt.compute_path(start_pos, np.array([]))

        if path is None or len(path) <= 1:
            self._action_sequence = []
            self._done = True
            return False

        # 경로 보간 → trajectory → action sequence
        interpolated_path = self._visualizer.interpolate_path(
            path, self.INTERPOLATION_MAX_DIST
        )
        trajectory = self._traj_gen.compute_c_space_trajectory(interpolated_path)

        if trajectory is None:
            self._action_sequence = []
            self._done = True
            return False

        art_trajectory = ArticulationTrajectory(
            self._robot, trajectory, self._physics_dt
        )
        self._action_sequence = list(art_trajectory.get_action_sequence())
        self._done = False
        return True

    def step(self) -> ArticulationAction | None:
        """매 physics step마다 호출. 다음 action을 반환."""
        if self._done or len(self._action_sequence) == 0:
            return None

        if len(self._action_sequence) == 1:
            # 마지막 action — velocity를 0으로 설정하여 정지
            final = self._action_sequence.pop(0)
            self._done = True
            return ArticulationAction(
                final.joint_positions,
                np.zeros_like(final.joint_positions),
                joint_indices=final.joint_indices,
            )

        return self._action_sequence.pop(0)

    def is_done(self) -> bool:
        return self._done

    def reset(self) -> None:
        self._rrt.reset()
        self._action_sequence = []
        self._done = True
