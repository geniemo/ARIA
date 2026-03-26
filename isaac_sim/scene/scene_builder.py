"""ARIA Scene Builder — Franka + 작업대 + 큐브 + 카메라 배치."""

import numpy as np
from pxr import UsdGeom, Gf

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.stage import get_current_stage
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path


# --- 상수 ---

# 작업대
TABLE_POSITION = np.array([0.5, 0.0, 0.2])
TABLE_SCALE = np.array([0.6, 0.8, 0.4])
TABLE_COLOR = np.array([139, 69, 19])

# 큐브 (pick 대상) — 작업대 왼쪽
CUBE_SIZE = 0.04
CUBE_DEFAULT_POSITION = np.array([0.5, -0.2, 0.42 + CUBE_SIZE / 2])
CUBE_COLOR = np.array([255, 0, 0])

# Franka
FRANKA_POSITION = np.array([0.0, 0.0, 0.4])

# 로봇 받침대
PEDESTAL_POSITION = np.array([0.0, 0.0, 0.2])
PEDESTAL_SCALE = np.array([0.3, 0.3, 0.4])
PEDESTAL_COLOR = np.array([80, 80, 80])

# Gripper
GRIPPER_OPEN = np.array([0.05, 0.05])
GRIPPER_CLOSED = np.array([0.0, 0.0])
GRIPPER_DELTAS = np.array([0.01, 0.01])

# Place 위치 — 작업대 오른쪽
PLACE_POSITION = np.array([0.5, 0.2, 0.42 + CUBE_SIZE / 2])

# 카메라
# USD 카메라 기본 시선: -Z축 방향
OVERHEAD_CAM_POSITION = Gf.Vec3d(0.5, 0.0, 1.5)
OVERHEAD_CAM_ROTATION = Gf.Vec3f(90.0, 0.0, 0.0)   # X+90 → 아래를 향함
OVERHEAD_CAM_FOCAL_LENGTH = 18.0

WRIST_CAM_OFFSET = Gf.Vec3d(0.0, 0.0, 0.08)
WRIST_CAM_ROTATION = Gf.Vec3f(0.0, 90.0, 0.0)      # Y+90 → 손가락 방향(아래)을 향함
WRIST_CAM_FOCAL_LENGTH = 24.0

CAMERA_APERTURE = 36.0


def build_scene(world: World) -> dict:
    """전체 씬을 구성한다.

    Args:
        world: Isaac Sim World 인스턴스.

    Returns:
        씬 객체 참조 딕셔너리:
            - "franka": SingleManipulator
            - "cube": DynamicCuboid
            - "table": FixedCuboid
            - "camera_paths": {"overhead": str, "wrist": str}
    """
    world.scene.add_default_ground_plane()

    table = _add_table(world)
    pedestal = _add_pedestal(world)
    franka = _add_franka(world)
    cube = _add_cube(world)
    camera_paths = _setup_cameras()

    return {
        "franka": franka,
        "cube": cube,
        "table": table,
        "pedestal": pedestal,
        "camera_paths": camera_paths,
    }


def _add_table(world: World) -> FixedCuboid:
    """작업대 배치."""
    return world.scene.add(
        FixedCuboid(
            prim_path="/World/Table",
            name="table",
            position=TABLE_POSITION,
            scale=TABLE_SCALE,
            size=1.0,
            color=TABLE_COLOR,
        )
    )


def _add_pedestal(world: World) -> FixedCuboid:
    """로봇 받침대 배치."""
    return world.scene.add(
        FixedCuboid(
            prim_path="/World/Pedestal",
            name="pedestal",
            position=PEDESTAL_POSITION,
            scale=PEDESTAL_SCALE,
            size=1.0,
            color=PEDESTAL_COLOR,
        )
    )


def _add_franka(world: World) -> SingleManipulator:
    """Franka Panda 배치 (작업대 높이에 맞춰 올림)."""
    assets_root = get_assets_root_path()
    franka_usd = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

    from isaacsim.core.utils.stage import add_reference_to_stage
    add_reference_to_stage(usd_path=franka_usd, prim_path="/World/Franka")

    gripper = ParallelGripper(
        end_effector_prim_path="/World/Franka/panda_rightfinger",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=GRIPPER_OPEN,
        joint_closed_positions=GRIPPER_CLOSED,
        action_deltas=GRIPPER_DELTAS,
    )

    franka = world.scene.add(
        SingleManipulator(
            prim_path="/World/Franka",
            name="franka",
            end_effector_prim_path="/World/Franka/panda_rightfinger",
            gripper=gripper,
            position=FRANKA_POSITION,
        )
    )

    return franka


def _add_cube(world: World) -> DynamicCuboid:
    """pick 대상 큐브 배치."""
    return world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cube",
            name="cube",
            position=CUBE_DEFAULT_POSITION,
            size=CUBE_SIZE,
            color=CUBE_COLOR,
        )
    )


def _setup_cameras() -> dict:
    """overhead 카메라 + wrist 카메라 배치.

    Returns:
        {"overhead": Camera, "wrist": prim_path}
    """
    stage = get_current_stage()
    result = {}

    # Overhead 카메라 — 작업대 위에서 아래를 내려다봄
    # Isaac Sim Camera convention: [0, 90, 0] = Y축 90도 → 아래를 봄
    overhead_cam = Camera(
        prim_path="/World/OverheadCamera",
        position=np.array([0.5, 0.0, 2.5]),
        frequency=20,
        resolution=(640, 480),
        orientation=rot_utils.euler_angles_to_quats(
            np.array([0, 90, 0]), degrees=True
        ),
    )
    result["overhead"] = overhead_cam

    # Wrist 카메라 — Franka panda_hand의 자식으로 마운트 (로봇과 함께 이동)
    wrist_path = "/World/Franka/panda_hand/WristCamera"
    wrist_cam = UsdGeom.Camera.Define(stage, wrist_path)
    wrist_cam.GetFocalLengthAttr().Set(WRIST_CAM_FOCAL_LENGTH)
    wrist_cam.GetHorizontalApertureAttr().Set(CAMERA_APERTURE)
    xf = UsdGeom.Xformable(wrist_cam.GetPrim())
    xf.AddTranslateOp().Set(WRIST_CAM_OFFSET)
    # panda_hand 로컬 좌표에서 카메라가 손가락 방향(아래)을 봐야 함
    xf.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))
    result["wrist"] = wrist_path

    return result
