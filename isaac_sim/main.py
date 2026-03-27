"""ARIA Isaac Sim 엔트리포인트 — P&P 실행 + HTTP 서버."""

import argparse

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import omni.replicator.core as rep
from isaacsim.core.api import World

from scene.scene_builder import build_scene, CUBE_DEFAULT_POSITION, PLACE_POSITION
from scene.error_injector import inject_offset, inject_absence
from control.rrt_controller import RRTController
from control.gripper_controller import GripperController
from task.pick_place_task import PickPlaceTask, TaskState
from server.sim_api import (
    start_server,
    get_pending_request,
    send_response,
    send_anomaly_report,
    rgba_to_base64_png,
)
from contracts.schemas import (
    AnomalyReport,
    ExecuteActionResponse,
    RobotState,
)
from contracts.skill_primitives import ActionIntent
from contracts.api_specs import AGENT_BASE_URL


# --- 인자 파싱 ---
parser = argparse.ArgumentParser()
parser.add_argument("--scenario", choices=["normal", "a", "b"], default="normal")
args, _ = parser.parse_known_args()

# --- World + Scene ---
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
scene_objects = build_scene(world)

franka = scene_objects["franka"]
franka.gripper.set_default_state(franka.gripper.joint_opened_positions)

world.reset()
franka.initialize()

cameras = scene_objects["camera_paths"]
cameras["overhead"].initialize()

# Overhead + Wrist — replicator annotator로 통일
oh_rp = rep.create.render_product("/World/OverheadCamera", (640, 480))
oh_ann = rep.AnnotatorRegistry.get_annotator("rgb")
oh_ann.attach([oh_rp])

wrist_rp = rep.create.render_product(cameras["wrist"], (640, 480))
wrist_ann = rep.AnnotatorRegistry.get_annotator("rgb")
wrist_ann.attach([wrist_rp])

cube = scene_objects["cube"]


def capture_images() -> tuple[str, str]:
    """overhead + wrist 카메라 이미지를 base64 PNG로 캡처."""
    for _ in range(30):
        world.step(render=True)

    oh_data = oh_ann.get_data()
    overhead_b64 = rgba_to_base64_png(oh_data) if oh_data is not None else ""

    wrist_data = wrist_ann.get_data()
    wrist_b64 = rgba_to_base64_png(wrist_data) if wrist_data is not None else ""

    return overhead_b64, wrist_b64


def get_robot_state() -> RobotState:
    """현재 로봇 상태."""
    ee_pos, _ = franka.end_effector.get_world_pose()
    joint_positions = franka.get_joint_positions().tolist()
    gripper_positions = franka.gripper.get_joint_positions()
    return RobotState(
        gripper_width=float(gripper_positions[0] + gripper_positions[1]),
        end_effector_position=ee_pos.tolist(),
        joint_positions=joint_positions[:7],
    )


def run_pick_place(
    grasp_position: np.ndarray,
    place_position: np.ndarray,
    grasp_orientation: np.ndarray,
) -> tuple[TaskState, list]:
    """P&P 실행. (state, execution_log) 반환."""
    # gripper 완전히 열기 (이전 시도에서 닫혀있을 수 있음)
    gripper_ctrl_tmp = GripperController(franka.gripper)
    gripper_ctrl_tmp.open()
    while not gripper_ctrl_tmp.is_done():
        franka.gripper.open()
        world.step(render=True)

    rrt = RRTController(robot_articulation=franka, physics_dt=1.0 / 60.0)
    gripper_ctrl = GripperController(franka.gripper)

    task = PickPlaceTask(
        franka=franka,
        rrt=rrt,
        gripper_ctrl=gripper_ctrl,
        grasp_position=grasp_position,
        place_position=place_position,
        grasp_orientation=grasp_orientation,
    )
    task.start()

    while simulation_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            continue

        state = task.step()
        if state in (TaskState.SUCCESS, TaskState.ANOMALY):
            return state, task.execution_log

    return TaskState.ANOMALY, task.execution_log


def handle_execute_action(request) -> ExecuteActionResponse:
    """Agent의 execute_action 요청을 처리."""
    target_pos = np.array([request.coords["x"], request.coords["y"], request.coords["z"]])
    ee_ori = np.array([0.0, 1.0, 0.0, 0.0])

    if request.intent == ActionIntent.RECOVER:
        state, logs = run_pick_place(target_pos, PLACE_POSITION.copy(), ee_ori)
    else:  # EXPLORE
        # 이동만 수행 (grasp 없이)
        rrt = RRTController(robot_articulation=franka, physics_dt=1.0 / 60.0)
        success = rrt.compute_plan(target_pos, ee_ori)
        if success:
            while not rrt.is_done():
                action = rrt.step()
                if action:
                    franka.get_articulation_controller().apply_action(action)
                world.step(render=True)
        state = TaskState.SUCCESS
        logs = []

    overhead_b64, wrist_b64 = capture_images()

    return ExecuteActionResponse(
        success=(state == TaskState.SUCCESS),
        gripper_width=get_robot_state().gripper_width,
        robot_state=get_robot_state(),
        overhead_image=overhead_b64,
        wrist_image=wrist_b64,
        execution_log=logs,
    )


# --- 에러 주입 ---
if args.scenario == "a":
    inject_offset(cube, offset_xy=np.array([0.04, 0.03]))
elif args.scenario == "b":
    inject_absence(cube)

for _ in range(10):
    world.step(render=True)

# --- HTTP 서버 시작 ---
start_server()

# --- 초기 P&P 실행 ---
ee_ori = np.array([0.0, 1.0, 0.0, 0.0])
state, logs = run_pick_place(CUBE_DEFAULT_POSITION.copy(), PLACE_POSITION.copy(), ee_ori)

if state == TaskState.ANOMALY:
    # anomaly 발생 → agent에 보고
    overhead_b64, wrist_b64 = capture_images()
    report = AnomalyReport(
        overhead_image=overhead_b64,
        wrist_image=wrist_b64,
        robot_state=get_robot_state(),
        execution_log=logs,
    )
    try:
        send_anomaly_report(report, AGENT_BASE_URL)
    except Exception:
        pass  # agent 서버가 없으면 무시

    # agent의 execute_action 요청 대기
    while simulation_app.is_running():
        world.step(render=True)
        if not world.is_playing():
            continue

        request = get_pending_request()
        if request is not None:
            response = handle_execute_action(request)
            send_response(response)

            if response.success:
                break  # 복구 성공

else:
    # 정상 완료 — 시뮬레이션 유지
    while simulation_app.is_running():
        world.step(render=True)

simulation_app.close()
