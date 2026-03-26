"""ARIA Isaac Sim 엔트리포인트 — 씬 확인용."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World

from scene.scene_builder import build_scene

# World 생성
world = World(stage_units_in_meters=1.0)

# 씬 구성
scene_objects = build_scene(world)
world.reset()

print("=== ARIA Scene Loaded ===")
print(f"Franka position: {scene_objects['franka'].get_world_pose()[0]}")
print(f"Cube position: {scene_objects['cube'].get_world_pose()[0]}")
print(f"Camera paths: {scene_objects['camera_paths']}")

# 시뮬레이션 루프
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
