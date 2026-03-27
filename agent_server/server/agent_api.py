"""Agent Server API — POST /anomaly → ReAct 루프 실행."""

from langchain_core.messages import HumanMessage

from contracts.schemas import AnomalyReport
from agent_server.agents.state import AgentState
from agent_server.agents.graph import build_graph


graph = build_graph()


def run_recovery(report: AnomalyReport) -> dict:
    """anomaly report를 받아 ReAct 루프를 실행하고 결과를 반환."""

    # 초기 메시지: 이미지 + 센서 + 로그를 LLM에 전달
    initial_message = HumanMessage(
        content=[
            {"type": "text", "text": _format_anomaly_text(report)},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{report.overhead_image}"},
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{report.wrist_image}"},
            },
        ],
    )

    initial_state: AgentState = {
        "messages": [initial_message],
        "anomaly_report": report.model_dump(),
        "recovery_attempts": 0,
    }

    # ReAct 루프 실행
    final_state = graph.invoke(initial_state)

    # 마지막 메시지에서 결과 추출
    last_message = final_state["messages"][-1]
    return {
        "result": last_message.content,
        "total_messages": len(final_state["messages"]),
    }


def _format_anomaly_text(report: AnomalyReport) -> str:
    """anomaly report를 LLM이 읽기 좋은 텍스트로 변환."""
    log_text = ""
    for entry in report.execution_log:
        log_text += (
            f"  - {entry.phase.value}: {entry.status.value}"
            f" ({entry.duration_steps} steps"
            f", gripper_width={entry.gripper_width_final}"
            f", reason={entry.reason})\n"
        )

    return f"""## Anomaly Report

**Robot State:**
- Gripper width: {report.robot_state.gripper_width:.4f} m
- End-effector position: {report.robot_state.end_effector_position}
- Joint positions: {report.robot_state.joint_positions}

**Execution Log:**
{log_text}

**Images:**
- Image 1: Overhead camera (top-down view of workspace)
- Image 2: Wrist camera (close-up from end-effector)

Analyze the images and execution log. Diagnose the failure and take action to recover."""
