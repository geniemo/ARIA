"""Agent Server API — POST /anomaly → ReAct 루프 실행."""

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from contracts.schemas import AnomalyReport
from agent_server.agents.state import AgentState
from agent_server.agents.graph import build_graph
from agent_server.agents.recovery_logger import RecoveryLog


graph = build_graph()


def run_recovery(report: AnomalyReport) -> dict:
    """anomaly report를 받아 ReAct 루프를 실행하고 결과를 반환."""

    recovery_log = RecoveryLog(
        overhead_image_b64=report.overhead_image,
        wrist_image_b64=report.wrist_image,
    )

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

    # 메시지 이력에서 로그 추출
    success = False
    for msg in final_state["messages"]:
        if isinstance(msg, AIMessage):
            recovery_log.add_llm_reasoning(msg.content)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    recovery_log.add_tool_call(tc["name"], tc["args"])
        elif isinstance(msg, ToolMessage):
            recovery_log.add_tool_result(msg.name, msg.content)
            if "'success': True" in msg.content or '"success": true' in msg.content.lower():
                success = True

    recovery_log.finalize(success=success)
    recovery_log.save("/home/park/workspace/ARIA/screenshots/recovery_log.json")

    last_message = final_state["messages"][-1]

    return {
        "result": last_message.content if hasattr(last_message, "content") else str(last_message),
        "success": success,
        "total_steps": len(recovery_log.steps),
        "log": recovery_log.to_dict(),
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
