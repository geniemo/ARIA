"""Agent Server API — POST /anomaly → ReAct 루프 실행."""

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from contracts.schemas import AnomalyReport
from agent_server.agents.state import AgentState
from agent_server.agents.graph import build_graph
from agent_server.agents.recovery_logger import RecoveryLog
from agent_server.agents.tools import set_current_report

graph = build_graph()


def run_recovery(report: AnomalyReport) -> dict:
    """anomaly report를 받아 ReAct 루프를 실행하고 결과를 반환."""
    set_current_report(report)

    recovery_log = RecoveryLog(
        overhead_image_b64=report.overhead_image,
        wrist_image_b64=report.wrist_image,
    )

    initial_message = HumanMessage(
        content=[
            {"type": "text", "text": _format_anomaly_text(report)},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{report.overhead_image}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{report.wrist_image}"}},
        ],
    )

    initial_state: AgentState = {
        "messages": [initial_message],
        "anomaly_report": report.model_dump(),
        "recovery_attempts": 0,
    }

    # 1차 ReAct 루프
    final_state = graph.invoke(initial_state)

    # Gemini가 explore 없이 포기했으면 피드백 후 재시도
    if not _has_explore(final_state) and not _has_success(final_state):
        feedback = HumanMessage(
            content="The cube is confirmed to still be on the table. It is hidden behind the robot arm. "
            "You MUST call execute_action with action='move', "
            "coords={'x': 0.6, 'y': 0.2, 'z': 0.6}, intent='explore' to move the arm. "
            "After the move, call extract_coordinates to find the cube."
        )
        retry_state: AgentState = {
            "messages": final_state["messages"] + [feedback],
            "anomaly_report": initial_state["anomaly_report"],
            "recovery_attempts": 1,
        }
        final_state = graph.invoke(retry_state)

    # 로그 수집
    success = _has_success(final_state)
    _collect_logs(final_state, recovery_log)
    recovery_log.finalize(success=success)

    last_message = final_state["messages"][-1]
    return {
        "result": last_message.content if hasattr(last_message, "content") else str(last_message),
        "success": success,
        "total_steps": len(recovery_log.steps),
        "log": recovery_log.to_dict(),
    }


def _has_success(state) -> bool:
    return any(
        isinstance(msg, ToolMessage) and
        ("'success': true" in msg.content.lower() or "'success': True" in msg.content)
        for msg in state["messages"]
    )


def _has_explore(state) -> bool:
    return any(
        isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and
        any(tc.get("args", {}).get("intent") == "explore" for tc in msg.tool_calls)
        for msg in state["messages"]
    )


def _collect_logs(state, recovery_log: RecoveryLog) -> None:
    for msg in state["messages"]:
        if isinstance(msg, AIMessage):
            recovery_log.add_llm_reasoning(msg.content)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    recovery_log.add_tool_call(tc["name"], tc["args"])
        elif isinstance(msg, ToolMessage):
            recovery_log.add_tool_result(msg.name, msg.content)


def _format_anomaly_text(report: AnomalyReport) -> str:
    log_text = "\n".join(
        f"  - {e.phase.value}: {e.status.value} ({e.duration_steps} steps, "
        f"gripper_width={e.gripper_width_final}, reason={e.reason})"
        for e in report.execution_log
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

**CRITICAL CONSTRAINT**: The cube is GUARANTEED to still be on the table. It CANNOT fall off. If extract_coordinates returns "object not found", the cube is hidden behind the robot arm. You MUST call execute_action with intent="explore" to move the arm and reveal the cube.

Your response MUST include a tool call. Do not respond with only text."""
