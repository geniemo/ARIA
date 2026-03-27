"""LangGraph 노드 — call_model, call_tool, should_continue."""

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from agent_server.agents.state import AgentState
from agent_server.agents.tools import extract_coordinates, execute_action

TOOLS = [extract_coordinates, execute_action]
tools_by_name = {t.name: t for t in TOOLS}


def call_model(state: AgentState, config: RunnableConfig):
    """LLM 노드: 현재 상태를 보고 진단 + tool 호출 결정."""
    from agent_server.agents.graph import get_model

    model = get_model()
    response = model.invoke(state["messages"], config)
    return {"messages": [response]}


def call_tool(state: AgentState):
    """Tool 노드: LLM이 결정한 tool을 실행."""
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=str(result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


def should_continue(state: AgentState):
    """라우팅: tool 호출이 있으면 계속, 없으면 종료."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"
