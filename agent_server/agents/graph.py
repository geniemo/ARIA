"""LangGraph StateGraph 조립 — ReAct 루프."""

import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from agent_server.agents.state import AgentState
from agent_server.agents.nodes import call_model, call_tool, should_continue, TOOLS
from agent_server.prompts.diagnosis import SYSTEM_PROMPT

_model = None


def get_model():
    """tool-bound LLM 인스턴스 반환 (싱글턴)."""
    global _model
    if _model is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0,
            max_retries=2,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
        )
        _model = llm.bind_tools(TOOLS)
    return _model


def build_graph():
    """ReAct 패턴 StateGraph를 조립하고 반환."""
    workflow = StateGraph(AgentState)

    workflow.add_node("llm", call_model)
    workflow.add_node("tools", call_tool)

    workflow.set_entry_point("llm")
    workflow.add_conditional_edges(
        "llm",
        should_continue,
        {"continue": "tools", "end": END},
    )
    workflow.add_edge("tools", "llm")

    return workflow.compile()
