"""ARIA Agent 상태 정의."""

from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """ARIA Agent의 상태.

    messages: LLM과 tool 간 대화 이력 (add_messages reducer로 자동 누적)
    anomaly_report: Isaac Sim에서 수신한 이상 보고 (dict)
    recovery_attempts: 복구 시도 횟수
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    anomaly_report: dict
    recovery_attempts: int
