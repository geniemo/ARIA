"""ReAct 루프 구조화 로그 — 추후 웹 대시보드 연동용."""

import json
import time
from dataclasses import dataclass, field, asdict


@dataclass
class RecoveryStep:
    """ReAct 루프의 단일 스텝."""
    step_type: str  # "llm_reasoning", "tool_call", "tool_result"
    content: str
    tool_name: str | None = None
    tool_args: dict | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class RecoveryLog:
    """하나의 복구 세션 전체 로그."""
    steps: list[RecoveryStep] = field(default_factory=list)
    overhead_image_b64: str = ""
    wrist_image_b64: str = ""
    success: bool = False
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0

    def add_llm_reasoning(self, content: str) -> None:
        self.steps.append(RecoveryStep(step_type="llm_reasoning", content=content))

    def add_tool_call(self, tool_name: str, tool_args: dict) -> None:
        self.steps.append(RecoveryStep(
            step_type="tool_call",
            content=f"Calling {tool_name}",
            tool_name=tool_name,
            tool_args=tool_args,
        ))

    def add_tool_result(self, tool_name: str, content: str) -> None:
        self.steps.append(RecoveryStep(
            step_type="tool_result",
            content=content,
            tool_name=tool_name,
        ))

    def finalize(self, success: bool) -> None:
        self.success = success
        self.end_time = time.time()

    def to_dict(self) -> dict:
        return {
            "steps": [asdict(s) for s in self.steps],
            "overhead_image_b64": self.overhead_image_b64[:50] + "..." if self.overhead_image_b64 else "",
            "wrist_image_b64": self.wrist_image_b64[:50] + "..." if self.wrist_image_b64 else "",
            "success": self.success,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": round(self.end_time - self.start_time, 2) if self.end_time else 0,
            "total_steps": len(self.steps),
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
