# ARIA — Autonomous Recovery with Intelligent Agent

> System Architecture Document v3.0
> 2026.03 | 박지원 | 성균관대학교 소프트웨어학과 졸업작품

---

## 1. 프로젝트 개요

ARIA는 Isaac Sim 환경에서 협동로봇(Franka Panda)의 Pick-and-Place 작업 중 발생하는 이상 상황을 진단하고 자율적으로 복구하는 시스템이다.

**핵심 주장:** MLLM이 고수준 진단과 전략 수립을 담당하고, rule-based 감지와 비전 알고리즘이 저수준 실행을 담당하는 하이브리드 아키텍처. Agent는 LangGraph ReAct 패턴으로 tool을 호출하며 복구 루프를 주도한다.

### 1.1 기술 스택

| 구분 | 기술 |
|------|------|
| 시뮬레이션 | NVIDIA Isaac Sim 5.1.0 |
| MLLM | Gemini 3.1 Pro (`langchain-google-genai`) |
| Agent 프레임워크 | LangGraph (ReAct 패턴, StateGraph) |
| Agent 상태 관리 | TypedDict + `add_messages` reducer |
| API 서버 | FastAPI + Pydantic |
| 비전 | OpenCV (좌표 추출) |
| 로봇 | Franka Panda (SingleManipulator + ParallelGripper) |
| 로봇 제어 | Lula RRT (trajectory planning) + ArticulationTrajectory (실행) + ArticulationAction (gripper) |

### 1.2 모델 선정 근거

Gemini Robotics-ER 1.5가 로보틱스 특화 모델로 존재하나, ARIA에는 Gemini 3.1 Pro가 더 적합하다:

- **Robotics-ER은 one-shot planning에 특화.** 이미지를 보고 전체 action sequence를 한 번에 생성하는 구조. ARIA는 실행 결과를 관찰하고 재판단하는 iterative ReAct 루프가 핵심이므로 범용 reasoning 모델이 적합.
- **Gemini 3.1 Pro는 최신 모델.** agentic capability, function calling, multimodal reasoning에서 최상위 성능. Robotics-ER 1.5는 2025년 9월 업데이트로 상대적으로 오래됨.
- **교체 가능한 설계.** `ChatGoogleGenerativeAI`의 model 파라미터만 변경하면 다른 모델로 교체 가능. 아키텍처가 특정 모델에 종속되지 않음.

### 1.3 로봇 제어 방식: Lula RRT

PickPlaceController(Isaac Sim 내장)와 RMPFlow 대신 **Lula RRT trajectory planner**로 pick-and-place를 구현한다.

RMPFlow(reactive controller)는 매 스텝 반응적으로 action을 계산하여 경로를 미리 계획하지 않으므로, 특정 위치+orientation 조합에서 관절이 불안정하게 흔들리거나 작업대와 충돌하는 문제가 발생했다. RRT는 **경로를 미리 계획**하고 보간된 trajectory를 따라가므로 부드럽고 안정적인 동작이 가능하다.

- **경로 사전 계획.** RRT가 충돌 없는 경로를 탐색하고, `LulaCSpaceTrajectoryGenerator`로 time-optimal trajectory를 생성. `ArticulationTrajectory`로 변환하여 매 스텝 action sequence를 실행.
- **매 스텝 체크 가능.** action sequence를 한 스텝씩 실행하므로, RMPFlow와 동일하게 매 스텝 센서 확인 및 중단이 가능.
- **phase별 체크포인트.** 각 phase 완료 후 센서 상태를 확인하고, 문제 발생 시 중단 → agent에 보고.
- **gripper 독립 제어.** `ArticulationAction`으로 gripper open/close를 직접 명령.
- **approach + descend 패턴.** 각 이동 phase는 목표 위 높은 위치로 먼저 이동 후 내려가는 2단계로 실행하여 작업대 충돌을 방지.

**Pick-and-Place phase 및 완료 조건:**

| Phase | 동작 | 완료 조건 | 이상 감지 |
|-------|------|----------|----------|
| approach | (1) 큐브 위 높은 위치로 이동 (2) grasp 위치로 하강 | trajectory 실행 완료 | 경로 계획 실패 시 ANOMALY |
| close gripper | gripper 닫기 | gripper 동작 완료 (더 이상 움직이지 않음) | **gripper width < 임계값 → ANOMALY** |
| lift | 물체를 들어올림 | trajectory 실행 완료 | - |
| move | (1) place 위 높은 위치로 이동 (2) place 위치로 하강 | trajectory 실행 완료 | - |
| open gripper | gripper 열기 | gripper 동작 완료 | - |

이상 감지 체크포인트는 **close gripper 완료 직후**. gripper width가 물체 크기에 근접하면 정상, 임계값 미만이면 ANOMALY로 판정하고 이후 phase를 중단한다.

---

## 2. 시스템 아키텍처

### 2.1 하이브리드 아키텍처 설계 원칙

시스템은 세 계층으로 분리된다:

**Rule-based 계층:** 이상 감지(gripper width 임계값) 및 로봇 실행(RMPFlow + gripper). 빠르고 결정적이며, 사전 정의된 규칙만 처리한다. **실행 결과의 성공/실패 판정도 이 계층에서 수행한다** (gripper width 기반).

**MLLM 계층:** Gemini 3.1 Pro가 카메라 이미지와 센서 데이터를 받아 상황을 진단하고 복구 전략을 수립한다. 고수준 추론만 담당하며, 정밀 좌표나 로봇 제어, 성공/실패 판정에는 관여하지 않는다.

**비전 알고리즘 계층:** 이미지에서 물체의 정밀 좌표를 추출한다. MLLM이 "물체가 보인다"는 판단을 내리면, 이 계층이 구체적인 (x, y, z) 좌표로 변환한다.

### 2.2 통신 구조

```
Isaac Sim (:8000)                    Agent Server (:8001)
┌─────────────────┐                 ┌──────────────────────────┐
│                  │  anomaly 발생   │                          │
│  Rule-based     │ ──────────────> │  LangGraph ReAct Agent   │
│  이상 감지       │  로그, 센서,    │                          │
│                  │  이미지 ×2      │  ┌──────────────────┐   │
│  로봇 정지       │                 │  │ MLLM (Gemini 3.1) │   │
│  (시뮬레이션은   │                 │  └────────┬─────────┘   │
│   계속 구동)     │                 │           │              │
│                  │  execute_action │  ┌────────▼─────────┐   │
│  HTTP 서버      │ <────────────── │  │ Tool 호출 결정     │   │
│  (항상 활성)     │  action+coords  │  │ - extract_coords  │   │
│                  │  +intent        │  │ - execute_action  │   │
│                  │                 │  └──────────────────┘   │
│  action 실행    │                 │                          │
│  결과 판정      │ ──────────────> │  LLM이 응답을 관찰하고   │
│  (rule-based)   │  success/fail   │  다음 행동을 결정        │
│                  │  +이미지+센서   │                          │
│                  │  +실행 로그     │                          │
└─────────────────┘                 └──────────────────────────┘
```

**역할 분리 원칙:**
- Isaac Sim이 "성공했다/실패했다"를 판정한다 (사실 판정, rule-based).
- Agent가 "왜 실패했고 다음에 뭘 할지"를 추론한다 (원인 분석 + 전략 수립, MLLM).

**데드락 방지:** Isaac Sim은 anomaly 발생 시 로봇만 정지하고 시뮬레이션은 계속 구동된다. HTTP 서버가 항상 활성 상태이므로 agent의 tool 호출을 언제든 수신할 수 있다.

### 2.3 execute_action 응답 스펙

Agent가 실패 원인을 정확히 추론하려면 실행 과정 정보가 필요하다. `execute_action`의 응답에는 최종 결과뿐 아니라 **phase별 실행 로그**가 포함된다.

```json
{
    "success": false,
    "gripper_width": 0.008,
    "sensor_state": { ... },
    "overhead_image": "base64...",
    "wrist_image": "base64...",
    "execution_log": [
        {"phase": "approach", "status": "completed", "duration_steps": 120},
        {"phase": "close_gripper", "status": "completed", "gripper_width_final": 0.008},
        {"phase": "lift", "status": "aborted", "reason": "gripper_width < threshold"}
    ]
}
```

이를 통해 MLLM은 "approach까지는 정상이었으나 grasp에서 물체를 못 잡았고 lift에서 중단됐다"와 같은 맥락 기반 진단이 가능하다. 이미지만으로는 알 수 없는 정보(어느 phase에서 실패했는지, 물체가 미끄러졌는지)를 로그가 보완한다.

---

## 3. Agent 설계

### 3.1 ReAct 패턴 (LangGraph 구현)

Agent는 LangGraph의 `StateGraph`로 구현된 ReAct(Reason + Act) 패턴을 따른다.

```
Reason(진단) → Act(tool 호출) → Observe(결과 확인) → Reason(재판단) → ...
```

각 단계의 역할:
- **Reason**: MLLM이 현재 상태(이미지 + 센서 + 실행 로그 + 이전 tool 결과)를 보고 진단 및 다음 행동을 결정.
- **Act**: 결정에 따라 tool을 호출 (`extract_coordinates` 또는 `execute_action`).
- **Observe**: tool 응답을 수신. `execute_action`의 경우 Isaac Sim이 판정한 `success` 플래그 + 최신 이미지 + 센서 상태 + 실행 로그가 포함됨.
- **Reason (재판단)**: `success=true`면 tool 호출 없이 종료. `success=false`면 이미지 + 로그를 보고 원인 분석 후 다음 tool 호출.

### 3.2 Agent 상태 정의

```python
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """ARIA Agent의 상태."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    anomaly_report: dict          # Isaac Sim에서 수신한 이상 보고
    recovery_attempts: int        # 복구 시도 횟수
```

### 3.3 모델 및 Tool 바인딩

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-pro",
    temperature=0,
    max_retries=2,
)

model = llm.bind_tools([extract_coordinates, execute_action])
```

### 3.4 Tool 정의

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class ExtractCoordinatesInput(BaseModel):
    image_base64: str = Field(description="Base64 encoded overhead camera image")

@tool("extract_coordinates", args_schema=ExtractCoordinatesInput)
def extract_coordinates(image_base64: str) -> dict:
    """이미지에서 물체의 정밀 (x, y, z) 월드 좌표를 추출한다.
    OpenCV 기반 비전 알고리즘으로 Agent 서버 내부에서 처리."""
    ...

class ExecuteActionInput(BaseModel):
    action: str = Field(description="Action type: grasp, move, etc.")
    coords: dict = Field(description="Target coordinates {x, y, z}")
    intent: str = Field(description="explore (정보 수집) or recover (실제 복구)")

@tool("execute_action", args_schema=ExecuteActionInput)
def execute_action(action: str, coords: dict, intent: str) -> dict:
    """Isaac Sim에 HTTP 요청을 보내 로봇 동작을 실행한다.
    Isaac Sim이 action을 수행하고 성공/실패를 판정하여
    결과 + 이미지 + 센서 상태 + 실행 로그를 반환."""
    ...
```

| Tool | 위치 | 설명 |
|------|------|------|
| `extract_coordinates` | 로컬 | 이미 수신한 이미지에서 OpenCV 기반으로 물체의 정밀 (x, y, z) 월드 좌표를 추출. Agent 서버 내부 처리. |
| `execute_action` | 원격 | Isaac Sim에 HTTP 요청. `action` + `coords` + `intent`를 전송. Isaac Sim이 실행 후 **성공/실패 판정 + 최신 이미지 + 센서 상태 + 실행 로그**를 응답으로 반환. |

### 3.5 Intent 구분

`execute_action`의 `intent` 필드는 Isaac Sim이 실행 후 후속 동작을 결정하는 데 사용된다:

| Intent | Isaac Sim 동작 | 용도 |
|--------|---------------|------|
| **explore** | action 실행 → 상태 + 이미지 + 로그 반환 → 대기 | 추가 정보 수집 (탐색, 이동 후 관찰) |
| **recover** | action 실행 → 성공 시 full P&P 수행, 실패 시 `success=false` + 로그 반환 | 실제 복구 시도 (grasp → lift → move → place) |

### 3.6 그래프 구성

```python
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

tools_by_name = {t.name: t for t in [extract_coordinates, execute_action]}

def call_model(state: AgentState, config: RunnableConfig):
    """LLM 노드: 진단 + tool 호출 결정."""
    response = model.invoke(state["messages"], config)
    return {"messages": [response]}

def call_tool(state: AgentState):
    """Tool 노드: LLM이 결정한 tool을 실행."""
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

def should_continue(state: AgentState):
    """라우팅: LLM 출력에 tool 호출이 있으면 계속, 없으면 종료."""
    if not state["messages"][-1].tool_calls:
        return "end"
    return "continue"

# 그래프 조립
workflow = StateGraph(AgentState)
workflow.add_node("llm", call_model)
workflow.add_node("tools", call_tool)
workflow.set_entry_point("llm")
workflow.add_conditional_edges("llm", should_continue, {"continue": "tools", "end": END})
workflow.add_edge("tools", "llm")
graph = workflow.compile()
```

`should_continue`는 별도의 판단 로직이 아니라, LLM의 출력을 그래프 경로로 변환하는 라우터다. LLM이 tool을 호출했으면 → tools 노드로, 호출하지 않았으면 → 종료.

```
         ┌─────┐
         │ llm │ ◄──────────────┐
         └──┬──┘                │
            │                   │
     ┌──────▼──────┐            │
     │should_continue│           │
     └──────┬──────┘            │
       ┌────┴────┐              │
       │         │              │
   continue     end             │
       │         │              │
  ┌────▼───┐  ┌──▼──┐          │
  │ tools  │  │ END │          │
  └────┬───┘  └─────┘          │
       │                       │
       └───────────────────────┘
```

### 3.7 Agent 흐름 (종합)

```
[Isaac Sim] anomaly 감지 → 로봇 정지
    │
    ▼ POST /anomaly: 로그 + 센서 + overhead 이미지 + wrist 이미지
    │
[Agent] LLM 노드 (1회차): Gemini 3.1 Pro 진단
    │
    ├─ 물체 보임 → extract_coordinates (로컬 좌표 추출)
    │              → execute_action(grasp, coords, intent=recover)
    │
    └─ 물체 안 보임 → execute_action(move, coords, intent=explore)
    │
    ▼
[Isaac Sim] action 실행 → 결과 판정 (rule-based)
    │
    ▼ 응답: { success, gripper_width, sensor_state, images, execution_log }
    │
[Agent] LLM 노드 (2회차~): 결과 + 로그 관찰 + 재판단
    │
    ├─ success=true → tool 호출 없음 → should_continue → END
    │
    ├─ success=false → 로그 + 이미지로 "왜 실패했는가?" 재진단
    │   → 새 이미지 기반으로 extract_coordinates 또는 execute_action 재호출
    │   → tools 노드 → llm 노드 (루프)
    │
    └─ intent=explore 후 → 새 이미지에서 물체 발견 여부 판단
        → 발견: extract_coordinates → execute_action(intent=recover)
        → 미발견: execute_action(move, 다른 위치, intent=explore)
```

---

## 4. 시나리오

### 4.1 이상 감지 (Rule-based)

| 항목 | 정상 | 이상 |
|------|------|------|
| Gripper width | 물체 크기에 근접 (> 임계값) | < 임계값 |
| 판정 | OK → P&P 계속 | ANOMALY → 로봇 정지 → Agent 호출 |

RRT 기반 pick-and-place에서 grasp phase 완료 후 gripper width를 체크한다. 물체를 잡았으면 물체 크기만큼에서 멈추고, 못 잡았으면 끝까지 닫힌다.

### 4.2 시나리오 A — 물체 위치 오프셋

| 항목 | 설명 |
|------|------|
| 에러 조건 | 물체가 원래 위치에서 오프셋되어 grasp 실패 |
| 감지 | gripper width < 임계값 → ANOMALY |
| Agent 행동 | LLM 진단(이미지+로그) → `extract_coordinates` → `execute_action(grasp, coords, intent=recover)` |
| Isaac Sim 응답 | `success=true` + 실행 로그 → LLM이 tool 미호출 → END |

### 4.3 시나리오 B — 물체 부재 (이동됨)

| 항목 | 설명 |
|------|------|
| 에러 조건 | 물체가 작업 영역 밖으로 이동하여 빈 곳을 grasp |
| 감지 | gripper width < 임계값 → ANOMALY |
| Agent 행동 (1) | LLM 진단(이미지+로그): "물체 안 보임" → `execute_action(move, search_coords, intent=explore)` |
| Isaac Sim 응답 (1) | 이동 완료 + 새 이미지 + 실행 로그 반환 |
| Agent 행동 (2) | LLM 재진단: 새 이미지에서 물체 발견 → `extract_coordinates` → `execute_action(grasp, coords, intent=recover)` |
| Isaac Sim 응답 (2) | `success=true` + 실행 로그 → END |

---

## 5. 코드 구조

```
contracts/
├── schemas.py                   # Pydantic 모델 (RobotState[gripper_width, ee_pos, joint_positions], AnomalyReport, ExecutionLog 등)
├── skill_primitives.py          # SkillName enum
├── api_specs.py                 # 포트/엔드포인트 상수
└── __init__.py

isaac_sim/
├── main.py                      # 엔트리포인트 (--scenario a|b)
├── scene/
│   ├── scene_builder.py         # Franka + 작업대 + 물체 + overhead/wrist 카메라
│   └── error_injector.py        # 시나리오 A/B 에러 주입
├── control/
│   ├── rrt_controller.py        # Lula RRT trajectory planning + ArticulationTrajectory 실행
│   └── gripper_controller.py    # ArticulationAction 기반 gripper 제어
├── sensors/
│   ├── anomaly_detector.py      # gripper width 기반 이상 감지
│   └── camera_capture.py        # 카메라 → base64 PNG
├── server/
│   └── sim_api.py               # HTTP 서버 (execute_action 수신 + 응답)
└── task/
    └── pick_place_task.py       # P&P phase 관리 + 실행 로그 기록 + 이상 감지

agent_server/
├── main.py                      # uvicorn :8001
├── server/
│   └── agent_api.py             # POST /anomaly → Agent 루프 시작
├── agents/
│   ├── graph.py                 # LangGraph StateGraph (ReAct 루프)
│   ├── state.py                 # AgentState 정의
│   ├── nodes.py                 # call_model, call_tool, should_continue
│   └── tools.py                 # extract_coordinates, execute_action
├── vision/
│   └── object_detector.py       # overhead/wrist 이미지에서 물체 좌표 검출
└── prompts/
    └── diagnosis.py             # Gemini 시스템 프롬프트
```

---

## 6. 설계 판단 근거

### 6.1 왜 MLLM인가?

현재 시나리오(A, B)는 rule-based로 대체 가능하다. 본 프로젝트의 핵심 기여는 MLLM 자체의 우수성이 아니라, **진단→전략→실행이 분리된 Agentic 복구 파이프라인의 설계와 구현**이다.

MLLM은 이 파이프라인의 판단 모듈로 사용되며, `ChatGoogleGenerativeAI`의 model 파라미터만 변경하면 아키텍처 수정 없이 교체 가능하다.

### 6.2 MLLM의 한계와 대응

| 한계 | 대응 |
|------|------|
| 정밀 좌표 출력 불가 | 비전 알고리즘(`extract_coordinates`)으로 분리 |
| 응답 지연 (2~10초) | 실시간 제어가 아닌 이상 복구 상황에만 개입 |
| 출력 불안정 (JSON 파싱 실패 등) | Pydantic 스키마로 출력 구조화, 재시도 로직 |

### 6.3 왜 Isaac Sim이 성공/실패를 판정하는가?

"gripper가 물체를 잡았는가"는 gripper width 수치 비교로 확정적으로 판단 가능하다. 이를 MLLM에 맡기면 불필요한 지연과 불확실성이 추가된다.

- **Isaac Sim (rule-based):** "성공했다 / 실패했다" — 사실 판정
- **Agent (MLLM):** "왜 실패했고 다음에 뭘 할지" — 원인 분석 + 전략 수립

### 6.4 왜 intent 구분인가?

`execute_action`에 `intent` 필드를 두어 탐색(explore)과 복구(recover)를 명시적으로 구분한다:

- `explore`: 이동 후 상태만 반환. P&P 파이프라인을 시작하지 않음.
- `recover`: grasp 성공 시 lift → move → place까지 full 파이프라인 수행.

### 6.5 왜 Lula RRT인가?

PickPlaceController(Isaac Sim 내장)는 phase를 통으로 실행하여 mid-phase 개입이 제한적이다. RMPFlow(reactive controller)는 매 스텝 제어가 가능하지만, 경로를 미리 계획하지 않아 특정 위치+orientation 조합에서 관절이 불안정하게 흔들리거나 작업대와 충돌하는 문제가 발생했다.

Lula RRT는 경로를 사전에 계획하고, 보간된 trajectory를 `ArticulationTrajectory`로 변환하여 매 스텝 실행한다:

- 충돌 없는 부드러운 경로 → 관절 불안정 문제 해소
- action sequence를 한 스텝씩 실행 → phase별 체크포인트 삽입 용이
- gripper를 `ArticulationAction`으로 독립 제어 → grasp 성공/실패 판정이 자연스러움
- 경로 계획 실패 시 agent에 보고 → 다른 복구 전략 시도 가능

### 6.6 왜 실행 로그를 응답에 포함하는가?

이미지만으로는 "어느 phase에서 실패했는지", "물체가 미끄러졌는지" 등을 알 수 없다. phase별 실행 로그(status, duration, gripper_width 등)를 함께 전송하면 MLLM의 진단 정확도가 향상된다. 이미지는 "현재 어떤 상태인가"를, 로그는 "어떤 과정을 거쳐 이 상태에 도달했는가"를 제공한다.

---

## 7. 개발 계획

### Phase 0: 환경 세팅

- [O] Isaac Sim 5.1.0 프로젝트 디렉토리 초기화
- [O] contracts/ 패키지 생성 (schemas.py, api_specs.py, skill_primitives.py)
- [O] Pydantic 모델 정의: RobotState, AnomalyReport, ExecuteActionRequest, ExecuteActionResponse, ExecutionLog

### Phase 1: Isaac Sim — Lula RRT 기반 Pick-and-Place

**1-1. Scene 구성**
- [O] scene_builder.py: Franka + 받침대 + 작업대 + ground plane 배치
- [O] scene_builder.py: 큐브(물체) 배치
- [O] scene_builder.py: overhead 카메라 배치 (Camera API, 아래 방향)
- [O] scene_builder.py: wrist 카메라 마운트 (panda_hand 자식, 튜닝 보류)

**1-2. 모션 제어**
- [O] rrt_controller.py: Lula RRT + ArticulationTrajectory 기반 경로 계획 + 실행
- [O] gripper_controller.py: ArticulationAction으로 gripper open/close + settle 판정

**1-3. Pick-and-Place phase 관리 + 이상 감지**
- [O] pick_place_task.py: approach phase — (1) 큐브 위 높은 위치 → (2) grasp 위치로 하강
- [O] pick_place_task.py: close gripper phase + **ANOMALY 체크포인트** (gripper width < 임계값)
- [O] pick_place_task.py: lift phase — 물체 들어올림
- [O] pick_place_task.py: move phase — (1) place 위 높은 위치 → (2) place 위치로 하강
- [O] pick_place_task.py: open gripper phase
- [O] pick_place_task.py: phase별 실행 로그 기록 (ExecutionLog)

**1-4. 이상 감지 + 로그**
- [O] pick_place_task.py: close gripper 완료 직후 gripper width 체크 → 임계값 미만이면 ANOMALY
- [O] pick_place_task.py: 경로 계획 실패 시 ANOMALY (reason 기록)
- [O] pick_place_task.py: phase별 실행 로그 기록 (phase명, status, duration_steps, gripper_width 등)

**1-5. 검증**
- [O] 정상 pick-and-place end-to-end 동작 검증 (approach → close → lift → move → open)
- [O] 9개 케이스 검증 (위치/방향/복합 변경) — 전부 성공, place 오차 < 1.5mm
- [O] 물체 없이 grasp 시도 → anomaly 감지 동작 확인 (width=0.0 → ANOMALY)

### Phase 2: Isaac Sim — 에러 주입 + HTTP 서버

- [O] error_injector.py: 시나리오 A (물체 오프셋) 에러 주입
- [ ] error_injector.py: 시나리오 B (물체 이동) 에러 주입
- [ ] anomaly 감지 시 로봇 정지 + 카메라 캡처 동작 확인
- [ ] sim_api.py: FastAPI HTTP 서버 (:8000) 구현
- [ ] sim_api.py: POST /anomaly → agent에 anomaly report 전송
- [ ] sim_api.py: POST /execute_action → action 실행 → 결과 + 이미지 + 로그 응답
- [ ] Isaac Sim 단독 테스트: anomaly 감지 → HTTP 전송 → execute_action 수신 → 실행 → 응답 (agent 없이 curl로 테스트)

### Phase 3: Agent Server — ReAct 파이프라인

- [ ] agent_server/main.py: uvicorn :8001 세팅
- [ ] state.py: AgentState 정의
- [ ] prompts/diagnosis.py: Gemini 시스템 프롬프트 작성 (역할, 입력 포맷, 출력 포맷, tool 사용 지침)
- [ ] tools.py: extract_coordinates 구현 (OpenCV 기반 좌표 추출)
- [ ] tools.py: execute_action 구현 (Isaac Sim에 HTTP POST)
- [ ] nodes.py: call_model, call_tool, should_continue 구현
- [ ] graph.py: StateGraph 조립 (llm → should_continue → tools → llm 루프)
- [ ] agent_api.py: POST /anomaly → AgentState 초기화 → graph 실행 → 결과 반환
- [ ] Agent Server 단독 테스트: mock anomaly report로 graph 실행, tool 호출 확인

### Phase 4: End-to-End 통합

- [ ] Isaac Sim + Agent Server 동시 기동
- [ ] 시나리오 A end-to-end: 오프셋 → 감지 → agent 진단 → 좌표 추출 → re-grasp → 성공
- [ ] 시나리오 B end-to-end: 물체 부재 → 감지 → agent 진단 → 탐색 → 물체 발견 → grasp → 성공
- [ ] 실패 후 재진단 루프 동작 확인: success=false → LLM 재판단 → 재시도
- [ ] 복구 불가 케이스 처리: 최대 시도 횟수 초과 시 graceful termination

### Phase 5: 시나리오 확장 + 데모 준비

- [ ] 시나리오 C 설계 및 구현 (시간 여유 시)
- [ ] 미정의 시나리오 3~5개에 대한 Gemini 진단 정확도 테스트
- [ ] rule-based baseline vs MLLM-augmented 비교 데이터 수집
- [ ] 데모 시나리오 구성 및 시연 준비
