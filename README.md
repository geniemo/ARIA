# ARIA — Autonomous Recovery with Intelligent Agent

> 성균관대학교 소프트웨어학과 졸업작품 | 박지원 | 2026.03

## 프로젝트 개요

ARIA는 **Isaac Sim 환경에서 협동로봇(Franka Panda)의 Pick-and-Place 작업 중 발생하는 이상 상황을 MLLM 에이전트가 자율적으로 진단하고 복구하는 시스템**이다.

로봇이 물체를 잡지 못하면(gripper width 기반 rule-based 감지), 카메라 이미지와 실행 로그를 Gemini에게 전달하고, Gemini가 LangGraph ReAct 패턴으로 원인을 진단하고 복구 전략을 수립하여 tool 호출로 로봇을 제어한다.

### 핵심 설계 원칙: 하이브리드 아키텍처

| 계층 | 역할 | 담당 |
|------|------|------|
| Rule-based | 이상 감지 (gripper width), 성공/실패 판정 | Isaac Sim |
| MLLM | "왜 실패했고 다음에 뭘 할지" 추론 | Gemini 3.1 Pro |
| 비전 알고리즘 | 정밀 좌표 추출 | OpenCV |

MLLM에게 정밀 좌표나 성공/실패 판정 같은 확정적인 일은 시키지 않고, 각 계층이 잘하는 것만 담당한다.

## 기술 스택

| 구분 | 기술 |
|------|------|
| 시뮬레이션 | NVIDIA Isaac Sim 5.1.0 |
| MLLM | Gemini 3.1 Pro Preview (`langchain-google-genai`) |
| Agent 프레임워크 | LangGraph (ReAct 패턴, StateGraph) |
| API 서버 | FastAPI + Pydantic |
| 비전 | OpenCV (HSV 큐브 검출 + 실측 캘리브레이션) |
| 로봇 | Franka Panda (SingleManipulator + ParallelGripper) |
| 로봇 제어 | Lula RRT (trajectory planning) + ArticulationTrajectory |
| 통신 | HTTP (Isaac Sim :8000 ↔ Agent Server :8001) |

## 시스템 아키텍처

```
Isaac Sim (:8000)                    Agent Server (:8001)
┌─────────────────┐                 ┌──────────────────────────┐
│                  │  anomaly 발생   │                          │
│  Rule-based     │ ──────────────> │  LangGraph ReAct Agent   │
│  이상 감지       │  로그+센서+     │                          │
│                  │  이미지 ×2      │  Gemini 3.1 Pro          │
│  로봇 정지       │                 │    ↓                     │
│                  │  execute_action │  Tool 호출 결정           │
│  HTTP 서버      │ <────────────── │  - extract_coordinates   │
│                  │  action+coords  │  - execute_action        │
│  action 실행    │ ──────────────> │                          │
│  결과 판정      │  success/fail   │  결과 관찰 → 재판단       │
└─────────────────┘                 └──────────────────────────┘
```

## 동작 흐름

### 정상 동작
1. Franka가 Lula RRT로 Pick-and-Place 수행 (approach → close → lift → move → open)
2. 각 phase 완료 시 센서 체크, close_gripper 후 gripper width로 grasp 성공 여부 판정

### 이상 발생 시 (시나리오 A: 물체 오프셋)
1. 큐브가 원래 위치에서 밀려남 → grasp 실패 → ANOMALY → 로봇 정지
2. Agent Server에 anomaly report 전송 (overhead/wrist 이미지 + 로봇 상태 + 실행 로그)
3. Gemini 진단: "이미지에서 큐브가 보이지만 위치가 다르다"
4. `extract_coordinates()` 호출 → OpenCV가 overhead 이미지에서 정밀 좌표 추출
5. `execute_action(grasp, coords, recover)` 호출 → Isaac Sim에서 복구 P&P 실행
6. 성공 → 종료

### 이상 발생 시 (시나리오 B: 물체 부재)
1. 큐브가 로봇 팔 뒤로 이동됨 → grasp 실패 → ANOMALY
2. Agent Server에 보고
3. Gemini 진단: "이미지에서 큐브가 안 보인다"
4. `execute_action(move, explore)` 호출 → 로봇 팔을 이동시켜 시야 확보
5. `extract_coordinates()` 호출 → 업데이트된 overhead 이미지에서 큐브 발견
6. `execute_action(grasp, coords, recover)` 호출 → 복구 성공

## 프로젝트 구조

```
ARIA/
├── contracts/                       # Isaac Sim ↔ Agent Server 공유 스키마
│   ├── schemas.py                   # Pydantic 모델 (RobotState, AnomalyReport, ExecutionLog 등)
│   ├── skill_primitives.py          # SkillName, ActionIntent, PhaseName enum
│   └── api_specs.py                 # 포트/엔드포인트 상수
│
├── isaac_sim/                       # Isaac Sim 시뮬레이션 측
│   ├── main.py                      # 엔트리포인트 (--scenario normal|a|b)
│   ├── scene/
│   │   ├── scene_builder.py         # Franka + 작업대 + 큐브 + 카메라 배치
│   │   └── error_injector.py        # 시나리오 A/B 에러 주입
│   ├── control/
│   │   ├── rrt_controller.py        # Lula RRT trajectory planning + 실행
│   │   └── gripper_controller.py    # gripper open/close + settle 판정
│   ├── server/
│   │   └── sim_api.py               # HTTP 서버 (:8000, /execute_action)
│   └── task/
│       └── pick_place_task.py       # 5-phase P&P 관리 + 이상 감지 + 실행 로그
│
├── agent_server/                    # AI Agent 측
│   ├── main.py                      # uvicorn :8001 + POST /anomaly
│   ├── agents/
│   │   ├── graph.py                 # LangGraph StateGraph (ReAct 루프)
│   │   ├── state.py                 # AgentState 정의
│   │   ├── nodes.py                 # call_model, call_tool, should_continue
│   │   ├── tools.py                 # extract_coordinates, execute_action
│   │   └── recovery_logger.py       # ReAct 루프 구조화 로그 (JSON)
│   ├── vision/
│   │   └── object_detector.py       # OpenCV HSV 큐브 검출 + 픽셀→월드 좌표 변환
│   ├── prompts/
│   │   └── diagnosis.py             # Gemini 시스템 프롬프트
│   └── server/
│       └── agent_api.py             # run_recovery (multimodal 메시지 + graph 실행)
│
├── ARIA_ARCHITECTURE.md             # 상세 아키텍처 문서 (v3.0)
├── .env                             # GOOGLE_API_KEY (gitignore 대상)
└── .env.example                     # 환경변수 템플릿
```

## 실행 방법

### 사전 요구사항
- NVIDIA Isaac Sim 5.1.0 (`~/isaacsim/`에 설치)
- Conda 환경 `ARIA` (Python 3.11)
- Google API Key (Gemini 3.1 Pro 접근)

### 환경 설정
```bash
conda activate ARIA
pip install pydantic fastapi uvicorn langchain-core langchain-google-genai langgraph httpx opencv-python-headless Pillow python-dotenv

# .env 파일에 API 키 설정
cp .env.example .env
# GOOGLE_API_KEY=your_key_here 입력
```

### 실행

**터미널 1 — Agent Server:**
```bash
cd ~/workspace/ARIA
python -m agent_server.main
# → http://localhost:8001 에서 대기
```

**터미널 2 — Isaac Sim:**
```bash
cd ~/workspace/ARIA/isaac_sim
PYTHONPATH=~/workspace/ARIA:$PYTHONPATH ~/isaacsim/python.sh main.py --scenario a
# --scenario: normal (정상 P&P), a (물체 오프셋), b (물체 부재)
```

### curl로 수동 테스트 (Agent Server 없이)
```bash
# Isaac Sim만 실행 후, ANOMALY 발생하면:
curl -X POST http://localhost:8000/execute_action \
  -H "Content-Type: application/json" \
  -d '{"action": "grasp", "coords": {"x": 0.54, "y": -0.17, "z": 0.44}, "intent": "recover"}'
```

## 검증 결과

### Pick-and-Place 정확도
- 9개 케이스 (위치/방향/복합 변경) 전부 성공
- Place 오차 < 1.5mm

### 시나리오 A (물체 오프셋) E2E
- 3/3 케이스 성공
- Gemini가 overhead 이미지에서 큐브 위치를 정확히 진단
- OpenCV 좌표 추출 후 복구 P&P 성공

### 시나리오 B (물체 부재) E2E
- 3/3 연속 성공
- Gemini가 explore(로봇 이동) → 큐브 발견 → 좌표 추출 → 복구
- 로봇 팔에 가려진 큐브를 팔을 옮겨서 찾는 자율 탐색 동작

### 좌표 매핑 정확도
- OpenCV 픽셀→월드 좌표 변환 오차 < 1.5mm (실측 캘리브레이션)

## 로봇 제어: RMPFlow → Lula RRT 전환 근거

초기에는 RMPFlow(reactive controller)를 사용했으나, 경로를 미리 계획하지 않아 특정 위치에서 관절이 불안정하게 흔들리거나 작업대와 충돌하는 문제가 발생했다. Lula RRT(trajectory planner)로 전환하여 부드럽고 안정적인 동작을 달성했다.

| | RMPFlow | Lula RRT |
|--|---------|----------|
| 경로 계획 | 없음 (매 스텝 reactive) | 사전 계획 |
| 동작 품질 | 불안정 (진동, 충돌) | 부드러움 |
| 매 스텝 체크 | 가능 | 가능 (action sequence) |
| 실패 가능성 | 없음 (항상 움직임) | 경로 못 찾을 수 있음 |

## 개발 진행 상황

- [x] Phase 0: 환경 세팅 + contracts
- [x] Phase 1: Isaac Sim P&P (Lula RRT)
- [x] Phase 2: 에러 주입 + HTTP 서버
- [x] Phase 3: Agent Server (LangGraph ReAct)
- [x] Phase 4: End-to-End 통합
- [ ] Phase 5: 시나리오 확장 + 웹 대시보드 + 데모 준비

## 참고 문서

- `ARIA_ARCHITECTURE.md` — 상세 아키텍처 설계 문서 (v3.0)
