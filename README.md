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

## 실험 결과

### 1. Pick-and-Place 모션 정확도

Lula RRT trajectory planner 기반 Pick-and-Place의 위치/방향 변화에 대한 robustness를 평가하였다. 작업대 위 safe workspace 내에서 큐브의 위치와 Z축 회전을 변경하며 9개 케이스를 실행하였다.

| 카테고리 | 케이스 | Grasp Width (m) | Place XY 오차 (mm) | 결과 |
|---------|--------|----------------|-------------------|------|
| 위치만 변경 | 기본 위치 [0.5, -0.2] | 0.0397 | 0.9 | SUCCESS |
| | 오른쪽 앞 [0.55, -0.15] | 0.0398 | 1.1 | SUCCESS |
| | 중앙 [0.5, -0.1] | 0.0398 | 0.4 | SUCCESS |
| 방향만 변경 | Z축 20도 회전 | 0.0398 | 0.6 | SUCCESS |
| | Z축 45도 회전 | 0.0399 | 1.5 | SUCCESS |
| | Z축 -15도 회전 | 0.0398 | 1.3 | SUCCESS |
| 복합 변경 | 위치+Z 15도 | 0.0398 | 1.0 | SUCCESS |
| | 위치+Z -20도 | 0.0398 | 0.5 | SUCCESS |
| | 위치+Z 30도 | 0.0399 | 0.5 | SUCCESS |

**9/9 전 케이스 성공, 평균 place 오차 0.87mm.** 큐브 크기(40mm)에 대해 gripper width가 39.8mm로 일관되게 측정되어 안정적인 grasp를 확인하였다.

### 2. 좌표 추출 정확도 (OpenCV 픽셀→월드 변환)

Overhead 카메라(640×480, z=2.5m)에서 촬영한 이미지로부터 빨간 큐브의 월드 좌표를 추출하는 정확도를 평가하였다. 실측 데이터 기반 선형 캘리브레이션을 적용하였다.

| 실제 좌표 (m) | 검출 좌표 (m) | 오차 (mm) |
|-------------|-------------|----------|
| [0.50, -0.20] | [0.50, -0.20] | 1.5 |
| [0.50, 0.20] | [0.50, 0.20] | 1.4 |
| [0.60, -0.15] | [0.60, -0.15] | 0.7 |
| [0.55, -0.10] | [0.55, -0.10] | 0.7 |

**평균 좌표 추출 오차 1.1mm.** 그림자에 의한 색상 변화(HSV S,V 하한 50으로 대응)에서도 검출에 성공하였다.

### 3. 시나리오 A — 물체 오프셋 복구 (End-to-End)

큐브를 원래 위치에서 2~5cm 오프셋시킨 후, 전체 파이프라인(ANOMALY 감지 → Agent 진단 → 좌표 추출 → 복구 P&P)을 실행하였다.

| 케이스 | 오프셋 (cm) | Agent 진단 | 추출 좌표 (m) | 복구 결과 | Place 오차 (cm) |
|--------|-----------|-----------|-------------|---------|----------------|
| A1 | [4, 3] | 큐브 위치 이상 | [0.551, -0.173] | SUCCESS | 1.1 |
| A2 | [-4, 0] | 큐브 위치 이상 | [0.457, -0.168] | SUCCESS | 0.3 |
| A3 | [4, -3] | 큐브 위치 이상 | [0.567, -0.231] | SUCCESS | 2.6 |

**3/3 전 케이스 복구 성공.** Gemini가 overhead 이미지에서 큐브 위치 이상을 정확히 진단하고, OpenCV 좌표 추출 후 1회 만에 복구를 완료하였다.

### 4. 시나리오 B — 물체 부재 복구 (End-to-End)

큐브를 로봇 팔 뒤쪽(카메라 시야 밖)으로 이동시킨 후, Agent가 자율적으로 탐색하여 큐브를 찾고 복구하는 과정을 검증하였다.

| 실행 | Agent 행동 흐름 | 복구 결과 | Place 오차 (cm) |
|------|---------------|---------|----------------|
| 1회 | extract_coords(실패) → explore → extract_coords(성공) → recover | SUCCESS | 0.4 |
| 2회 | extract_coords(실패) → explore → extract_coords(성공) → recover | SUCCESS | 0.4 |
| 3회 | extract_coords(실패) → explore → extract_coords(성공) → recover | SUCCESS | 0.1 |

**3/3 연속 복구 성공.** Agent가 초기 이미지에서 큐브를 발견하지 못하면 자율적으로 로봇 팔을 이동시켜 overhead 카메라의 시야를 확보하고, 업데이트된 이미지에서 큐브를 검출하여 복구하였다. 이는 사전 정의된 규칙 없이 MLLM이 상황을 판단하고 탐색 전략을 수립한 결과이다.

### 5. 이상 감지 정확도

| 조건 | Gripper Width (m) | 판정 | 정확도 |
|------|------------------|------|--------|
| 큐브 정상 grasp | 0.0398 (큐브 크기 근접) | 정상 | O |
| 오프셋으로 빈 공간 grasp | 0.0000 (완전 닫힘) | ANOMALY | O |
| 큐브 부재 시 grasp | 0.0000 (완전 닫힘) | ANOMALY | O |

Gripper width 임계값(0.005m) 기반 rule-based 감지가 모든 테스트 케이스에서 정확하게 동작하였다.

## 로봇 제어: RMPFlow → Lula RRT 전환 근거

초기에는 RMPFlow(reactive controller)를 사용했으나, 경로를 미리 계획하지 않아 특정 위치에서 관절이 불안정하게 흔들리거나 작업대와 충돌하는 문제가 발생했다. Lula RRT(trajectory planner)로 전환하여 부드럽고 안정적인 동작을 달성했다.

| | RMPFlow | Lula RRT |
|--|---------|----------|
| 경로 계획 | 없음 (매 스텝 reactive) | 사전 계획 |
| 동작 품질 | 불안정 (진동, 충돌) | 부드러움 |
| 매 스텝 체크 | 가능 | 가능 (action sequence) |
| P&P 성공률 (9 케이스) | 3/9 (위치만 변경 시) | 9/9 |

## 의의

### 1. 사전 정의 없는 이상 대응

기존 산업용 로봇의 이상 대응은 예상 가능한 에러 유형을 사전에 정의하고, 각 유형에 대한 복구 루틴을 하드코딩하는 방식이다. 이 접근은 정의되지 않은 이상 상황에 대응할 수 없다는 근본적 한계가 있다.

ARIA의 MLLM 에이전트는 이미지와 실행 로그를 종합적으로 분석하여 **사전 정의되지 않은 상황에서도 원인을 추론하고 복구 전략을 자율적으로 수립**한다. 시나리오 B에서 Agent가 "카메라에 물체가 보이지 않으니 로봇 팔을 옮겨 시야를 확보하자"라는 탐색 전략을 스스로 결정한 것이 이를 보여준다.

### 2. 하이브리드 아키텍처의 실용성

MLLM 단독으로 로봇을 제어하면 정밀도, 응답 지연, 출력 불안정성 문제가 발생한다. ARIA는 각 계층이 강점을 가진 영역만 담당하는 하이브리드 구조로 이 문제를 해결하였다:

- **MLLM은 "왜"와 "다음에 뭘"만 판단** — 정밀 좌표(OpenCV)나 성공/실패 판정(rule-based)을 시키지 않음
- **확정적 판단은 rule-based로** — gripper width 수치 비교로 즉시 판정, MLLM 지연 없음
- **정밀 실행은 비전+제어로** — OpenCV 좌표 추출(오차 1.1mm) + RRT 경로 계획

이 분리를 통해 MLLM의 강점(고수준 추론, 멀티모달 이해)을 활용하면서 약점(정밀도, 속도)을 보완하는 실용적 아키텍처를 구현하였다.

### 3. 모듈 교체 가능한 설계

`ChatGoogleGenerativeAI`의 model 파라미터만 변경하면 MLLM을 교체할 수 있고, OpenCV 좌표 추출을 다른 비전 모델로 교체해도 tool 인터페이스만 유지하면 나머지 시스템에 영향이 없다. 이는 향후 도메인 특화 모델이나 더 정밀한 비전 알고리즘으로의 확장을 용이하게 한다.

## 참고 문서

- `ARIA_ARCHITECTURE.md` — 상세 아키텍처 설계 문서 (v3.0)
