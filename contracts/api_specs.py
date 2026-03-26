"""포트 및 엔드포인트 상수."""

# Isaac Sim HTTP 서버
SIM_HOST = "localhost"
SIM_PORT = 8000
SIM_BASE_URL = f"http://{SIM_HOST}:{SIM_PORT}"

# Agent 서버
AGENT_HOST = "localhost"
AGENT_PORT = 8001
AGENT_BASE_URL = f"http://{AGENT_HOST}:{AGENT_PORT}"

# 엔드포인트
ENDPOINT_ANOMALY = "/anomaly"
ENDPOINT_EXECUTE_ACTION = "/execute_action"
