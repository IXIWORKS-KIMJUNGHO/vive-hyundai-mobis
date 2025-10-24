#!/bin/bash
# Production 모드로 TCP 서버 시작
# Raw 이미지 데이터 수신 → 분석 → JSON 결과 반환

# Config에서 포트 번호 읽기
PORT=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['server']['port'])")

echo "=================================="
echo "  🚀 Starting Production Mode Server"
echo "=================================="
echo "Mode: Production (1)"
echo "Port: ${PORT}"
echo "Protocol: Send raw image data (PNG/JPEG)"
echo "Waiting for Unreal Engine connection..."
echo ""

# 기존 포트 사용 중인 프로세스 종료
echo "🔍 Checking port ${PORT}..."
if lsof -ti:${PORT} > /dev/null 2>&1; then
    echo "⚠️  Port ${PORT} is already in use. Killing existing process..."
    lsof -ti:${PORT} | xargs kill -9
    sleep 1
    echo "✅ Port ${PORT} is now available"
else
    echo "✅ Port ${PORT} is available"
fi

echo ""
./venv/bin/python3 tcp_server_simple.py --mode 1
