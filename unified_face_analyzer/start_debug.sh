#!/bin/bash
# Debug 모드로 TCP 서버 시작
# "start" 명령으로 샘플 이미지 분석

# Config에서 포트 번호 읽기
PORT=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['server']['port'])")

echo "=================================="
echo "  🐛 Starting Debug Mode Server"
echo "=================================="
echo "Mode: Debug (0)"
echo "Port: ${PORT}"
echo "Protocol: Send 'start' command"
echo "Sample Image: sample_images/camera_capture_20250513_180034.png"
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
./venv/bin/python3 tcp_server_simple.py --mode 0
