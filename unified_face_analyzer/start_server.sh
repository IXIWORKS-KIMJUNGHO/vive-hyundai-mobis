#!/bin/bash
# TCP Server Startup Script

# 설정 (여기서 수정하세요!)
HOST="0.0.0.0"        # 모든 인터페이스 (외부 연결 허용)
PORT=5001             # 포트 번호
MAX_CONNECTIONS=5     # 최대 연결 수

# 서버 시작
echo "🚀 Starting TCP Server..."
echo "   Host: $HOST"
echo "   Port: $PORT"
echo ""

./venv/bin/python3 tcp_server.py --host $HOST --port $PORT --max-connections $MAX_CONNECTIONS
