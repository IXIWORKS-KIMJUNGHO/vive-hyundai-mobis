@echo off
REM Production 모드로 TCP 서버 시작
REM Raw 이미지 데이터 수신 → 분석 → JSON 결과 반환

REM Config에서 포트 번호 읽기
for /f %%i in ('python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['server']['port'])"') do set PORT=%%i

echo ==================================
echo   🚀 Starting Production Mode Server
echo ==================================
echo Mode: Production (1)
echo Port: %PORT%
echo Protocol: Send raw image data (PNG/JPEG)
echo Waiting for Unreal Engine connection...
echo.

REM 기존 포트 사용 중인 프로세스 종료
echo 🔍 Checking port %PORT%...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
    echo ⚠️  Port %PORT% is already in use. Killing existing process...
    taskkill /F /PID %%a > nul 2>&1
    timeout /t 1 /nobreak > nul
    echo ✅ Port %PORT% is now available
    goto :continue
)
echo ✅ Port %PORT% is available

:continue
echo.
venv\Scripts\python.exe tcp_server_simple.py --mode 1
