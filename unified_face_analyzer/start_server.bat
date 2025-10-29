@echo off
chcp 65001 >nul
echo ========================================
echo  Unified Face Analyzer TCP Server
echo ========================================
echo.

REM venv 존재 여부 확인
if not exist "venv\" (
    echo 📦 가상환경이 없습니다. 새로 생성합니다...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ 가상환경 생성 실패!
        pause
        exit /b 1
    )
    echo ✅ 가상환경 생성 완료
    echo.

    echo 🔧 가상환경 활성화 중...
    call venv\Scripts\activate.bat
    if errorlevel 1 (
        echo ❌ 가상환경 활성화 실패!
        pause
        exit /b 1
    )
    echo ✅ 가상환경 활성화 완료
    echo.

    echo 📥 의존성 패키지 설치 중...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ 패키지 설치 실패!
        pause
        exit /b 1
    )
    echo ✅ 패키지 설치 완료
    echo.
) else (
    echo ✅ 가상환경이 이미 존재합니다.
    echo 🔧 가상환경 활성화 중...
    call venv\Scripts\activate.bat
    if errorlevel 1 (
        echo ❌ 가상환경 활성화 실패!
        pause
        exit /b 1
    )
    echo ✅ 가상환경 활성화 완료
    echo.
)

echo ========================================
echo  🚀 TCP 서버 시작
echo ========================================
echo.

python tcp_server.py

REM 서버 종료 후
echo.
echo ========================================
echo  서버가 종료되었습니다.
echo ========================================
pause
