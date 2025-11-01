@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ========================================
echo  Unified Face Analyzer TCP Server
echo ========================================
echo.

REM ========================================
REM 1. Python 버전 체크
REM ========================================
echo 🔍 Python 버전 확인 중...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python이 설치되어 있지 않습니다!
    echo.
    echo 💡 Python 3.9 ~ 3.12를 설치해주세요.
    echo    다운로드: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python %PYTHON_VERSION% 감지됨
echo.

REM Python 3.13+ 경고 (3.9-3.12 지원)
echo %PYTHON_VERSION% | findstr /R "3\.1[3-9]\." >nul
if not errorlevel 1 (
    echo ⚠️  경고: Python %PYTHON_VERSION%가 감지되었습니다.
    echo    권장 버전: Python 3.9 ~ 3.12
    echo    일부 라이브러리가 호환되지 않을 수 있습니다.
    echo.
    choice /C YN /M "계속 진행하시겠습니까? (Y/N)"
    if errorlevel 2 exit /b 1
    echo.
)

REM ========================================
REM 2. 가상환경 체크 및 생성
REM ========================================
if not exist "venv\" (
    echo 📦 가상환경이 없습니다. 새로 생성합니다...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ 가상환경 생성 실패!
        echo.
        echo 💡 다음을 확인해주세요:
        echo    1. Python이 PATH에 등록되어 있는지
        echo    2. 관리자 권한으로 실행했는지
        echo    3. 디스크 공간이 충분한지
        echo.
        pause
        exit /b 1
    )
    echo ✅ 가상환경 생성 완료
    echo.
) else (
    echo ✅ 가상환경이 이미 존재합니다.
    echo.
)

REM ========================================
REM 3. 가상환경 활성화
REM ========================================
echo 🔧 가상환경 활성화 중...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ 가상환경 활성화 실패!
    echo.
    echo 💡 venv\Scripts\activate.bat 파일이 존재하는지 확인해주세요.
    echo.
    pause
    exit /b 1
)
echo ✅ 가상환경 활성화 완료
echo.

REM ========================================
REM 4. pip 업그레이드 (최신 버전 필요)
REM ========================================
echo 🔧 pip 업그레이드 중...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo ⚠️  pip 업그레이드 실패 (계속 진행)
) else (
    echo ✅ pip 업그레이드 완료
)
echo.

REM ========================================
REM 5. 의존성 패키지 설치 체크
REM ========================================
echo 🔍 설치된 패키지 확인 중...
pip show torch >nul 2>&1
if errorlevel 1 (
    set NEED_INSTALL=1
    echo ⚠️  필수 패키지가 설치되어 있지 않습니다.
) else (
    set NEED_INSTALL=0
    echo ✅ 필수 패키지가 이미 설치되어 있습니다.
)
echo.

if !NEED_INSTALL!==1 (
    echo 📥 의존성 패키지 설치 중... (5-10분 소요)
    echo    💡 처음 설치 시 시간이 오래 걸릴 수 있습니다.
    echo.

    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ❌ 패키지 설치 실패!
        echo.
        echo 💡 다음을 시도해보세요:
        echo    1. 인터넷 연결 확인
        echo    2. 방화벽 설정 확인
        echo    3. pip install -r requirements.txt 수동 실행
        echo    4. GPU 버전이 필요한 경우:
        echo       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        echo.
        pause
        exit /b 1
    )
    echo.
    echo ✅ 패키지 설치 완료
    echo.
) else (
    echo 💡 패키지 재설치가 필요한 경우:
    echo    pip install -r requirements.txt --force-reinstall
    echo.
)

REM ========================================
REM 6. 모델 파일 체크 (BiSeNet)
REM ========================================
echo 🔍 모델 파일 확인 중...
if not exist "data\79999_iter.pth" (
    echo ⚠️  BiSeNet 모델 파일이 없습니다!
    echo.
    echo 💡 다음 위치에서 다운로드해주세요:
    echo    https://github.com/zllrunning/face-parsing.PyTorch
    echo    파일: 79999_iter.pth (169MB)
    echo    위치: unified_face_analyzer\data\79999_iter.pth
    echo.
    echo    CLIP 모델은 첫 실행 시 자동 다운로드됩니다.
    echo.
    choice /C YN /M "모델 없이 계속 진행하시겠습니까? (오류 발생 가능) (Y/N)"
    if errorlevel 2 exit /b 1
    echo.
) else (
    echo ✅ BiSeNet 모델 파일 확인 완료
    echo.
)

REM ========================================
REM 7. TCP 서버 시작
REM ========================================
echo ========================================
echo  🚀 TCP 서버 시작
echo ========================================
echo.
echo 💡 서버 포트:
echo    - IR 카메라 수신: 5001
echo    - 분석 요청: 10000
echo    - 실시간 뷰어: 7001
echo.
echo 💡 종료: Ctrl+C
echo.

python tcp_server.py

REM ========================================
REM 서버 종료 후 처리
REM ========================================
echo.
echo ========================================
echo  서버가 종료되었습니다.
echo ========================================
echo.

REM 에러 레벨 체크
if errorlevel 1 (
    echo ❌ 서버가 오류로 인해 종료되었습니다.
    echo.
    echo 💡 로그를 확인하거나 다음을 시도해보세요:
    echo    1. config.yaml 설정 확인
    echo    2. 포트 충돌 확인 (다른 프로그램이 5001/10000 사용 중인지)
    echo    3. python tcp_server.py 직접 실행하여 에러 메시지 확인
    echo.
) else (
    echo ✅ 정상 종료
    echo.
)

pause
endlocal
