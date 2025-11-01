@echo off
REM ========================================
REM nul 파일 삭제 스크립트
REM Windows에서 "nul" 파일 강제 삭제
REM ========================================

echo ========================================
echo  nul 파일 정리 도구
echo ========================================
echo.

REM 현재 디렉토리 nul 삭제
if exist "nul" (
    echo 🔍 루트 nul 파일 발견
    del /F /Q "\\?\%CD%\nul" 2>NUL
    if exist "nul" (
        echo ❌ 삭제 실패 - 관리자 권한으로 실행하세요
    ) else (
        echo ✅ 루트 nul 파일 삭제 완료
    )
) else (
    echo ✅ 루트 nul 파일 없음
)
echo.

REM unified_face_analyzer/nul 삭제
if exist "unified_face_analyzer\nul" (
    echo 🔍 unified_face_analyzer/nul 파일 발견
    del /F /Q "\\?\%CD%\unified_face_analyzer\nul" 2>NUL
    if exist "unified_face_analyzer\nul" (
        echo ❌ 삭제 실패 - 관리자 권한으로 실행하세요
    ) else (
        echo ✅ unified_face_analyzer/nul 파일 삭제 완료
    )
) else (
    echo ✅ unified_face_analyzer/nul 파일 없음
)
echo.

REM sample_ir_server/nul 삭제
if exist "sample_ir_server\nul" (
    echo 🔍 sample_ir_server/nul 파일 발견
    del /F /Q "\\?\%CD%\sample_ir_server\nul" 2>NUL
    if exist "sample_ir_server\nul" (
        echo ❌ 삭제 실패 - 관리자 권한으로 실행하세요
    ) else (
        echo ✅ sample_ir_server/nul 파일 삭제 완료
    )
) else (
    echo ✅ sample_ir_server/nul 파일 없음
)
echo.

echo ========================================
echo  정리 완료
echo ========================================
echo.
echo 💡 nul 파일 생성 방지:
echo    - Bash 명령어에서 ^> nul 대신 ^>NUL 사용
echo    - 또는 ^>/dev/null 사용 (크로스 플랫폼)
echo.

pause
