# -*- coding: utf-8 -*-
"""
Color Log Utility for Clean Console Output
중요 이벤트만 컬러로 출력하는 로깅 유틸리티
"""

import sys
from datetime import datetime
from typing import Optional


class ColorLog:
    """
    컬러 로그 유틸리티

    Features:
    - 타임스탬프 자동 추가
    - 이모지 + 컬러로 가시성 향상
    - 한 줄 상태 업데이트 (실시간 덮어쓰기)
    - 중요 이벤트만 출력
    """

    # ANSI 색상 코드
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def timestamp() -> str:
        """현재 시간 반환 (HH:MM:SS 형식)"""
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def success(msg: str):
        """성공 메시지 (초록색 + ✅)"""
        print(f"{ColorLog.GREEN}✅ [{ColorLog.timestamp()}] {msg}{ColorLog.RESET}")

    @staticmethod
    def info(msg: str):
        """정보 메시지 (파란색 + ℹ️)"""
        print(f"{ColorLog.BLUE}ℹ️  [{ColorLog.timestamp()}] {msg}{ColorLog.RESET}")

    @staticmethod
    def warning(msg: str):
        """경고 메시지 (노란색 + ⚠️)"""
        print(f"{ColorLog.YELLOW}⚠️  [{ColorLog.timestamp()}] {msg}{ColorLog.RESET}")

    @staticmethod
    def error(msg: str):
        """에러 메시지 (빨간색 + ❌)"""
        print(f"{ColorLog.RED}❌ [{ColorLog.timestamp()}] {msg}{ColorLog.RESET}")

    @staticmethod
    def event(msg: str):
        """이벤트 메시지 (시안색 + 📡)"""
        print(f"{ColorLog.CYAN}📡 [{ColorLog.timestamp()}] {msg}{ColorLog.RESET}")

    @staticmethod
    def analysis(msg: str):
        """분석 메시지 (마젠타 + 🔍)"""
        print(f"{ColorLog.MAGENTA}🔍 [{ColorLog.timestamp()}] {msg}{ColorLog.RESET}")

    @staticmethod
    def separator():
        """구분선 출력"""
        print(f"{ColorLog.BLUE}{'─' * 80}{ColorLog.RESET}")

    @staticmethod
    def header(title: str):
        """헤더 출력 (굵게)"""
        print(f"\n{ColorLog.BOLD}{ColorLog.CYAN}═══ {title} ═══{ColorLog.RESET}\n")

    @staticmethod
    def status_line(
        ir_connected: bool,
        viewer_count: int,
        buffer_count: int,
        max_buffer: int = 5
    ):
        """
        한 줄 상태 업데이트 (실시간 덮어쓰기)

        Args:
            ir_connected: IR 카메라 연결 상태
            viewer_count: Viewer 클라이언트 수
            buffer_count: 현재 버퍼에 있는 이미지 수
            max_buffer: 최대 버퍼 크기
        """
        ir_icon = "✅" if ir_connected else "⏳"
        status = (
            f"\r{ColorLog.CYAN}[{ColorLog.timestamp()}] "
            f"IR:{ir_icon} | "
            f"Viewer:{viewer_count}👁️  | "
            f"Buffer:{buffer_count}/{max_buffer}📸{ColorLog.RESET}"
        )
        sys.stdout.write(status)
        sys.stdout.flush()

    @staticmethod
    def clear_line():
        """현재 줄 지우기"""
        sys.stdout.write('\r' + ' ' * 100 + '\r')
        sys.stdout.flush()

    @staticmethod
    def newline():
        """새 줄 추가"""
        print()


# 간편 사용을 위한 전역 함수
def log_success(msg: str):
    ColorLog.success(msg)

def log_info(msg: str):
    ColorLog.info(msg)

def log_warning(msg: str):
    ColorLog.warning(msg)

def log_error(msg: str):
    ColorLog.error(msg)

def log_event(msg: str):
    ColorLog.event(msg)

def log_analysis(msg: str):
    ColorLog.analysis(msg)
