#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Face Analyzer TCP Server - Main Orchestrator
클라이언트로부터 이미지를 수신하여 실시간 얼굴 분석 수행

Architecture:
- IRCameraReceiver (Port 5001): IR 카메라 데이터 수신 → ImageBuffer 업데이트
- ViewerBroadcaster (Port 7001): Viewer 클라이언트에게 실시간 브로드캐스트
- AnalysisServer (Port 10000): 'analyze' 명령 처리 → 얼굴 분석 결과 반환
"""

import sys
import io

# Windows 콘솔 UTF-8 설정 (이모지 지원)
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except:
        pass

import time
import signal
from core.unified_analyzer import UnifiedFaceAnalyzer
from server import ImageBuffer, IRCameraReceiver, ViewerBroadcaster, AnalysisServer
from utils import get_logger
from utils.config_loader import get_config

logger = get_logger(__name__)
config = get_config()


class UnifiedFaceAnalysisTCPServer:
    """
    통합 얼굴 분석 TCP 서버 - Orchestrator

    Features:
    - 3개의 독립적인 서버 조율
    - 공유 리소스 관리 (ImageBuffer, UnifiedFaceAnalyzer)
    - 전체 라이프사이클 관리
    """

    def __init__(
        self,
        ir_host: str = '127.0.0.1',
        ir_port: int = 5001,
        viewer_port: int = 7001,
        analysis_port: int = 10000,
        buffer_size: int = 5
    ):
        """
        Args:
            ir_host: IR 카메라 서버 호스트
            ir_port: IR 카메라 서버 포트
            viewer_port: Viewer 브로드캐스트 포트
            analysis_port: 얼굴 분석 요청 포트
            buffer_size: 이미지 버퍼 크기
        """
        # 공유 리소스 생성
        self.image_buffer = ImageBuffer(max_size=buffer_size)
        self.analyzer = UnifiedFaceAnalyzer()

        # ViewerBroadcaster 생성 (먼저 생성해서 콜백에서 사용)
        self.viewer_broadcaster = ViewerBroadcaster(
            host='0.0.0.0',
            port=viewer_port
        )

        # 서버 인스턴스 생성
        self.ir_camera_receiver = IRCameraReceiver(
            image_buffer=self.image_buffer,
            ir_host=ir_host,
            ir_port=ir_port,
            on_frame_received=self.viewer_broadcaster.broadcast  # 콜백 연결
        )

        self.analysis_server = AnalysisServer(
            image_buffer=self.image_buffer,
            analyzer=self.analyzer,
            host='0.0.0.0',
            port=analysis_port
        )

        logger.info("UnifiedFaceAnalysisTCPServer initialized")

    def start(self):
        """모든 서버 시작"""
        print("\n" + "=" * 80)
        print("  Unified Face Analysis TCP Server")
        print("=" * 80)
        print()

        # 서버 시작 (순서 중요: Viewer → IR Camera → Analysis)
        self.viewer_broadcaster.start()
        time.sleep(0.5)  # 서버 시작 대기

        self.ir_camera_receiver.start()
        time.sleep(0.5)

        self.analysis_server.start()
        time.sleep(0.5)

        print("\n" + "=" * 80)
        print("  All servers started successfully!")
        print("=" * 80)
        print()

        logger.info("All servers started")

    def stop(self):
        """모든 서버 종료"""
        print("\n" + "=" * 80)
        print("  Shutting down all servers...")
        print("=" * 80)

        # 서버 종료 (역순)
        self.analysis_server.stop()
        self.ir_camera_receiver.stop()
        self.viewer_broadcaster.stop()

        print("✅ All servers stopped")
        logger.info("All servers stopped")

    def run(self):
        """서버 실행 (blocking)"""
        self.start()

        try:
            # 메인 스레드 대기
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n⚠️  Keyboard interrupt received")
        finally:
            self.stop()

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


def main():
    """메인 엔트리 포인트"""
    # Config에서 포트 설정 로드
    server_config = config.get('server', {})

    ir_port = server_config.get('ir_camera_port', 5001)
    viewer_port = server_config.get('viewer_port', 7001)
    analysis_port = server_config.get('analyze_port', 10000)
    buffer_size = server_config.get('image_buffer_size', 5)

    # 서버 생성 및 실행
    server = UnifiedFaceAnalysisTCPServer(
        ir_host='127.0.0.1',
        ir_port=ir_port,
        viewer_port=viewer_port,
        analysis_port=analysis_port,
        buffer_size=buffer_size
    )

    # SIGINT 핸들러 등록
    def signal_handler(sig, frame):
        print("\n\n⚠️  Signal received, shutting down...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # 서버 실행
    server.run()


if __name__ == "__main__":
    main()
