"""
TCP Model - TCP 서버 래퍼 및 네트워크 통신 관리
"""
from utils import get_tcp_server
from utils import get_logger
from typing import Dict, Any

logger = get_logger(__name__)


class TCPModel:
    """
    TCP 서버 모델
    TCPServer를 캡슐화하고 네트워크 통신을 관리합니다.
    """

    def __init__(self):
        """TCP 모델 초기화"""
        self.server = get_tcp_server()
        self.is_streaming = False

    def start_streaming(self) -> bool:
        """
        TCP 스트리밍 시작

        Returns:
            bool: 시작 성공 여부
        """
        if self.is_streaming:
            logger.warning("TCP streaming already started")
            return False

        try:
            self.server.start_server()
            self.is_streaming = True
            logger.info("TCP streaming started")
            return True
        except Exception as e:
            logger.error(f"Failed to start TCP streaming: {e}", exc_info=True)
            return False

    def stop_streaming(self) -> bool:
        """
        TCP 스트리밍 중지

        Returns:
            bool: 중지 성공 여부
        """
        if not self.is_streaming:
            logger.warning("TCP streaming not running")
            return False

        try:
            self.server.stop_server()
            self.is_streaming = False
            logger.info("TCP streaming stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop TCP streaming: {e}", exc_info=True)
            return False

    def update_result(self, results: Dict[str, Any], image_path: str = ""):
        """
        분석 결과 업데이트 (브로드캐스트)

        Args:
            results: 분석 결과
            image_path: 이미지 경로
        """
        if not self.is_streaming:
            logger.debug("TCP not streaming, result not sent")
            return

        try:
            self.server.update_result(results, image_path)
            classification = results.get('classification', 'Unknown')
            client_count = self.get_client_count()
            logger.debug(f"TCP broadcast: {classification} to {client_count} client(s)")
        except Exception as e:
            logger.error(f"Failed to update TCP result: {e}", exc_info=True)

    def get_client_count(self) -> int:
        """
        연결된 클라이언트 수 조회

        Returns:
            int: 클라이언트 수
        """
        try:
            return self.server.get_client_count()
        except Exception as e:
            logger.error(f"Failed to get client count: {e}")
            return 0

    def is_running(self) -> bool:
        """
        스트리밍 실행 상태 확인

        Returns:
            bool: 실행 중 여부
        """
        return self.is_streaming
