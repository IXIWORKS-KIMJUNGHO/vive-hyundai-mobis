# -*- coding: utf-8 -*-
"""
Viewer Broadcaster - Port 7001
Broadcasts Y8 image data to connected viewer clients
"""

import socket
import threading
import numpy as np
from typing import List, Tuple, Union
from .base_server import BaseTCPServer
from utils import get_logger

logger = get_logger(__name__)


class ViewerBroadcaster(BaseTCPServer):
    """
    Viewer 브로드캐스트 서버 (Port 7001)

    Features:
    - Viewer 클라이언트 연결 관리
    - Y8 이미지 데이터 브로드캐스트
    - 실시간 스트리밍
    """

    def __init__(self, host: str = '0.0.0.0', port: int = 7001):
        """
        Args:
            host: 바인딩할 호스트
            port: 바인딩할 포트
        """
        super().__init__("ViewerBroadcaster")
        self.host = host
        self.port = port
        self.viewers: List[Tuple[socket.socket, tuple]] = []
        self.viewers_lock = threading.Lock()
        self.server_socket = None

    def _run(self):
        """Viewer 서버 메인 루프"""
        try:
            # 서버 소켓 생성
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)

            print("=" * 80)
            print(f"  Viewer Broadcast Server")
            print("=" * 80)
            print(f"Listening on: {self.host}:{self.port}")
            print("=" * 80)
            print()

            logger.info(f"Viewer server listening on {self.host}:{self.port}")

            # 클라이언트 연결 수락
            while self.is_running:
                try:
                    self.server_socket.settimeout(1.0)
                    client_socket, client_address = self.server_socket.accept()

                    with self.viewers_lock:
                        self.viewers.append((client_socket, client_address))

                    logger.info(f"Viewer connected: {client_address}")
                    print(f"📺 Viewer connected: {client_address[0]}:{client_address[1]} (Total: {len(self.viewers)})")

                except socket.timeout:
                    continue
                except Exception as e:
                    if self.is_running:
                        logger.error(f"Error accepting viewer: {e}")

        except Exception as e:
            logger.error(f"Viewer server error: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()
            logger.info("Viewer server stopped")

    def broadcast(self, data: Union[bytes, np.ndarray]):
        """
        모든 연결된 viewer에게 데이터 브로드캐스트

        Args:
            data: 브로드캐스트할 BGR 이미지 (numpy array) 또는 bytes
        """
        # numpy array를 bytes로 변환
        if isinstance(data, np.ndarray):
            data = data.tobytes()

        with self.viewers_lock:
            disconnected_viewers = []

            for viewer_socket, viewer_address in self.viewers:
                try:
                    viewer_socket.sendall(data)
                except Exception as e:
                    logger.warning(f"Failed to send to viewer {viewer_address}: {e}")
                    disconnected_viewers.append((viewer_socket, viewer_address))

            # 연결 끊긴 viewer 제거
            for viewer_socket, viewer_address in disconnected_viewers:
                try:
                    viewer_socket.close()
                except:
                    pass
                self.viewers.remove((viewer_socket, viewer_address))
                logger.info(f"Viewer disconnected: {viewer_address}")
                print(f"🔌 Viewer disconnected: {viewer_address[0]}:{viewer_address[1]} (Total: {len(self.viewers)})")

    def stop(self):
        """서버 종료 및 모든 viewer 연결 해제"""
        super().stop()

        with self.viewers_lock:
            for viewer_socket, viewer_address in self.viewers:
                try:
                    viewer_socket.close()
                    logger.info(f"Closed viewer: {viewer_address}")
                except:
                    pass
            self.viewers.clear()
