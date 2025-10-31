# -*- coding: utf-8 -*-
"""
IR Camera Receiver - Port 5001
Connects to IR camera server and receives Y8 image data
"""

import socket
import time
from typing import Callable, Optional
from .base_server import BaseTCPServer
from .image_buffer import ImageBuffer
from .socket_utils import setup_socket_buffers
from utils import get_logger
from utils.image_utils import decode_y8_to_bgr
from utils.color_log import ColorLog
import numpy as np

logger = get_logger(__name__)


class IRCameraReceiver(BaseTCPServer):
    """
    IR 카메라 이미지 수신 클라이언트 (Port 5001)

    Features:
    - controlled_dual_server.py의 Port 5001에 연결
    - Y8 이미지 데이터 수신 (1280x800 = 1,024,000 bytes)
    - Y8 → BGR 변환 (한 번만 수행)
    - ImageBuffer에 BGR 이미지 저장
    - Viewer로 BGR 이미지 브로드캐스트 콜백
    """

    def __init__(
        self,
        image_buffer: ImageBuffer,
        ir_host: str = '127.0.0.1',
        ir_port: int = 5001,
        on_frame_received: Optional[Callable[[np.ndarray], None]] = None
    ):
        """
        Args:
            image_buffer: 공유 ImageBuffer 인스턴스
            ir_host: IR 카메라 서버 호스트
            ir_port: IR 카메라 서버 포트
            on_frame_received: BGR 이미지 수신 시 호출될 콜백 함수
        """
        super().__init__("IRCameraReceiver")
        self.image_buffer = image_buffer
        self.ir_host = ir_host
        self.ir_port = ir_port
        self.on_frame_received = on_frame_received
        self.ir_camera_connected = False

        # Unity 스타일: Stateful 버퍼 (프레임 경계 처리용)
        self.frame_size = 1280 * 800  # 1,024,000 bytes
        self.frame_buffer = bytearray(self.frame_size)
        self.buffer_position = 0

        # Chunk 크기 통계 (주기적 모니터링용)
        self.chunk_stats = {1024000: 0, 1048576: 0, 'other': 0}
        self.frame_count = 0

    def _run(self):
        """IR 카메라 클라이언트 메인 루프"""
        try:
            ColorLog.header("IR Camera Receiver (Client Mode)")
            ColorLog.info(f"Connecting to {self.ir_host}:{self.ir_port}")

            last_retry_print = 0

            while self.is_running:
                try:
                    # IR 카메라 서버에 연결 (Client 모드)
                    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

                    # 소켓 버퍼 최적화
                    setup_socket_buffers(client_socket, buffer_size_mb=2)

                    client_socket.connect((self.ir_host, self.ir_port))

                    ColorLog.success(f"Connected to IR Camera Server: {self.ir_host}:{self.ir_port}")

                    # 연결된 서버로부터 데이터 수신
                    self._handle_ir_camera_client(client_socket, (self.ir_host, self.ir_port))

                except ConnectionRefusedError:
                    if self.is_running:
                        # 5초마다 한 번씩만 재시도 메시지 출력
                        current_time = time.time()
                        if current_time - last_retry_print >= 5.0:
                            ColorLog.warning(f"IR Camera Server not available, retrying...")
                            last_retry_print = current_time
                        time.sleep(5)
                except Exception as e:
                    if self.is_running:
                        ColorLog.error(f"Connection error: {e}, retrying in 5s...")
                        time.sleep(5)

            ColorLog.info("IR camera client stopped")

        except Exception as e:
            logger.error(f"IR camera client error: {e}")

    def _reset_buffer(self):
        """Unity 스타일: 버퍼 position만 0으로 리셋"""
        self.buffer_position = 0

    def _handle_received_chunk(self, chunk: bytes) -> bool:
        """
        Unity 스타일: 수신한 청크 처리 (프레임 경계 자동 처리)

        Args:
            chunk: 소켓에서 수신한 데이터 청크

        Returns:
            프레임이 완성되었으면 True, 아니면 False
        """
        chunk_len = len(chunk)
        remaining = self.frame_size - self.buffer_position
        is_len_over = chunk_len > remaining

        # 청크 크기 통계 집계
        if chunk_len == 1024000:
            self.chunk_stats[1024000] += 1
        elif chunk_len == 1048576:
            self.chunk_stats[1048576] += 1
        else:
            self.chunk_stats['other'] += 1

        # 현재 프레임에 필요한 만큼만 복사
        copy_size = remaining if is_len_over else chunk_len
        self.frame_buffer[self.buffer_position:self.buffer_position + copy_size] = chunk[:copy_size]

        # 프레임 완성 체크
        if self.buffer_position + copy_size >= self.frame_size:
            # 프레임 처리
            self._process_frame()

            # leftover 처리 (다음 프레임의 시작)
            if is_len_over:
                self._reset_buffer()
                leftover = chunk_len - remaining
                # 남은 부분을 버퍼 시작에 복사
                self.frame_buffer[0:leftover] = chunk[remaining:remaining + leftover]
                self.buffer_position = leftover
                logger.debug(f"Frame completed with leftover: {leftover} bytes")
            else:
                self._reset_buffer()

            return True
        else:
            # 아직 프레임 미완성
            self.buffer_position += copy_size  # ✅ FIX: chunk_len → copy_size
            return False

    def _process_frame(self):
        """
        Unity 스타일: 완성된 프레임 처리
        - Y8 → BGR 변환
        - ImageBuffer 업데이트
        - Viewer 브로드캐스트
        - 주기적 통계 출력
        """
        try:
            # Y8 → BGR 변환
            frame_data = bytes(self.frame_buffer[:self.frame_size])
            bgr_image = decode_y8_to_bgr(frame_data, width=1280, height=800)

            if bgr_image is None:
                logger.warning("Failed to convert Y8 to BGR")
                return

            # ImageBuffer 업데이트 (BGR 이미지 저장)
            self.image_buffer.update(bgr_image)

            # Viewer 브로드캐스트 콜백 호출 (BGR 이미지 전송)
            if self.on_frame_received:
                self.on_frame_received(bgr_image)

            # 프레임 카운트 증가 및 주기적 통계 출력 (100 프레임마다)
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                total_chunks = sum(self.chunk_stats.values())
                if total_chunks > 0:
                    ColorLog.info(
                        f"📊 Chunk stats (last 100 frames): "
                        f"1024KB={self.chunk_stats[1024000]}, "
                        f"1048KB={self.chunk_stats[1048576]}, "
                        f"other={self.chunk_stats['other']}"
                    )
                # 통계 리셋
                self.chunk_stats = {1024000: 0, 1048576: 0, 'other': 0}

        except Exception as e:
            ColorLog.error(f"Frame processing error: {e}")

    def _handle_ir_camera_client(self, client_socket: socket.socket, client_address):
        """
        Unity 스타일: IR 카메라 클라이언트 처리
        - 65KB 청크 단위로 수신
        - 프레임 경계 자동 처리 (leftover 관리)
        - Y8 → BGR 변환
        - ImageBuffer에 BGR 이미지 업데이트
        - Viewer로 BGR 이미지 브로드캐스트
        """
        ColorLog.event(f"IR camera stream started")
        self.ir_camera_connected = True

        # 버퍼 초기화 (새 연결마다 리셋)
        self._reset_buffer()

        CHUNK_SIZE = 65536  # 65KB (Unity와 동일)

        try:
            while self.is_running:
                # Unity 스타일: 청크 단위 수신 (프레임 크기 무관)
                chunk = client_socket.recv(CHUNK_SIZE)

                if not chunk:
                    # 연결 종료
                    ColorLog.warning("IR camera stream ended")
                    break

                # Unity 스타일: 프레임 경계 자동 처리
                self._handle_received_chunk(chunk)

        except Exception as e:
            ColorLog.error(f"Stream error: {e}")
        finally:
            # 연결 상태 업데이트
            self.ir_camera_connected = False
            client_socket.close()
            ColorLog.info("IR camera disconnected")
