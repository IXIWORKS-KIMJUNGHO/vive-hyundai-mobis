"""TCP 서버 모듈 - 얼굴 분석 결과 전송"""

import socket
import json
import logging
from typing import Optional, Dict, Any
from dataclasses import asdict
from io import BytesIO
from PIL import Image
import numpy as np

from src.models import DetailedFaceAnalysis


class FaceAnalysisTCPServer:
    """
    얼굴 분석 결과를 TCP로 전송하는 서버

    Features:
    - 비동기 연결 처리
    - JSON 형식 데이터 전송
    - 자동 재연결 지원
    """

    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 5000,
        max_connections: int = 5,
        buffer_size: int = 4096
    ):
        """
        TCP 서버 초기화

        Args:
            host: 서버 호스트 주소 (기본: 0.0.0.0, 모든 인터페이스)
            port: 서버 포트 번호 (기본: 5000)
            max_connections: 최대 동시 연결 수 (기본: 5)
            buffer_size: 수신 버퍼 크기 (기본: 4096 bytes)
        """
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.buffer_size = buffer_size

        self.server_socket: Optional[socket.socket] = None
        self.is_running = False

        # 로깅 설정
        self.logger = logging.getLogger(__name__)

    def start(self):
        """서버 시작"""
        try:
            # 소켓 생성
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # 소켓 옵션 설정 (주소 재사용)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # 바인딩
            self.server_socket.bind((self.host, self.port))

            # 리스닝 시작
            self.server_socket.listen(self.max_connections)

            self.is_running = True
            self.logger.info(f"TCP 서버 시작: {self.host}:{self.port}")
            print(f"✅ TCP 서버 시작: {self.host}:{self.port}")
            print(f"📡 연결 대기 중... (최대 {self.max_connections}개 연결)")

        except Exception as e:
            self.logger.error(f"서버 시작 실패: {e}")
            raise

    def accept_connection(self) -> tuple[socket.socket, tuple]:
        """
        클라이언트 연결 수락

        Returns:
            (client_socket, client_address) 튜플
        """
        if not self.is_running or self.server_socket is None:
            raise RuntimeError("서버가 시작되지 않았습니다")

        client_socket, client_address = self.server_socket.accept()
        self.logger.info(f"클라이언트 연결: {client_address}")
        print(f"🔗 클라이언트 연결: {client_address[0]}:{client_address[1]}")

        return client_socket, client_address

    def send_analysis_result(
        self,
        client_socket: socket.socket,
        analysis_result: DetailedFaceAnalysis
    ) -> bool:
        """
        얼굴 분석 결과를 클라이언트에 전송 (단순화된 형식)

        Args:
            client_socket: 클라이언트 소켓
            analysis_result: 얼굴 분석 결과

        Returns:
            전송 성공 여부
        """
        try:
            # 단순화된 분석 결과 (eye_shape, face_shape만 전송)
            data = {
                "eye_shape": analysis_result.eye_analysis.overall_eye_shape.value,
                "face_shape": analysis_result.face_shape_analysis.face_shape.value
            }

            # JSON 직렬화
            json_data = json.dumps(data, ensure_ascii=False, indent=2)

            # 데이터 크기 전송 (4 bytes, big-endian)
            data_size = len(json_data.encode('utf-8'))
            client_socket.sendall(data_size.to_bytes(4, byteorder='big'))

            # JSON 데이터 전송
            client_socket.sendall(json_data.encode('utf-8'))

            self.logger.info(f"데이터 전송 완료: {data_size} bytes")
            print(f"📤 데이터 전송: {data_size} bytes")

            return True

        except Exception as e:
            self.logger.error(f"데이터 전송 실패: {e}")
            print(f"❌ 전송 실패: {e}")
            return False

    def receive_command(self, client_socket: socket.socket) -> Optional[Dict[str, Any]]:
        """
        클라이언트로부터 명령 수신

        Args:
            client_socket: 클라이언트 소켓

        Returns:
            수신한 명령 딕셔너리 (실패 시 None)
        """
        try:
            # 데이터 크기 수신 (4 bytes)
            size_data = client_socket.recv(4)
            if not size_data:
                return None

            data_size = int.from_bytes(size_data, byteorder='big')

            # JSON 데이터 수신
            json_data = b''
            while len(json_data) < data_size:
                chunk = client_socket.recv(min(self.buffer_size, data_size - len(json_data)))
                if not chunk:
                    break
                json_data += chunk

            # JSON 파싱
            command = json.loads(json_data.decode('utf-8'))
            self.logger.info(f"명령 수신: {command}")

            return command

        except Exception as e:
            self.logger.error(f"명령 수신 실패: {e}")
            return None

    def receive_raw_image(self, client_socket: socket.socket) -> Optional[np.ndarray]:
        """
        클라이언트로부터 raw 이미지 데이터를 수신하여 numpy 배열로 변환

        Protocol:
            [4 bytes: image data size] + [raw image bytes]

        Args:
            client_socket: 클라이언트 소켓

        Returns:
            numpy 배열 형태의 이미지 (BGR format) 또는 None
        """
        try:
            # 이미지 데이터 크기 수신 (4 bytes, big-endian)
            size_data = client_socket.recv(4)
            if not size_data:
                self.logger.warning("이미지 크기 데이터 수신 실패")
                return None

            image_size = int.from_bytes(size_data, byteorder='big')
            self.logger.info(f"수신할 이미지 크기: {image_size} bytes")
            print(f"📥 이미지 데이터 수신 중... ({image_size} bytes)")

            # raw 이미지 데이터 수신
            image_data = b''
            while len(image_data) < image_size:
                remaining = image_size - len(image_data)
                chunk_size = min(self.buffer_size, remaining)
                chunk = client_socket.recv(chunk_size)

                if not chunk:
                    self.logger.error("이미지 데이터 수신 중단")
                    return None

                image_data += chunk

            self.logger.info(f"이미지 데이터 수신 완료: {len(image_data)} bytes")
            print(f"✅ 이미지 데이터 수신 완료")

            # bytes를 PIL Image로 변환
            image_bytes = BytesIO(image_data)
            pil_image = Image.open(image_bytes)

            # PIL Image를 numpy 배열로 변환 (RGB -> BGR for OpenCV)
            rgb_array = np.array(pil_image)

            # RGB to BGR 변환 (OpenCV는 BGR 포맷 사용)
            if len(rgb_array.shape) == 3 and rgb_array.shape[2] == 3:
                bgr_array = rgb_array[:, :, ::-1]
            else:
                bgr_array = rgb_array

            self.logger.info(f"이미지 변환 완료: shape={bgr_array.shape}, dtype={bgr_array.dtype}")
            print(f"🖼️  이미지 변환 완료: {bgr_array.shape}")

            return bgr_array

        except Exception as e:
            self.logger.error(f"raw 이미지 수신 실패: {e}")
            print(f"❌ 이미지 수신 실패: {e}")
            return None

    def close_client(self, client_socket: socket.socket, client_address: tuple):
        """
        클라이언트 연결 종료

        Args:
            client_socket: 클라이언트 소켓
            client_address: 클라이언트 주소
        """
        try:
            client_socket.close()
            self.logger.info(f"클라이언트 연결 종료: {client_address}")
            print(f"🔌 연결 종료: {client_address[0]}:{client_address[1]}")
        except Exception as e:
            self.logger.error(f"연결 종료 실패: {e}")

    def stop(self):
        """서버 종료"""
        self.is_running = False

        if self.server_socket:
            try:
                self.server_socket.close()
                self.logger.info("TCP 서버 종료")
                print("🛑 TCP 서버 종료")
            except Exception as e:
                self.logger.error(f"서버 종료 실패: {e}")

    def __enter__(self):
        """Context manager 진입"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.stop()
