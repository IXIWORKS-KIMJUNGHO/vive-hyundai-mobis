#!/usr/bin/env python3
"""
Unified Face Analyzer TCP Server
클라이언트로부터 이미지를 수신하여 실시간 얼굴 분석 수행

⚠️ 주의: 이 파일은 camera_client_python.py와 다른 역할입니다!
- tcp_server.py: 얼굴 분석 서비스 제공 (서버 역할, Port 10000)
- camera_client_python.py: IR Camera에서 Raw Y8 수신 (클라이언트 역할, Port 5001)

Protocol:
1. Client → Server: 이미지 데이터 (JPEG/PNG/Raw Y8 자동 감지)
2. Server: UnifiedFaceAnalyzer로 분석
3. Server → Client: JSON 분석 결과 (TCP_SPEC 형식)

지원 이미지 형식:
- JPEG (시그니처 감지)
- PNG (시그니처 감지)
- Raw Y8 (크기 기반 감지: 1280x800, 1280x720, 1920x1080)

Port: 10000 (default)
"""

import socket
import json
import logging
from typing import Optional, Dict, Any
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import time

from core.unified_analyzer import UnifiedFaceAnalyzer
from utils import get_logger

logger = get_logger(__name__)


class UnifiedFaceAnalysisTCPServer:
    """
    통합 얼굴 분석 TCP 서버

    Features:
    - Raw 이미지 데이터 수신 (Unreal Engine)
    - UnifiedFaceAnalyzer로 분석
    - JSON 결과 전송
    """

    # Enum 매핑 (TCP_SPEC.md 규격)
    HAIRSTYLE_ENUM = {
        "Bangs": 0,
        "All-Back": 1,
        "Center Part": 2,
        "Right Side Part": 3,
        "Left Side Part": 4,
        "Short Hair": 5,
        "Long Hair": 6,
    }

    GENDER_ENUM = {
        "Female": 0,
        "Male": 1,
    }

    FACE_SHAPE_ENUM = {
        "oval": 0,   # 계란형
        "round": 1,  # 둥근형
    }

    EYE_SHAPE_ENUM = {
        "upturned": 0,    # 올라간 눈
        "downturned": 1,  # 내려간 눈
        "neutral": 2,     # 기본형
    }

    def __init__(
        self,
        host: str = '0.0.0.0',
        port: int = 10000,
        max_connections: int = 5
    ):
        """
        TCP 서버 초기화

        Args:
            host: 서버 주소 (기본: 0.0.0.0 - 모든 인터페이스)
            port: 포트 번호 (기본: 5000)
            max_connections: 최대 동시 연결 수
        """
        self.host = host
        self.port = port
        self.max_connections = max_connections

        self.server_socket: Optional[socket.socket] = None
        self.is_running = False

        # UnifiedFaceAnalyzer 초기화
        print("🔧 UnifiedFaceAnalyzer 초기화 중...")
        self.analyzer = UnifiedFaceAnalyzer()
        print("✅ 초기화 완료\n")

    def start(self):
        """서버 시작"""
        try:
            # 소켓 생성
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            # 소켓 옵션 설정 (주소 재사용 허용)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            # 바인딩
            self.server_socket.bind((self.host, self.port))

            # 리스닝
            self.server_socket.listen(self.max_connections)

            self.is_running = True

            print("=" * 80)
            print("  Unified Face Analysis TCP Server")
            print("=" * 80)
            print(f"✅ TCP 서버 시작: {self.host}:{self.port}")
            print(f"📡 연결 대기 중... (최대 {self.max_connections}개 연결)")
            print(f"🎯 분석 모듈: MediaPipe + Hairstyle + Eye Shape + Face Shape")
            print("=" * 80)
            print()

            logger.info(f"TCP server started on {self.host}:{self.port}")

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            print(f"❌ 서버 시작 실패: {e}")
            raise

    def stop(self):
        """서버 종료"""
        self.is_running = False
        if self.server_socket:
            self.server_socket.close()
            logger.info("TCP server stopped")
            print("\n✅ TCP 서버 종료")

    def _detect_image_format(self, data: bytes) -> str:
        """
        이미지 데이터 형식 자동 감지

        Args:
            data: 이미지 바이너리 데이터

        Returns:
            'png', 'jpeg', 'raw_y8', 또는 'unknown'
        """
        if len(data) < 8:
            return 'unknown'

        # PNG 시그니처: 89 50 4E 47 0D 0A 1A 0A
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return 'png'

        # JPEG 시그니처: FF D8 FF
        if data[:3] == b'\xff\xd8\xff':
            return 'jpeg'

        # Raw Y8 추정 (인코딩된 이미지가 아닌 경우)
        # 일반적으로 1280x800 = 1,024,000 bytes
        if len(data) in [1024000, 921600, 2073600]:  # 1280x800, 1280x720, 1920x1080
            return 'raw_y8'

        return 'unknown'

    def _decode_raw_y8(self, data: bytes, width: int = 1280, height: int = 800) -> Optional[np.ndarray]:
        """
        Raw Y8 데이터를 BGR 이미지로 변환

        Args:
            data: Raw Y8 바이너리 데이터
            width: 이미지 너비
            height: 이미지 높이

        Returns:
            numpy array (BGR 포맷) 또는 None
        """
        try:
            expected_size = width * height

            if len(data) != expected_size:
                logger.warning(f"Y8 data size mismatch: expected {expected_size}, got {len(data)}")
                # 일반적인 해상도로 재시도
                common_resolutions = [
                    (1280, 800),
                    (1280, 720),
                    (1920, 1080),
                    (640, 480)
                ]
                for w, h in common_resolutions:
                    if len(data) == w * h:
                        width, height = w, h
                        logger.info(f"Auto-detected resolution: {width}x{height}")
                        break
                else:
                    logger.error("Cannot determine Y8 image resolution")
                    return None

            # Y8 배열로 변환
            y8_array = np.frombuffer(data, dtype=np.uint8)

            # 2D 배열로 reshape
            y8_image = y8_array.reshape((height, width))

            # Grayscale → BGR 변환
            bgr_image = cv2.cvtColor(y8_image, cv2.COLOR_GRAY2BGR)

            logger.info(f"Raw Y8 decoded: {bgr_image.shape}")
            return bgr_image

        except Exception as e:
            logger.error(f"Error decoding raw Y8: {e}")
            return None

    def receive_image(self, client_socket: socket.socket, buffer_size: int = 1024000) -> Optional[np.ndarray]:
        """
        클라이언트로부터 이미지 데이터 수신 (자동 형식 감지)

        Protocol (No size header):
        - 직접 raw 데이터 수신 (JPEG/PNG/Raw Y8)
        - 첫 번째 청크에서 형식 자동 감지

        Args:
            client_socket: 클라이언트 소켓
            buffer_size: 수신 버퍼 크기 (기본: 1024000 = 1280x800 Y8)

        Returns:
            numpy array (BGR 포맷) 또는 None
        """
        try:
            # 1. 첫 번째 청크 수신 (형식 감지용)
            logger.info(f"Waiting to receive image data (buffer_size: {buffer_size})...")
            first_chunk = client_socket.recv(buffer_size)

            if not first_chunk:
                logger.warning("No data received")
                return None

            logger.info(f"Received {len(first_chunk)} bytes")

            # 디버그: 첫 16 bytes hex dump
            hex_preview = ' '.join(f'{b:02x}' for b in first_chunk[:16])
            logger.info(f"First 16 bytes (hex): {hex_preview}")

            # 2. 이미지 형식 자동 감지
            image_format = self._detect_image_format(first_chunk)
            logger.info(f"Detected image format: {image_format}")

            # 3. Raw Y8인 경우 정확한 크기로 수신
            if image_format == 'raw_y8':
                expected_size = len(first_chunk)
                logger.info(f"Raw Y8 detected, received size: {expected_size}")

                # 일반적인 Y8 크기 확인
                if expected_size not in [1024000, 921600, 2073600]:
                    # 크기가 정확하지 않으면 더 수신 필요
                    logger.warning(f"Unexpected Y8 size: {expected_size}, trying to receive more")
                    image_data = first_chunk
                    while len(image_data) < buffer_size:
                        chunk = client_socket.recv(buffer_size - len(image_data))
                        if not chunk:
                            break
                        image_data += chunk
                    logger.info(f"After additional recv: {len(image_data)} bytes total")
                else:
                    image_data = first_chunk
                    logger.info(f"Y8 size is valid: {expected_size} bytes")

                # Raw Y8 디코딩
                logger.info(f"Decoding Y8 data: {len(image_data)} bytes")
                image = self._decode_raw_y8(image_data)

                if image is not None:
                    logger.info(f"Y8 decoding successful: {image.shape}")
                else:
                    logger.error("Y8 decoding failed")

            elif image_format in ['png', 'jpeg']:
                # 인코딩된 이미지는 더 많은 데이터 수신 필요할 수 있음
                image_data = first_chunk

                # PNG/JPEG는 전체 데이터가 필요 (일반적으로 첫 청크에 포함되지만 큰 경우 추가 수신)
                # 타임아웃 설정으로 추가 데이터가 없으면 중단
                client_socket.settimeout(0.1)  # 100ms timeout
                try:
                    while True:
                        chunk = client_socket.recv(65536)
                        if not chunk:
                            break
                        image_data += chunk
                except socket.timeout:
                    # 타임아웃은 정상 (더 이상 데이터 없음)
                    pass
                finally:
                    client_socket.settimeout(None)  # 타임아웃 해제

                # 이미지 디코딩
                image_array = np.frombuffer(image_data, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            else:
                logger.warning(f"Unknown image format, trying standard decode...")
                image_data = first_chunk

                # Fallback: 일반 디코딩 시도
                image_array = np.frombuffer(image_data, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                # 디코딩 실패 시 Raw Y8로 재시도
                if image is None:
                    logger.info("Standard decode failed, trying raw Y8...")
                    image = self._decode_raw_y8(image_data)

            if image is None:
                logger.error("Failed to decode image")
                return None

            logger.info(f"Image received: {image.shape}")
            return image

        except Exception as e:
            logger.error(f"Error receiving image: {e}")
            return None

    def _sanitize_for_json(self, obj):
        """
        JSON 직렬화가 불가능한 객체를 제거/변환

        Args:
            obj: 정리할 객체

        Returns:
            JSON 직렬화 가능한 객체
        """
        try:
            import dlib
            has_dlib = True
        except ImportError:
            has_dlib = False

        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                sanitized = self._sanitize_for_json(v)
                if sanitized is not None:
                    result[k] = sanitized
            return result
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj if self._sanitize_for_json(item) is not None]
        elif isinstance(obj, tuple):
            return [self._sanitize_for_json(item) for item in obj if self._sanitize_for_json(item) is not None]
        elif isinstance(obj, np.ndarray):
            # numpy array는 제거 (visualization_image 등)
            return None
        elif has_dlib and isinstance(obj, (dlib.rectangle, dlib.full_object_detection)):
            # dlib 객체는 제거
            return None
        elif isinstance(obj, (str, int, float, bool, type(None))):
            # JSON 기본 타입
            return obj
        elif hasattr(obj, '__dict__') and not callable(obj):
            # 일반 객체는 딕셔너리로 변환 시도
            try:
                return str(obj)
            except:
                return None
        else:
            # 알 수 없는 타입은 문자열로 변환 시도
            try:
                return str(obj)
            except:
                return None

    def _convert_to_tcp_spec_format(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        UnifiedFaceAnalyzer 결과를 TCP_SPEC.md 형식으로 변환

        Args:
            result: UnifiedFaceAnalyzer 분석 결과

        Returns:
            TCP_SPEC.md 규격의 JSON 딕셔너리
        """
        tcp_result = {}

        # Timestamp
        if 'metadata' in result and 'timestamp' in result['metadata']:
            tcp_result['timestamp'] = result['metadata']['timestamp']
        else:
            from datetime import datetime
            tcp_result['timestamp'] = datetime.now().isoformat()

        # Image path
        if 'metadata' in result and 'image_path' in result['metadata']:
            tcp_result['image_path'] = result['metadata']['image_path']
        else:
            tcp_result['image_path'] = ""

        # Hairstyle
        if 'hairstyle' in result and result['hairstyle'].get('classification'):
            hairstyle_name = result['hairstyle']['classification']
            tcp_result['hairstyle'] = self.HAIRSTYLE_ENUM.get(hairstyle_name, -1)
            tcp_result['hairstyle_name'] = hairstyle_name
        else:
            tcp_result['hairstyle'] = -1
            tcp_result['hairstyle_name'] = "Unknown"

        # Gender (from CLIP results)
        if 'hairstyle' in result and 'clip_results' in result['hairstyle']:
            clip = result['hairstyle']['clip_results']
            gender_name = clip.get('gender', 'Male')
            tcp_result['gender'] = self.GENDER_ENUM.get(gender_name, -1)
            tcp_result['gender_name'] = gender_name

            # Gender confidence (문자열을 float로 변환)
            gender_conf = clip.get('gender_confidence', '0.0')
            if isinstance(gender_conf, str):
                tcp_result['gender_confidence'] = float(gender_conf)
            else:
                tcp_result['gender_confidence'] = float(gender_conf)

            # Glasses
            has_glasses = 1 if clip.get('glasses') == "With Glasses" else 0
            tcp_result['has_glasses'] = has_glasses

            glasses_conf = clip.get('glasses_confidence', '0.0')
            if isinstance(glasses_conf, str):
                tcp_result['glasses_confidence'] = float(glasses_conf)
            else:
                tcp_result['glasses_confidence'] = float(glasses_conf)

            # Beard
            has_beard = 1 if clip.get('beard') == "With Beard" else 0
            tcp_result['has_beard'] = has_beard

            beard_conf = clip.get('beard_confidence', '0.0')
            if isinstance(beard_conf, str):
                tcp_result['beard_confidence'] = float(beard_conf)
            else:
                tcp_result['beard_confidence'] = float(beard_conf)
        else:
            tcp_result['gender'] = -1
            tcp_result['gender_name'] = "Unknown"
            tcp_result['gender_confidence'] = 0.0
            tcp_result['has_glasses'] = 0
            tcp_result['glasses_confidence'] = 0.0
            tcp_result['has_beard'] = 0
            tcp_result['beard_confidence'] = 0.0

        # Face Shape (from MediaPipe)
        if 'mediapipe' in result and 'face_shape_analysis' in result['mediapipe']:
            face_shape_name = result['mediapipe']['face_shape_analysis'].get('face_shape', 'oval')
            tcp_result['face_shape'] = self.FACE_SHAPE_ENUM.get(face_shape_name, -1)
            tcp_result['face_shape_name'] = face_shape_name
        else:
            tcp_result['face_shape'] = -1
            tcp_result['face_shape_name'] = "Unknown"

        # Eye Shape (from MediaPipe)
        if 'mediapipe' in result and 'eye_analysis' in result['mediapipe']:
            eye_shape_name = result['mediapipe']['eye_analysis'].get('overall_eye_shape', 'almond')
            tcp_result['eye_shape'] = self.EYE_SHAPE_ENUM.get(eye_shape_name, -1)
            tcp_result['eye_shape_name'] = eye_shape_name
        else:
            tcp_result['eye_shape'] = -1
            tcp_result['eye_shape_name'] = "Unknown"

        return tcp_result

    def send_json_result(self, client_socket: socket.socket, result: Dict[str, Any]) -> bool:
        """
        JSON 결과를 클라이언트에 전송

        Protocol (No size header):
        - 직접 JSON 문자열 전송 (UTF-8)

        Args:
            client_socket: 클라이언트 소켓
            result: 분석 결과 딕셔너리

        Returns:
            전송 성공 여부
        """
        try:
            # TCP_SPEC.md 형식으로 변환
            logger.debug("Converting result to TCP_SPEC format...")
            tcp_result = self._convert_to_tcp_spec_format(result)
            logger.debug(f"Conversion complete. Keys: {list(tcp_result.keys())}")

            # JSON 직렬화
            logger.debug("Attempting JSON serialization...")
            json_str = json.dumps(tcp_result, ensure_ascii=False)
            json_bytes = json_str.encode('utf-8')
            logger.debug(f"JSON serialization successful: {len(json_bytes)} bytes")

            # JSON 데이터 전송 (4-byte size header 없이)
            client_socket.sendall(json_bytes)

            logger.info(f"JSON result sent: {len(json_bytes)} bytes (no size header)")
            return True

        except TypeError as e:
            logger.error(f"JSON serialization error: {e}")
            logger.debug(f"Result type: {type(result)}")
            if isinstance(result, dict):
                for key, value in result.items():
                    logger.debug(f"  {key}: {type(value)}")
            return False
        except Exception as e:
            logger.error(f"Error sending JSON result: {e}", exc_info=True)
            return False

    def handle_client(self, client_socket: socket.socket, client_address: tuple):
        """
        클라이언트 요청 처리

        Args:
            client_socket: 클라이언트 소켓
            client_address: 클라이언트 주소
        """
        print(f"🔗 클라이언트 연결: {client_address[0]}:{client_address[1]}")
        logger.info(f"Client connected: {client_address}")

        try:
            while self.is_running:
                # 1. 이미지 수신
                print(f"\n📥 이미지 수신 중...")
                image = self.receive_image(client_socket)

                if image is None:
                    print("❌ 이미지 수신 실패")
                    break

                print(f"✅ 이미지 수신 완료: {image.shape}")

                # 2. 이미지 분석
                print("🔍 얼굴 분석 중...")
                start_time = time.time()

                # 임시 파일로 저장 (UnifiedFaceAnalyzer가 파일 경로를 요구함)
                import tempfile
                import os

                # 크로스 플랫폼 임시 파일 생성
                temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg', prefix='unreal_')
                os.close(temp_fd)  # 파일 디스크립터 닫기

                logger.info(f"Saving image to temp file: {temp_path}")
                write_success = cv2.imwrite(temp_path, image)

                if not write_success:
                    logger.error(f"Failed to write image to {temp_path}")
                    print(f"❌ 이미지 저장 실패: {temp_path}")
                    break

                logger.info(f"Image saved successfully, starting analysis...")
                result = self.analyzer.analyze_image(temp_path)
                logger.info(f"Analysis completed")

                # 임시 파일 삭제
                try:
                    os.unlink(temp_path)
                except:
                    pass

                analysis_time = (time.time() - start_time) * 1000
                print(f"✅ 분석 완료: {analysis_time:.2f}ms")

                # 3. 결과 요약 출력
                if result.get('success'):
                    if 'mediapipe' in result and result['mediapipe'].get('success'):
                        mp = result['mediapipe']
                        print(f"   📍 MediaPipe: {mp.get('landmarks_count', 0)} landmarks")
                        if 'eye_analysis' in mp:
                            print(f"   👁️  Eye: {mp['eye_analysis']['overall_eye_shape']}")
                        if 'face_shape_analysis' in mp:
                            print(f"   😊 Face: {mp['face_shape_analysis']['face_shape']}")

                    if 'hairstyle' in result:
                        hs = result['hairstyle']
                        print(f"   💇 Hairstyle: {hs.get('classification', 'Unknown')}")

                # 4. JSON 결과 전송
                print("📤 JSON 결과 전송 중...")
                success = self.send_json_result(client_socket, result)

                if success:
                    print("✅ 결과 전송 완료")
                else:
                    print("❌ 결과 전송 실패")
                    break

        except Exception as e:
            logger.error(f"Error handling client: {e}", exc_info=True)
            print(f"❌ 클라이언트 처리 오류: {e}")

        finally:
            client_socket.close()
            print(f"🔌 클라이언트 연결 종료: {client_address[0]}:{client_address[1]}\n")
            logger.info(f"Client disconnected: {client_address}")

    def run(self):
        """서버 실행 (메인 루프)"""
        if not self.is_running:
            self.start()

        try:
            while self.is_running:
                # 클라이언트 연결 대기
                client_socket, client_address = self.server_socket.accept()

                # 클라이언트 처리 (동기 방식)
                self.handle_client(client_socket, client_address)

        except KeyboardInterrupt:
            print("\n\n⚠️  서버 종료 요청 (Ctrl+C)")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            print(f"\n❌ 서버 오류: {e}")
        finally:
            self.stop()

    def __enter__(self):
        """Context manager 진입"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.stop()


def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Config 파일 로드

    Args:
        config_path: config.yaml 파일 경로

    Returns:
        설정 딕셔너리
    """
    import yaml
    import os

    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return {
            'server': {
                'host': '0.0.0.0',
                'port': 5001,
                'max_connections': 5
            }
        }

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Config loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {
            'server': {
                'host': '0.0.0.0',
                'port': 5001,
                'max_connections': 5
            }
        }


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='Unified Face Analysis TCP Server')
    parser.add_argument('--config', default='config.yaml', help='Config file path (default: config.yaml)')
    parser.add_argument('--host', default=None, help='Server host (overrides config)')
    parser.add_argument('--port', type=int, default=None, help='Server port (overrides config)')
    parser.add_argument('--max-connections', type=int, default=None, help='Max connections (overrides config)')
    parser.add_argument('--env', default='production', choices=['development', 'production', 'restricted'],
                        help='Environment (default: production)')

    args = parser.parse_args()

    # Config 파일 로드
    config = load_config(args.config)

    # 환경별 설정 적용
    if 'environments' in config and args.env in config['environments']:
        env_config = config['environments'][args.env]
        server_config = {**config.get('server', {}), **env_config}
    else:
        server_config = config.get('server', {})

    # 명령줄 인자가 우선순위 (config 덮어쓰기)
    host = args.host if args.host is not None else server_config.get('host', '0.0.0.0')
    port = args.port if args.port is not None else server_config.get('port', 5001)
    max_connections = args.max_connections if args.max_connections is not None else server_config.get('max_connections', 5)

    print(f"🔧 Server Configuration:")
    print(f"   Environment: {args.env}")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Max Connections: {max_connections}")

    # 서버 생성 및 실행
    server = UnifiedFaceAnalysisTCPServer(
        host=host,
        port=port,
        max_connections=max_connections
    )

    server.run()


if __name__ == '__main__':
    main()
