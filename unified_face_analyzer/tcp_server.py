#!/usr/bin/env python3
"""
Unified Face Analyzer TCP Server
Unreal Engine과 연동하여 실시간 얼굴 분석 수행

Protocol:
1. Unreal → Python: Raw image data (바이너리)
2. Python: UnifiedFaceAnalyzer로 분석
3. Python → Unreal: JSON 결과

Port: 5000 (default)
"""

import socket
import json
import logging
import struct
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
        port: int = 5000,
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

    def receive_image(self, client_socket: socket.socket) -> Optional[np.ndarray]:
        """
        클라이언트로부터 이미지 데이터 수신

        Protocol:
        1. 4 bytes: 이미지 데이터 크기 (uint32, little-endian)
        2. N bytes: 이미지 바이너리 데이터 (JPEG/PNG)

        Args:
            client_socket: 클라이언트 소켓

        Returns:
            numpy array (BGR 포맷) 또는 None
        """
        try:
            # 1. 이미지 크기 수신 (4 bytes)
            size_data = client_socket.recv(4)
            if len(size_data) < 4:
                logger.warning("Failed to receive image size")
                return None

            image_size = struct.unpack('<I', size_data)[0]  # little-endian uint32
            logger.debug(f"Image size: {image_size} bytes")

            # 2. 이미지 데이터 수신
            image_data = b''
            remaining = image_size

            while remaining > 0:
                chunk = client_socket.recv(min(remaining, 65536))  # 64KB chunks
                if not chunk:
                    logger.error("Connection closed while receiving image")
                    return None
                image_data += chunk
                remaining -= len(chunk)

            # 3. 이미지 디코딩
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

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

        Protocol:
        1. 4 bytes: JSON 데이터 크기 (uint32, little-endian)
        2. N bytes: JSON 문자열 (UTF-8)

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

            # 크기 전송 (4 bytes)
            size_bytes = struct.pack('<I', len(json_bytes))
            client_socket.sendall(size_bytes)

            # JSON 데이터 전송
            client_socket.sendall(json_bytes)

            logger.info(f"JSON result sent: {len(json_bytes)} bytes")
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
                temp_path = "/tmp/unreal_temp_image.jpg"
                cv2.imwrite(temp_path, image)

                result = self.analyzer.analyze_image(temp_path)

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
