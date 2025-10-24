#!/usr/bin/env python3
"""
간단한 TCP 서버 - Debug/Production 모드 지원
- Debug Mode (0): "start" 명령으로 샘플 이미지 분석
- Production Mode (1): Raw 이미지 데이터 수신 → 분석
"""
import socket
import json
import time
import cv2
import numpy as np
from pathlib import Path
from core.unified_analyzer import UnifiedFaceAnalyzer
from utils import get_logger, get_config

logger = get_logger(__name__)


class SimpleTCPServer:
    """간단한 텍스트 기반 TCP 서버"""

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

    def __init__(self, host: str = "0.0.0.0", port: int = 10000, mode: int = 0, sample_image: str = None):
        """
        Args:
            host: 서버 호스트
            port: 서버 포트
            mode: 0=Debug (start 명령), 1=Production (Raw 이미지 데이터)
            sample_image: Debug 모드에서 사용할 샘플 이미지 경로
        """
        self.host = host
        self.port = port
        self.mode = mode
        self.sample_image = sample_image or "sample_images/camera_capture_20250513_180034.png"
        self.analyzer = UnifiedFaceAnalyzer()
        self.server_socket = None
        self.is_running = False

    def start(self):
        """서버 시작"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.is_running = True

        mode_name = "Debug" if self.mode == 0 else "Production"
        print("=" * 80)
        print(f"  📡 Simple TCP Server Started ({mode_name} Mode)")
        print("=" * 80)
        print(f"🌐 Host: {self.host}")
        print(f"🔌 Port: {self.port}")
        print(f"🔧 Mode: {self.mode} ({mode_name})")
        if self.mode == 0:
            print(f"📸 Sample Image: {self.sample_image}")
            print(f"💬 Protocol: Send 'start' to get analysis result")
        else:
            print(f"📸 Protocol: Send raw image data (PNG/JPEG)")
        print("=" * 80)
        print()

        logger.info(f"Simple TCP server started on {self.host}:{self.port} in {mode_name} mode")

    def stop(self):
        """서버 종료"""
        self.is_running = False
        if self.server_socket:
            self.server_socket.close()
        logger.info("Server stopped")

    def analyze_sample_image(self):
        """샘플 이미지 분석 (Debug 모드)"""
        try:
            logger.info(f"Analyzing sample image: {self.sample_image}")
            result = self.analyzer.analyze_image(self.sample_image)
            return result
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {"success": False, "error": str(e)}

    def receive_image_data(self, client_socket: socket.socket) -> bytes:
        """Production 모드: Raw 이미지 데이터 수신"""
        try:
            chunks = []
            total_received = 0

            logger.info("Receiving image data...")

            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                total_received += len(chunk)

            image_data = b''.join(chunks)
            logger.info(f"Received {total_received} bytes of image data")

            return image_data

        except Exception as e:
            logger.error(f"Error receiving image data: {e}")
            return b''

    def analyze_image_data(self, image_data: bytes):
        """Production 모드: Raw 이미지 데이터 분석"""
        try:
            # bytes를 numpy array로 변환
            nparr = np.frombuffer(image_data, np.uint8)

            # 이미지 디코딩 (PNG/JPEG)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Failed to decode image data")

            logger.info(f"Image decoded: {img.shape}")

            # 임시 파일로 저장 (분석을 위해)
            temp_path = "/tmp/temp_received_image.png"
            cv2.imwrite(temp_path, img)

            # 분석 실행
            result = self.analyzer.analyze_image(temp_path)

            return result

        except Exception as e:
            logger.error(f"Error analyzing image data: {e}")
            return {"success": False, "error": str(e)}

    def convert_to_tcp_spec(self, result: dict) -> dict:
        """분석 결과를 TCP_SPEC 형식으로 변환"""
        tcp_result = {}

        # Timestamp
        if 'metadata' in result and 'timestamp' in result['metadata']:
            tcp_result['timestamp'] = result['metadata']['timestamp']
        else:
            tcp_result['timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Image path
        tcp_result['image_path'] = self.sample_image

        # Hairstyle
        if 'hairstyle' in result and result['hairstyle'].get('classification'):
            hairstyle_name = result['hairstyle']['classification']
            tcp_result['hairstyle'] = self.HAIRSTYLE_ENUM.get(hairstyle_name, -1)
            tcp_result['hairstyle_name'] = hairstyle_name
        else:
            tcp_result['hairstyle'] = -1
            tcp_result['hairstyle_name'] = "Unknown"

        # Gender
        if 'hairstyle' in result and 'gender_analysis' in result['hairstyle']:
            gender_data = result['hairstyle']['gender_analysis']
            gender_name = gender_data.get('gender', 'Unknown')
            tcp_result['gender'] = self.GENDER_ENUM.get(gender_name, -1)
            tcp_result['gender_name'] = gender_name
            tcp_result['gender_confidence'] = float(gender_data.get('confidence', 0.0))
        else:
            tcp_result['gender'] = -1
            tcp_result['gender_name'] = "Unknown"
            tcp_result['gender_confidence'] = 0.0

        # Glasses
        if 'hairstyle' in result and 'glasses_analysis' in result['hairstyle']:
            glasses_data = result['hairstyle']['glasses_analysis']
            tcp_result['has_glasses'] = 1 if glasses_data.get('has_glasses', False) else 0
            tcp_result['glasses_confidence'] = float(glasses_data.get('confidence', 0.0))
        else:
            tcp_result['has_glasses'] = 0
            tcp_result['glasses_confidence'] = 0.0

        # Beard
        if 'hairstyle' in result and 'beard_analysis' in result['hairstyle']:
            beard_data = result['hairstyle']['beard_analysis']
            tcp_result['has_beard'] = 1 if beard_data.get('has_beard', False) else 0
            tcp_result['beard_confidence'] = float(beard_data.get('confidence', 0.0))
        else:
            tcp_result['has_beard'] = 0
            tcp_result['beard_confidence'] = 0.0

        # Face Shape
        if 'mediapipe' in result and 'face_shape_analysis' in result['mediapipe']:
            face_shape_name = result['mediapipe']['face_shape_analysis'].get('face_shape', 'oval')
            tcp_result['face_shape'] = self.FACE_SHAPE_ENUM.get(face_shape_name, -1)
            tcp_result['face_shape_name'] = face_shape_name
        else:
            tcp_result['face_shape'] = -1
            tcp_result['face_shape_name'] = "Unknown"

        # Eye Shape
        if 'mediapipe' in result and 'eye_analysis' in result['mediapipe']:
            eye_shape_name = result['mediapipe']['eye_analysis'].get('overall_eye_shape', 'neutral')
            tcp_result['eye_shape'] = self.EYE_SHAPE_ENUM.get(eye_shape_name, -1)
            tcp_result['eye_shape_name'] = eye_shape_name
        else:
            tcp_result['eye_shape'] = -1
            tcp_result['eye_shape_name'] = "Unknown"

        return tcp_result

    def handle_client(self, client_socket: socket.socket, client_address: tuple):
        """클라이언트 처리 (Mode에 따라 분기)"""
        print(f"🔗 Client connected: {client_address[0]}:{client_address[1]}")
        logger.info(f"Client connected: {client_address}")

        try:
            if self.mode == 0:
                # Debug Mode: "start" 명령 처리
                self._handle_debug_mode(client_socket)
            else:
                # Production Mode: Raw 이미지 데이터 처리
                self._handle_production_mode(client_socket)

        except Exception as e:
            logger.error(f"Error handling client: {e}", exc_info=True)
            print(f"❌ Error: {e}")
        finally:
            client_socket.close()
            print(f"🔌 Client disconnected: {client_address[0]}:{client_address[1]}\n")
            logger.info(f"Client disconnected: {client_address}")

    def _handle_debug_mode(self, client_socket: socket.socket):
        """Debug Mode: "start" 명령 처리"""
        while self.is_running:
            # 명령어 수신
            data = client_socket.recv(1024)
            if not data:
                print("❌ No data received")
                break

            command = data.decode('utf-8').strip()
            print(f"📨 Command received: '{command}'")
            logger.info(f"Command received: {command}")

            # "start" 명령어 확인
            if command.lower() == "start":
                print("🔍 Analyzing sample image...")
                start_time = time.time()

                # 샘플 이미지 분석
                result = self.analyze_sample_image()

                analysis_time = (time.time() - start_time) * 1000
                print(f"✅ Analysis completed in {analysis_time:.2f}ms")

                # TCP_SPEC 형식으로 변환
                tcp_result = self.convert_to_tcp_spec(result)

                # JSON 생성 및 전송
                json_str = json.dumps(tcp_result, ensure_ascii=False, indent=2)
                json_bytes = json_str.encode('utf-8')

                print(f"📤 Sending JSON result ({len(json_bytes)} bytes)...")
                client_socket.sendall(json_bytes)
                print("✅ Result sent successfully")
                logger.info(f"JSON result sent: {len(json_bytes)} bytes")

                # 결과 미리보기
                print("\n" + "=" * 40)
                print("📊 Analysis Result:")
                print(f"   Hairstyle: {tcp_result['hairstyle_name']}")
                print(f"   Gender: {tcp_result['gender_name']}")
                print(f"   Face Shape: {tcp_result['face_shape_name']}")
                print(f"   Eye Shape: {tcp_result['eye_shape_name']}")
                print("=" * 40 + "\n")
            else:
                # 알 수 없는 명령어
                error_msg = json.dumps({"error": f"Unknown command: {command}"})
                client_socket.sendall(error_msg.encode('utf-8'))
                print(f"⚠️  Unknown command: '{command}'")

    def _handle_production_mode(self, client_socket: socket.socket):
        """Production Mode: Raw 이미지 데이터 처리"""
        print("📸 Production Mode: Waiting for image data...")
        start_time = time.time()

        # 이미지 데이터 수신
        image_data = self.receive_image_data(client_socket)

        if not image_data:
            error_msg = json.dumps({"error": "No image data received"})
            client_socket.sendall(error_msg.encode('utf-8'))
            print("❌ No image data received")
            return

        receive_time = (time.time() - start_time) * 1000
        print(f"✅ Image received in {receive_time:.2f}ms")

        # 이미지 분석
        print("🔍 Analyzing received image...")
        analysis_start = time.time()

        result = self.analyze_image_data(image_data)

        analysis_time = (time.time() - analysis_start) * 1000
        print(f"✅ Analysis completed in {analysis_time:.2f}ms")

        # TCP_SPEC 형식으로 변환
        tcp_result = self.convert_to_tcp_spec(result)

        # JSON 생성 및 전송
        json_str = json.dumps(tcp_result, ensure_ascii=False, indent=2)
        json_bytes = json_str.encode('utf-8')

        print(f"📤 Sending JSON result ({len(json_bytes)} bytes)...")
        client_socket.sendall(json_bytes)
        print("✅ Result sent successfully")
        logger.info(f"JSON result sent: {len(json_bytes)} bytes")

        # 결과 미리보기
        print("\n" + "=" * 40)
        print("📊 Analysis Result:")
        print(f"   Hairstyle: {tcp_result['hairstyle_name']}")
        print(f"   Gender: {tcp_result['gender_name']}")
        print(f"   Face Shape: {tcp_result['face_shape_name']}")
        print(f"   Eye Shape: {tcp_result['eye_shape_name']}")
        print("=" * 40 + "\n")

    def run(self):
        """서버 실행"""
        if not self.is_running:
            self.start()

        try:
            while self.is_running:
                client_socket, client_address = self.server_socket.accept()
                self.handle_client(client_socket, client_address)
        except KeyboardInterrupt:
            print("\n⚠️  Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            print(f"❌ Server error: {e}")
        finally:
            self.stop()
            print("\n✅ Server stopped\n")


def main():
    import argparse

    # Config 로드
    config = get_config()

    parser = argparse.ArgumentParser(description='Simple TCP Server for Face Analysis')
    parser.add_argument('--host', help=f'Server host (default: from config)')
    parser.add_argument('--port', type=int, help=f'Server port (default: from config)')
    parser.add_argument('--mode', type=int, choices=[0, 1], help='0=Debug, 1=Production (default: from config)')
    parser.add_argument('--image', help='Sample image path for Debug mode')

    args = parser.parse_args()

    # Config 또는 args에서 값 가져오기
    host = args.host or config.server.host
    port = args.port or config.server.port
    mode = args.mode if args.mode is not None else config.server.mode

    server = SimpleTCPServer(
        host=host,
        port=port,
        mode=mode,
        sample_image=args.image
    )

    server.run()


if __name__ == "__main__":
    main()
