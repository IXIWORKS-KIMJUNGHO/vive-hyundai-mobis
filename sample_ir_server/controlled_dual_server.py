#!/usr/bin/env python3
"""
Controlled Dual Purpose Server
- Port 5000: 명령 수신 및 JSON 전송
  - "icc_start" 수신 → JSON 전송 + Port 5001 스트리밍 시작
  - "icc_stop" 수신 → Port 5001 스트리밍 중지
- Port 5001: Raw Y8 데이터 스트리밍 (제어됨)
"""

import socket
import threading
import json
import cv2
import numpy as np
import time
import struct
from pathlib import Path
from typing import Optional
import argparse


class ControlledJSONServer:
    """
    명령 기반 JSON 스트리밍 서버 (Port 5000)
    - "icc_start" → JSON 연속 전송 + Y8 스트리밍 시작
    - "icc_stop" → 모든 스트리밍 중지
    """

    def __init__(self, port: int, json_path: str, y8_server, fps: int = 30):
        """
        Args:
            port: TCP 포트
            json_path: 전송할 JSON 파일 경로
            y8_server: Y8RawDataServer 인스턴스 (제어용)
            fps: JSON 전송 FPS
        """
        self.port = port
        self.json_path = json_path
        self.y8_server = y8_server
        self.fps = fps
        self.json_bytes = self._load_json()

        self.server_socket: Optional[socket.socket] = None
        self.is_running = False
        self.is_streaming = False  # JSON 스트리밍 제어 플래그
        self.clients = []
        self.clients_lock = threading.Lock()

        print(f"📄 [JSON Server] 초기화")
        print(f"   포트: {port}")
        print(f"   JSON 파일: {json_path}")
        print(f"   JSON 크기: {len(self.json_bytes)} bytes")
        print(f"   전송 FPS: {fps}")

    def _load_json(self) -> bytes:
        """JSON 파일을 바이트로 로드 (파싱 없이 전체 전송)"""
        if not Path(self.json_path).exists():
            raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {self.json_path}")

        # 파일 전체를 바이트로 읽기 (파싱 안함)
        with open(self.json_path, 'rb') as f:
            json_bytes = f.read()

        print(f"   ✅ JSON 파일 로드 완료: {len(json_bytes)} bytes")
        return json_bytes

    def start(self):
        """서버 시작 (리스닝 + 스트리밍)"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Nagle 알고리즘 비활성화 (즉시 전송)
            self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.listen(5)

            self.is_running = True
            print(f"✅ [JSON Server] TCP 서버 시작: 0.0.0.0:{self.port}")
            print(f"   ⏸️  스트리밍 대기 중 (icc_start 명령 대기)\n")

            # 클라이언트 수락 + 명령 처리
            accept_thread = threading.Thread(target=self._accept_clients, daemon=True)
            accept_thread.start()

            # JSON 스트리밍 스레드
            stream_thread = threading.Thread(target=self._stream_json, daemon=True)
            stream_thread.start()

        except Exception as e:
            print(f"❌ [JSON Server] 서버 시작 실패: {e}")
            raise

    def stop(self):
        """서버 종료"""
        self.is_running = False
        self.is_streaming = False

        with self.clients_lock:
            for client_socket, addr in self.clients:
                try:
                    client_socket.close()
                except:
                    pass
            self.clients.clear()

        if self.server_socket:
            self.server_socket.close()
        print(f"✅ [JSON Server] TCP 서버 종료")

    def start_streaming(self):
        """JSON 스트리밍 시작"""
        if not self.is_streaming:
            self.is_streaming = True
            print(f"▶️  [JSON Server] 스트리밍 시작!")

    def stop_streaming(self):
        """JSON 스트리밍 중지"""
        if self.is_streaming:
            self.is_streaming = False
            print(f"⏹️  [JSON Server] 스트리밍 중지!")

    def _accept_clients(self):
        """클라이언트 연결 수락 및 명령 처리"""
        while self.is_running:
            try:
                client_socket, client_address = self.server_socket.accept()
                print(f"🔗 [JSON Server] 클라이언트 연결: {client_address[0]}:{client_address[1]}")

                # 명령 처리 스레드
                command_thread = threading.Thread(
                    target=self._handle_commands,
                    args=(client_socket, client_address),
                    daemon=True
                )
                command_thread.start()

            except Exception as e:
                if self.is_running:
                    print(f"❌ [JSON Server] 클라이언트 수락 에러: {e}")
                break

    def _handle_commands(self, client_socket: socket.socket, client_address: tuple):
        """클라이언트 명령 처리 (별도 스레드)"""
        try:
            # 클라이언트 소켓 최적화
            client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)  # 1MB 송신 버퍼

            # 스트리밍 클라이언트 리스트에 추가
            with self.clients_lock:
                self.clients.append((client_socket, client_address))

            # 명령 대기
            while self.is_running:
                command_data = client_socket.recv(1024)
                if not command_data:
                    break

                command = command_data.decode('utf-8').strip()
                print(f"📨 [JSON Server] 명령 수신: '{command}' from {client_address[0]}:{client_address[1]}")

                if command == "icc_start":
                    # JSON + Y8 스트리밍 시작
                    print(f"🎬 [JSON Server] 스트리밍 시작 명령")
                    self.start_streaming()
                    self.y8_server.start_streaming()

                elif command == "icc_stop":
                    # 모든 스트리밍 중지
                    print(f"🛑 [JSON Server] 스트리밍 중지 명령")
                    self.stop_streaming()
                    self.y8_server.stop_streaming()

                else:
                    print(f"⚠️  [JSON Server] 알 수 없는 명령: '{command}'")

        except Exception as e:
            print(f"❌ [JSON Server] 명령 처리 에러: {client_address} - {e}")

        finally:
            with self.clients_lock:
                if (client_socket, client_address) in self.clients:
                    self.clients.remove((client_socket, client_address))
            try:
                client_socket.close()
            except:
                pass
            print(f"🔌 [JSON Server] 클라이언트 연결 해제: {client_address[0]}:{client_address[1]}")

    def _stream_json(self):
        """JSON 스트리밍 (제어 가능)"""
        frame_number = 0
        frame_interval = 1.0 / self.fps

        while self.is_running:
            # 스트리밍이 활성화되지 않으면 대기
            if not self.is_streaming:
                time.sleep(0.1)
                continue

            start_time = time.time()

            # 모든 클라이언트에게 JSON 전송
            with self.clients_lock:
                disconnected_clients = []

                for client_socket, client_address in self.clients:
                    try:
                        client_socket.sendall(self.json_bytes)
                    except Exception as e:
                        print(f"❌ [JSON Server] 클라이언트 전송 실패: {client_address} - {e}")
                        disconnected_clients.append((client_socket, client_address))

                # 연결 끊긴 클라이언트 제거
                for client in disconnected_clients:
                    try:
                        client[0].close()
                    except:
                        pass
                    self.clients.remove(client)
                    print(f"🔌 [JSON Server] 클라이언트 연결 해제: {client[1]}")

            frame_number += 1

            # FPS 유지
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)

            # 주기적 상태 출력
            if frame_number % (self.fps * 5) == 0:
                print(f"📊 [JSON Server] 프레임: {frame_number}, 클라이언트: {len(self.clients)}, 스트리밍: {'ON' if self.is_streaming else 'OFF'}")


class ControlledY8Server:
    """
    제어 가능한 Y8 Raw Data 서버 (Port 5001)
    외부 명령으로 스트리밍 시작/중지 제어
    """

    def __init__(self, port: int, image_path: str, fps: int = 30, width: int = 1280, height: int = 800):
        self.port = port
        self.image_path = image_path
        self.fps = fps
        self.width = width
        self.height = height

        self.y8_data = self._load_and_convert_image()

        self.server_socket: Optional[socket.socket] = None
        self.is_running = False
        self.is_streaming = False  # 스트리밍 제어 플래그
        self.clients = []
        self.clients_lock = threading.Lock()

        print(f"🎥 [Y8 Server] 초기화")
        print(f"   포트: {port}")
        print(f"   이미지: {image_path}")
        print(f"   해상도: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Y8 데이터 크기: {len(self.y8_data)} bytes")

    def _load_and_convert_image(self) -> bytes:
        """PNG → Y8 변환"""
        if not Path(self.image_path).exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {self.image_path}")

        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"이미지 로드 실패: {self.image_path}")

        if image.shape != (self.height, self.width):
            original_shape = image.shape
            image = cv2.resize(image, (self.width, self.height))
            print(f"   ℹ️  이미지 리사이즈: {original_shape} → ({self.height}, {self.width})")

        y8_bytes = image.tobytes()
        print(f"   ✅ PNG → Y8 변환 완료: {len(y8_bytes)} bytes")
        return y8_bytes

    def start(self):
        """서버 시작 (리스닝만, 스트리밍은 별도 제어)"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Nagle 알고리즘 비활성화 (즉시 전송)
            self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.listen(5)

            self.is_running = True
            print(f"✅ [Y8 Server] TCP 서버 시작: 0.0.0.0:{self.port}")
            print(f"   ⏸️  스트리밍 대기 중 (icc_start 명령 대기)\n")

            # 클라이언트 수락
            accept_thread = threading.Thread(target=self._accept_clients, daemon=True)
            accept_thread.start()

            # 스트리밍 스레드 (제어 가능)
            stream_thread = threading.Thread(target=self._stream_frames, daemon=True)
            stream_thread.start()

        except Exception as e:
            print(f"❌ [Y8 Server] 서버 시작 실패: {e}")
            raise

    def stop(self):
        """서버 종료"""
        self.is_running = False
        self.is_streaming = False

        with self.clients_lock:
            for client_socket, addr in self.clients:
                try:
                    client_socket.close()
                except:
                    pass
            self.clients.clear()

        if self.server_socket:
            self.server_socket.close()

        print(f"✅ [Y8 Server] TCP 서버 종료")

    def start_streaming(self):
        """스트리밍 시작"""
        if not self.is_streaming:
            self.is_streaming = True
            print(f"▶️  [Y8 Server] 스트리밍 시작!")

    def stop_streaming(self):
        """스트리밍 중지"""
        if self.is_streaming:
            self.is_streaming = False
            print(f"⏹️  [Y8 Server] 스트리밍 중지!")

    def _accept_clients(self):
        """클라이언트 연결 수락"""
        while self.is_running:
            try:
                client_socket, client_address = self.server_socket.accept()

                # 클라이언트 소켓 최적화
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)  # 2MB 송신 버퍼

                with self.clients_lock:
                    self.clients.append((client_socket, client_address))

                print(f"🔗 [Y8 Server] 클라이언트 연결: {client_address[0]}:{client_address[1]}")
                print(f"   총 클라이언트 수: {len(self.clients)}")

            except Exception as e:
                if self.is_running:
                    print(f"❌ [Y8 Server] 클라이언트 수락 에러: {e}")
                break

    def _stream_frames(self):
        """프레임 스트리밍 (제어 가능)"""
        frame_number = 0
        frame_interval = 1.0 / self.fps

        while self.is_running:
            # 스트리밍이 활성화되지 않으면 대기
            if not self.is_streaming:
                time.sleep(0.1)
                continue

            start_time = time.time()

            # 모든 클라이언트에게 전송
            with self.clients_lock:
                disconnected_clients = []

                for client_socket, client_address in self.clients:
                    try:
                        client_socket.sendall(self.y8_data)
                    except Exception as e:
                        print(f"❌ [Y8 Server] 클라이언트 전송 실패: {client_address} - {e}")
                        disconnected_clients.append((client_socket, client_address))

                # 연결 끊긴 클라이언트 제거
                for client in disconnected_clients:
                    try:
                        client[0].close()
                    except:
                        pass
                    self.clients.remove(client)
                    print(f"🔌 [Y8 Server] 클라이언트 연결 해제: {client[1]}")

            frame_number += 1

            # FPS 유지
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed)
            time.sleep(sleep_time)

            # 주기적 상태 출력
            if frame_number % (self.fps * 5) == 0:
                print(f"📊 [Y8 Server] 프레임: {frame_number}, 클라이언트: {len(self.clients)}, 스트리밍: {'ON' if self.is_streaming else 'OFF'}")


class ControlledDualServer:
    """제어 가능한 듀얼 서버"""

    def __init__(
        self,
        json_path: str,
        image_path: str,
        control_port: int = 5000,
        y8_port: int = 5001,
        fps: int = 30,
        width: int = 1280,
        height: int = 800
    ):
        print("=" * 80)
        print("  Controlled Dual Purpose Server")
        print("=" * 80)
        print()

        # Y8 서버 먼저 생성
        self.y8_server = ControlledY8Server(y8_port, image_path, fps, width, height)
        print()

        # JSON 서버 생성 (Y8 서버 참조, 동일한 FPS 사용)
        self.control_server = ControlledJSONServer(control_port, json_path, self.y8_server, fps)

        print()
        print("=" * 80)

    def start(self):
        """두 서버 시작"""
        self.y8_server.start()
        time.sleep(0.5)
        self.control_server.start()

    def stop(self):
        """두 서버 종료"""
        self.control_server.stop()
        self.y8_server.stop()

    def run(self):
        """메인 루프 실행"""
        self.start()

        try:
            print("\n📡 제어 가능한 듀얼 서버 실행 중...")
            print("   - Port 5000: 명령 수신 (icc_start/icc_stop) + JSON 전송")
            print("   - Port 5001: Raw Y8 스트리밍 (제어됨)")
            print("\n📝 사용법:")
            print("   1. Port 5000에 'icc_start' 전송 → JSON 수신 + Y8 스트리밍 시작")
            print("   2. Port 5001에 연결하여 Y8 스트림 수신")
            print("   3. Port 5000에 'icc_stop' 전송 → Y8 스트리밍 중지")
            print("\n⚠️  종료하려면 Ctrl+C를 누르세요.\n")

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\n⚠️  서버 종료 요청 (Ctrl+C)")
        finally:
            self.stop()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Controlled Dual Purpose Server')
    parser.add_argument('--json', type=str, default='result.json', help='JSON file path')
    parser.add_argument('--image', type=str, default='camera_capture_20250513_185039.png', help='PNG image path')
    parser.add_argument('--control-port', type=int, default=5000, help='Control server port (default: 5000)')
    parser.add_argument('--y8-port', type=int, default=5001, help='Y8 server port (default: 5001)')
    parser.add_argument('--width', type=int, default=1280, help='Image width (default: 1280)')
    parser.add_argument('--height', type=int, default=800, help='Image height (default: 800)')
    parser.add_argument('--fps', type=int, default=30, help='Y8 streaming FPS (default: 30)')

    args = parser.parse_args()

    server = ControlledDualServer(
        json_path=args.json,
        image_path=args.image,
        control_port=args.control_port,
        y8_port=args.y8_port,
        fps=args.fps,
        width=args.width,
        height=args.height
    )

    server.run()


if __name__ == '__main__':
    main()
