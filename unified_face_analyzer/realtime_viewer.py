#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Y8 Image Viewer
TCP 서버로 들어오는 Y8 데이터를 실시간으로 표시

사용법:
    python realtime_viewer.py [--port PORT] [--width WIDTH] [--height HEIGHT]

예시:
    python realtime_viewer.py --port 10000
    python realtime_viewer.py --port 10000 --width 1280 --height 800
"""

import socket
import threading
import time
import numpy as np
import cv2
import argparse
from queue import Queue, Empty
from pathlib import Path


class RealtimeY8Viewer:
    """실시간 Y8 이미지 뷰어"""

    def __init__(self, host='0.0.0.0', port=10000, width=1280, height=800):
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.expected_size = width * height

        self.server_socket = None
        self.is_running = False
        self.image_queue = Queue(maxsize=5)  # 최대 5개 이미지 버퍼

        # 통계
        self.frames_received = 0
        self.last_frame_time = None
        self.fps = 0.0

    def start_server(self):
        """TCP 서버 시작"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.server_socket.settimeout(1.0)
            self.is_running = True

            print(f"✅ 뷰어 서버 시작: {self.host}:{self.port}")
            print(f"   이미지 크기: {self.width}x{self.height}")
            print(f"   예상 데이터 크기: {self.expected_size:,} bytes")
            print()
            print("클라이언트 연결 대기 중...")
            print("(종료: 'q' 키 또는 Ctrl+C)")
            print()

            # 서버 스레드 시작
            server_thread = threading.Thread(target=self._server_loop, daemon=True)
            server_thread.start()

            # 디스플레이 루프 (메인 스레드)
            self._display_loop()

        except Exception as e:
            print(f"❌ 서버 시작 실패: {e}")
        finally:
            self.stop()

    def _server_loop(self):
        """서버 연결 및 데이터 수신 루프"""
        while self.is_running:
            try:
                client_socket, client_address = self.server_socket.accept()
                print(f"✅ 클라이언트 연결됨: {client_address[0]}:{client_address[1]}")

                # 클라이언트 처리
                self._handle_client(client_socket, client_address)

            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"⚠️  서버 루프 에러: {e}")
                break

    def _handle_client(self, client_socket, client_address):
        """클라이언트로부터 데이터 수신 및 크기 검증"""
        try:
            while self.is_running:
                # 첫 번째 청크 수신 (최대 1MB)
                first_chunk = client_socket.recv(1024 * 1024)  # 1MB 버퍼

                if not first_chunk:
                    print(f"⚠️  데이터 수신 실패 또는 연결 종료")
                    break

                received_size = len(first_chunk)
                print(f"📦 수신: {received_size:,} bytes", end='')

                # 크기 검증 및 해상도 자동 감지
                if received_size == self.expected_size:
                    # 정확한 크기 (1280x800)
                    image_data = first_chunk
                    width, height = self.width, self.height
                    print(" ✅")
                elif received_size == 1048576:  # 1024x1024
                    # 1024x1024 해상도
                    print(f" ⚠️  다른 해상도 감지: 1024x1024")
                    image_data = first_chunk
                    width, height = 1024, 1024
                elif received_size > self.expected_size:
                    # 크기가 큼 - 처음 1,024,000만 사용
                    print(f" ⚠️  과다 수신")
                    print(f"   → 처음 {self.expected_size:,} bytes만 사용 (나머지 {received_size - self.expected_size:,} bytes 버림)")
                    image_data = first_chunk[:self.expected_size]
                    width, height = self.width, self.height
                elif received_size < self.expected_size:
                    # 크기가 작음 - 나머지 수신
                    print(f" ⚠️  부분 수신")
                    remaining = self.expected_size - received_size
                    print(f"   → 나머지 {remaining:,} bytes 수신 중...")

                    additional_data = self._recv_exactly(client_socket, remaining, timeout=5.0)

                    if additional_data is None:
                        print(f"   ❌ 나머지 데이터 수신 실패")
                        continue

                    image_data = first_chunk + additional_data
                    width, height = self.width, self.height
                    print(f"   ✅ 전체 수신 완료: {len(image_data):,} bytes")

                # 이미지 처리 및 큐에 추가
                bgr_image = self._process_y8_data(image_data, width, height)

                if bgr_image is not None:
                    # 큐가 가득 차면 오래된 프레임 버림
                    if self.image_queue.full():
                        try:
                            self.image_queue.get_nowait()
                        except Empty:
                            pass

                    self.image_queue.put(bgr_image)
                    self._update_stats()

        except Exception as e:
            print(f"❌ 클라이언트 처리 에러: {e}")
        finally:
            client_socket.close()
            print(f"🔌 클라이언트 연결 종료: {client_address[0]}:{client_address[1]}")

    def _recv_exactly(self, sock, size, timeout=10.0):
        """정확히 size 바이트 수신"""
        sock.settimeout(timeout)
        data = bytearray()

        try:
            while len(data) < size:
                remaining = size - len(data)
                chunk = sock.recv(min(remaining, 65536))

                if not chunk:
                    return None

                data.extend(chunk)

            return bytes(data)

        except socket.timeout:
            print(f"⚠️  수신 타임아웃 ({timeout}초)")
            return None
        except Exception as e:
            print(f"❌ 수신 에러: {e}")
            return None

    def _process_y8_data(self, data: bytes, width: int = None, height: int = None) -> np.ndarray:
        """
        CameraClient.cs와 동일한 Y8 처리
        - Y8 → Grayscale
        - Y축 뒤집기
        - Grayscale → BGR

        Args:
            data: Y8 raw bytes
            width: 이미지 너비 (None이면 self.width 사용)
            height: 이미지 높이 (None이면 self.height 사용)
        """
        try:
            # 해상도 결정
            if width is None:
                width = self.width
            if height is None:
                height = self.height

            # Y8 데이터를 numpy array로 변환
            y8_array = np.frombuffer(data, dtype=np.uint8)
            y8_image = y8_array.reshape((height, width))

            # Y축 뒤집기 (CameraClient.cs 호환)
            y8_flipped = np.flipud(y8_image)

            # Grayscale → BGR
            bgr_image = cv2.cvtColor(y8_flipped, cv2.COLOR_GRAY2BGR)

            return bgr_image

        except Exception as e:
            print(f"❌ 이미지 처리 에러: {e}")
            return None

    def _display_loop(self):
        """이미지 디스플레이 루프 (메인 스레드)"""
        window_name = f"Realtime Y8 Viewer - {self.width}x{self.height}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # 기본 검은 화면
        blank_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(blank_image, "Waiting for connection...", (50, self.height // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(window_name, blank_image)

        print("🖼️  디스플레이 창 열림")
        print()

        while self.is_running:
            try:
                # 큐에서 이미지 가져오기 (100ms 타임아웃)
                image = self.image_queue.get(timeout=0.1)

                # FPS 정보 오버레이
                info_image = image.copy()
                cv2.putText(info_image, f"FPS: {self.fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(info_image, f"Frames: {self.frames_received}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow(window_name, info_image)

                # 키 입력 처리 (1ms 대기)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n'q' 키 감지 - 종료 중...")
                    self.is_running = False
                    break
                elif key == ord('s'):
                    # 스크린샷 저장
                    self._save_screenshot(image)

            except Empty:
                # 큐가 비어있으면 계속 대기
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n'q' 키 감지 - 종료 중...")
                    self.is_running = False
                    break

        cv2.destroyAllWindows()

    def _save_screenshot(self, image: np.ndarray):
        """스크린샷 저장"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"

        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)

        filepath = screenshots_dir / filename
        cv2.imwrite(str(filepath), image)
        print(f"📸 스크린샷 저장: {filepath}")

    def _update_stats(self):
        """통계 업데이트"""
        self.frames_received += 1
        current_time = time.time()

        if self.last_frame_time is not None:
            interval = current_time - self.last_frame_time
            if interval > 0:
                self.fps = 1.0 / interval

        self.last_frame_time = current_time

        # 10프레임마다 통계 출력
        if self.frames_received % 10 == 0:
            print(f"📊 프레임: {self.frames_received}, FPS: {self.fps:.1f}")

    def stop(self):
        """서버 종료"""
        self.is_running = False

        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

        print()
        print("=" * 80)
        print(f"총 수신 프레임: {self.frames_received}")
        print("뷰어 종료됨")
        print("=" * 80)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Real-time Y8 Image Viewer')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=10000, help='Server port (default: 10000)')
    parser.add_argument('--width', type=int, default=1280, help='Image width (default: 1280)')
    parser.add_argument('--height', type=int, default=800, help='Image height (default: 800)')

    args = parser.parse_args()

    print("=" * 80)
    print("  Real-time Y8 Image Viewer")
    print("=" * 80)
    print()

    viewer = RealtimeY8Viewer(
        host=args.host,
        port=args.port,
        width=args.width,
        height=args.height
    )

    try:
        viewer.start_server()
    except KeyboardInterrupt:
        print("\n\nCtrl+C 감지 - 종료 중...")
        viewer.stop()


if __name__ == '__main__':
    main()
