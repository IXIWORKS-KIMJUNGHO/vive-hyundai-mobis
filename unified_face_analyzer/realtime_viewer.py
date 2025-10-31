#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realtime Y8 Viewer - Simple BGR Image Display Client
Port 7001에서 BGR 이미지를 받아 실시간 표시

Architecture:
- ViewerBroadcaster (Port 7001) → BGR bytes 전송
- realtime_viewer.py → BGR bytes 수신 → OpenCV 디스플레이
"""

import sys
import io

# Windows 콘솔 UTF-8 설정
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except:
        pass

import socket
import numpy as np
import cv2
import time
from datetime import datetime


class RealtimeBGRViewer:
    """
    실시간 BGR 이미지 뷰어

    Features:
    - Port 7001에서 BGR bytes 수신
    - OpenCV 창으로 실시간 표시
    - FPS 및 성능 통계 표시
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 7001, width: int = 1280, height: int = 800):
        """
        Args:
            host: ViewerBroadcaster 호스트
            port: ViewerBroadcaster 포트
            width: 이미지 너비
            height: 이미지 높이
        """
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.expected_size = width * height * 3  # BGR = 3 bytes per pixel

        # 성능 측정
        self.frames_received = 0
        self.start_time = None
        self.last_fps_time = None
        self.fps = 0.0
        self.frame_count_in_interval = 0  # 1초 구간 내 프레임 수

        # OpenCV 창 설정
        self.window_name = "Realtime BGR Viewer"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 600)  # 1280x800의 75% 크기

    def connect(self) -> socket.socket:
        """ViewerBroadcaster에 연결"""
        print("=" * 80)
        print(f"  Realtime BGR Viewer")
        print("=" * 80)
        print(f"Connecting to: {self.host}:{self.port}")
        print(f"Expected image size: {self.width}x{self.height} BGR ({self.expected_size:,} bytes)")
        print("=" * 80)
        print()

        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self.host, self.port))

        print(f"✅ Connected to ViewerBroadcaster: {self.host}:{self.port}")
        print()

        return client_socket

    def receive_bgr_frame(self, client_socket: socket.socket) -> np.ndarray:
        """
        BGR 이미지 프레임 수신

        Args:
            client_socket: 연결된 소켓

        Returns:
            BGR numpy array (height, width, 3)
        """
        # BGR bytes 수신 (정확한 크기)
        data = b''
        while len(data) < self.expected_size:
            chunk = client_socket.recv(min(self.expected_size - len(data), 65536))
            if not chunk:
                raise ConnectionError("Connection closed while receiving frame")
            data += chunk

        # BGR bytes → numpy array
        bgr_array = np.frombuffer(data, dtype=np.uint8)
        bgr_image = bgr_array.reshape((self.height, self.width, 3))

        return bgr_image

    def update_fps(self):
        """FPS 계산 및 업데이트"""
        current_time = time.time()

        if self.start_time is None:
            self.start_time = current_time
            self.last_fps_time = current_time

        self.frames_received += 1
        self.frame_count_in_interval += 1

        # 1초마다 FPS 업데이트
        if current_time - self.last_fps_time >= 1.0:
            elapsed = current_time - self.last_fps_time
            self.fps = self.frame_count_in_interval / elapsed
            self.last_fps_time = current_time
            self.frame_count_in_interval = 0  # 구간 카운터 리셋

    def display_frame(self, bgr_image: np.ndarray):
        """
        BGR 이미지를 OpenCV 창에 표시

        Args:
            bgr_image: BGR numpy array
        """
        # FPS 정보를 이미지에 오버레이
        display_image = bgr_image.copy()

        # FPS 텍스트
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(display_image, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # 프레임 카운트
        frame_text = f"Frames: {self.frames_received}"
        cv2.putText(display_image, frame_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 이미지 표시
        cv2.imshow(self.window_name, display_image)

    def run(self):
        """메인 실행 루프"""
        client_socket = None

        try:
            client_socket = self.connect()

            print("🎬 Receiving BGR frames... (Press 'q' to quit)")
            print()

            while True:
                # BGR 프레임 수신
                bgr_image = self.receive_bgr_frame(client_socket)

                # FPS 업데이트
                self.update_fps()

                # 디스플레이
                self.display_frame(bgr_image)

                # 콘솔 출력 (1초마다)
                if self.frames_received % int(max(1, self.fps)) == 0:
                    print(f"\r📺 Frames: {self.frames_received} | FPS: {self.fps:.1f}",
                          end='', flush=True)

                # 'q' 키로 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n\n⏹️  Quit requested")
                    break

        except ConnectionRefusedError:
            print(f"❌ Cannot connect to {self.host}:{self.port}")
            print("   Make sure ViewerBroadcaster is running!")
        except ConnectionError as e:
            print(f"\n❌ Connection error: {e}")
        except KeyboardInterrupt:
            print("\n\n⚠️  Keyboard interrupt")
        finally:
            if client_socket:
                client_socket.close()
            cv2.destroyAllWindows()

            # 최종 통계
            if self.frames_received > 0:
                print("\n")
                print("=" * 80)
                print("  Session Statistics")
                print("=" * 80)
                print(f"Total frames received: {self.frames_received}")
                if self.start_time:
                    total_time = time.time() - self.start_time
                    avg_fps = self.frames_received / total_time
                    print(f"Average FPS: {avg_fps:.2f}")
                    print(f"Total time: {total_time:.1f}s")
                print("=" * 80)


def main():
    """메인 엔트리 포인트"""
    viewer = RealtimeBGRViewer(
        host='127.0.0.1',
        port=7001,
        width=1280,
        height=800
    )
    viewer.run()


if __name__ == "__main__":
    main()
