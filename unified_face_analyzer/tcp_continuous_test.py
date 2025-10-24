#!/usr/bin/env python3
"""
TCP 연속 테스트 클라이언트
5초 간격으로 샘플 이미지를 서버로 보내고 결과를 받아 출력
"""
import socket
import struct
import json
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse


class TCPContinuousTest:
    def __init__(self, host: str = "172.30.1.45", port: int = 10000):
        """
        Args:
            host: 서버 IP (기본값: 172.30.1.45 - MacBook IP)
            port: 서버 포트 (기본값: 10000)
        """
        self.host = host
        self.port = port
        self.sample_image = None

    def load_sample_image(self, image_path: str):
        """샘플 이미지 로드"""
        self.sample_image = cv2.imread(image_path)
        if self.sample_image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        print(f"📸 샘플 이미지 로드: {image_path}")
        print(f"   해상도: {self.sample_image.shape}")

    def send_image(self, sock: socket.socket) -> bool:
        """이미지를 서버로 전송"""
        try:
            # JPEG 인코딩
            _, encoded = cv2.imencode('.jpg', self.sample_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_bytes = encoded.tobytes()

            # 크기 전송 (4 bytes, little-endian)
            size_header = struct.pack('<I', len(image_bytes))
            sock.sendall(size_header)

            # 이미지 데이터 전송
            sock.sendall(image_bytes)

            return True
        except Exception as e:
            print(f"❌ 이미지 전송 실패: {e}")
            return False

    def receive_json(self, sock: socket.socket) -> dict:
        """서버로부터 JSON 결과 수신"""
        try:
            # JSON 크기 수신 (4 bytes)
            size_data = sock.recv(4)
            if len(size_data) < 4:
                return None

            json_size = struct.unpack('<I', size_data)[0]

            # JSON 데이터 수신
            json_data = b''
            remaining = json_size
            while remaining > 0:
                chunk = sock.recv(min(remaining, 65536))
                if not chunk:
                    return None
                json_data += chunk
                remaining -= len(chunk)

            # JSON 파싱
            result = json.loads(json_data.decode('utf-8'))
            return result
        except Exception as e:
            print(f"❌ JSON 수신 실패: {e}")
            return None

    def display_compact_result(self, result: dict, iteration: int):
        """결과를 간단하게 출력"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        # TCP_SPEC 형식 확인
        if 'hairstyle' in result and isinstance(result['hairstyle'], int):
            print(f"[{timestamp}] #{iteration:03d} → "
                  f"Hair:{result['hairstyle_name']:12s} | "
                  f"Face:{result.get('face_shape_name', 'Unknown'):8s} | "
                  f"Eye:{result.get('eye_shape_name', 'Unknown'):10s}")
        else:
            print(f"[{timestamp}] #{iteration:03d} → Analysis completed (raw format)")

    def run_continuous_test(self, interval: float = 5.0, max_iterations: int = None):
        """
        연속 테스트 실행

        Args:
            interval: 전송 간격 (초)
            max_iterations: 최대 반복 횟수 (None이면 무한 반복)
        """
        print(f"\n{'='*80}")
        print(f"  TCP 연속 테스트 시작")
        print(f"{'='*80}")
        print(f"서버: {self.host}:{self.port}")
        print(f"전송 간격: {interval}초")
        print(f"최대 반복: {'무한' if max_iterations is None else max_iterations}회")
        print(f"{'='*80}\n")
        print("📡 연결 중...\n")

        iteration = 0

        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1

                try:
                    # 매번 새로운 연결 생성 (언리얼 방식)
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        sock.settimeout(10.0)
                        sock.connect((self.host, self.port))

                        # 이미지 전송
                        if not self.send_image(sock):
                            continue

                        # 결과 수신
                        result = self.receive_json(sock)
                        if result:
                            self.display_compact_result(result, iteration)
                        else:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] #{iteration:03d} → ❌ 결과 수신 실패")

                except ConnectionRefusedError:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] #{iteration:03d} → ❌ 서버 연결 거부")
                except socket.timeout:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] #{iteration:03d} → ⏱️  타임아웃")
                except Exception as e:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] #{iteration:03d} → ❌ 오류: {e}")

                # 다음 전송까지 대기
                if max_iterations is None or iteration < max_iterations:
                    time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n\n{'='*80}")
            print(f"  테스트 중단 (Ctrl+C)")
            print(f"{'='*80}")
            print(f"총 전송 횟수: {iteration}회\n")


def main():
    parser = argparse.ArgumentParser(description='TCP 연속 테스트 클라이언트')
    parser.add_argument('image', help='샘플 이미지 경로')
    parser.add_argument('--host', default='172.30.1.45', help='서버 IP (기본값: 172.30.1.45)')
    parser.add_argument('--port', type=int, default=10000, help='서버 포트 (기본값: 10000)')
    parser.add_argument('--interval', type=float, default=5.0, help='전송 간격 (초, 기본값: 5.0)')
    parser.add_argument('--count', type=int, help='최대 전송 횟수 (기본값: 무한)')
    parser.add_argument('--localhost', action='store_true', help='localhost (127.0.0.1) 사용')

    args = parser.parse_args()

    # localhost 옵션 처리
    host = '127.0.0.1' if args.localhost else args.host

    # 테스트 클라이언트 생성
    client = TCPContinuousTest(host=host, port=args.port)

    # 샘플 이미지 로드
    client.load_sample_image(args.image)

    # 연속 테스트 실행
    client.run_continuous_test(interval=args.interval, max_iterations=args.count)


if __name__ == "__main__":
    main()
