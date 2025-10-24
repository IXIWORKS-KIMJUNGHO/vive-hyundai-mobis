#!/usr/bin/env python3
"""
TCP 클라이언트 테스트
Unreal Engine을 시뮬레이션하여 이미지 전송 및 결과 수신
"""

import socket
import struct
import json
import cv2
import argparse
from pathlib import Path


class TCPClient:
    """TCP 클라이언트 (Unreal Engine 시뮬레이션)"""

    def __init__(self, host: str = 'localhost', port: int = 5000):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        """서버에 연결"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        print(f"✅ 서버 연결 성공: {self.host}:{self.port}\n")

    def send_image(self, image_path: str):
        """
        이미지 전송

        Protocol:
        1. 4 bytes: 이미지 크기 (uint32, little-endian)
        2. N bytes: 이미지 바이너리 데이터 (JPEG)
        """
        # 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # JPEG 인코딩
        _, encoded = cv2.imencode('.jpg', image)
        image_bytes = encoded.tobytes()

        print(f"📤 이미지 전송 중...")
        print(f"   경로: {image_path}")
        print(f"   크기: {len(image_bytes)} bytes")
        print(f"   해상도: {image.shape}")

        # 크기 전송 (4 bytes)
        size_bytes = struct.pack('<I', len(image_bytes))
        self.socket.sendall(size_bytes)

        # 이미지 데이터 전송
        self.socket.sendall(image_bytes)

        print(f"✅ 전송 완료\n")

    def receive_json(self):
        """
        JSON 결과 수신

        Protocol:
        1. 4 bytes: JSON 크기 (uint32, little-endian)
        2. N bytes: JSON 문자열 (UTF-8)
        """
        print(f"📥 JSON 결과 수신 중...")

        # 크기 수신 (4 bytes)
        size_data = self.socket.recv(4)
        if len(size_data) < 4:
            raise ValueError("Failed to receive JSON size")

        json_size = struct.unpack('<I', size_data)[0]
        print(f"   JSON 크기: {json_size} bytes")

        # JSON 데이터 수신
        json_data = b''
        remaining = json_size

        while remaining > 0:
            chunk = self.socket.recv(min(remaining, 65536))
            if not chunk:
                raise ValueError("Connection closed while receiving JSON")
            json_data += chunk
            remaining -= len(chunk)

        # JSON 파싱
        json_str = json_data.decode('utf-8')
        result = json.loads(json_str)

        print(f"✅ 수신 완료\n")

        return result

    def close(self):
        """연결 종료"""
        if self.socket:
            self.socket.close()
            print("🔌 연결 종료")

    def display_result(self, result: dict):
        """결과 표시 (TCP_SPEC.md 형식)"""
        print("=" * 80)
        print("  분석 결과 (TCP_SPEC 형식)")
        print("=" * 80)

        # TCP_SPEC 형식 확인
        if 'hairstyle' in result and isinstance(result['hairstyle'], int):
            # TCP_SPEC 형식
            print("✅ 분석 성공\n")

            print("💇 Hairstyle:")
            print(f"   ID: {result['hairstyle']}")
            print(f"   Name: {result['hairstyle_name']}")

            print("\n👤 Gender:")
            print(f"   ID: {result['gender']}")
            print(f"   Name: {result['gender_name']}")
            print(f"   Confidence: {result['gender_confidence']:.4f}")

            print("\n👓 Glasses:")
            print(f"   Has Glasses: {'Yes' if result['has_glasses'] else 'No'}")
            print(f"   Confidence: {result['glasses_confidence']:.4f}")

            print("\n🧔 Beard:")
            print(f"   Has Beard: {'Yes' if result['has_beard'] else 'No'}")
            print(f"   Confidence: {result['beard_confidence']:.4f}")

            print("\n😊 Face Shape:")
            print(f"   ID: {result['face_shape']}")
            print(f"   Name: {result['face_shape_name']}")

            print("\n👁️  Eye Shape:")
            print(f"   ID: {result['eye_shape']}")
            print(f"   Name: {result['eye_shape_name']}")

            print("\n⚙️  Metadata:")
            print(f"   Timestamp: {result['timestamp']}")
            print(f"   Image Path: {result['image_path']}")

        else:
            # 구형식 (호환성)
            if not result.get('success'):
                print("❌ 분석 실패")
                if 'error' in result:
                    print(f"   오류: {result['error']}")
                return

            print("✅ 분석 성공\n")

            # MediaPipe 결과
            if 'mediapipe' in result:
                mp = result['mediapipe']
                print("📍 MediaPipe:")
                print(f"   Landmarks: {mp.get('landmarks_count', 0)}")

                if 'face_geometry' in mp:
                    geom = mp['face_geometry']
                    print(f"   Geometry: pitch={geom['pitch']:.1f}° yaw={geom['yaw']:.1f}° roll={geom['roll']:.1f}°")

                if 'eye_analysis' in mp:
                    eye = mp['eye_analysis']
                    print(f"   👁️  Eye Shape: {eye['overall_eye_shape']}")

                if 'face_shape_analysis' in mp:
                    face = mp['face_shape_analysis']
                    print(f"   😊 Face Shape: {face['face_shape']}")

                print()

            # Hairstyle 결과
            if 'hairstyle' in result:
                hs = result['hairstyle']
                print("💇 Hairstyle:")
                print(f"   Classification: {hs.get('classification', 'Unknown')}")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='TCP Client Test for Unified Face Analyzer')
    parser.add_argument('image', help='Image file to analyze')
    parser.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=5000, help='Server port (default: 5000)')

    args = parser.parse_args()

    # 이미지 파일 확인
    if not Path(args.image).exists():
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {args.image}")
        return 1

    client = TCPClient(host=args.host, port=args.port)

    try:
        # 서버 연결
        client.connect()

        # 이미지 전송
        client.send_image(args.image)

        # 결과 수신
        result = client.receive_json()

        # 결과 표시
        client.display_result(result)

        return 0

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        client.close()


if __name__ == '__main__':
    exit(main())
