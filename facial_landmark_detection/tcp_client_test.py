"""TCP 클라이언트 테스트 스크립트"""

import socket
import json
import sys
from pathlib import Path
from PIL import Image
from io import BytesIO


class FaceAnalysisClient:
    """얼굴 분석 TCP 클라이언트"""

    def __init__(self, host: str = 'localhost', port: int = 5000):
        """
        클라이언트 초기화

        Args:
            host: 서버 호스트
            port: 서버 포트
        """
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        """서버 연결"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"✅ 서버 연결 성공: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ 연결 실패: {e}")
            return False

    def send_command(self, command: dict) -> bool:
        """
        명령 전송

        Args:
            command: 전송할 명령 딕셔너리

        Returns:
            전송 성공 여부
        """
        try:
            # JSON 직렬화
            json_data = json.dumps(command, ensure_ascii=False)
            data_bytes = json_data.encode('utf-8')

            # 데이터 크기 전송 (4 bytes)
            self.socket.sendall(len(data_bytes).to_bytes(4, byteorder='big'))

            # JSON 데이터 전송
            self.socket.sendall(data_bytes)

            print(f"📤 명령 전송: {command.get('command', 'unknown')}")
            return True

        except Exception as e:
            print(f"❌ 전송 실패: {e}")
            return False

    def receive_response(self) -> dict:
        """
        응답 수신

        Returns:
            수신한 응답 딕셔너리
        """
        try:
            # 데이터 크기 수신 (4 bytes)
            size_data = self.socket.recv(4)
            if not size_data:
                return None

            data_size = int.from_bytes(size_data, byteorder='big')
            print(f"📥 응답 수신 중... ({data_size} bytes)")

            # JSON 데이터 수신
            json_data = b''
            while len(json_data) < data_size:
                chunk = self.socket.recv(min(4096, data_size - len(json_data)))
                if not chunk:
                    break
                json_data += chunk

            # JSON 파싱
            response = json.loads(json_data.decode('utf-8'))
            print(f"✅ 응답 수신 완료")

            return response

        except Exception as e:
            print(f"❌ 수신 실패: {e}")
            return None

    def send_raw_image(self, image_path: str) -> bool:
        """
        raw 이미지 데이터 전송

        Args:
            image_path: 전송할 이미지 파일 경로

        Returns:
            전송 성공 여부
        """
        try:
            # 이미지 로드
            img_path = Path(image_path)
            if not img_path.exists():
                print(f"❌ 이미지 파일이 존재하지 않습니다: {image_path}")
                return False

            # PIL로 이미지 읽기
            pil_image = Image.open(img_path)

            # 이미지를 PNG 포맷의 bytes로 변환
            img_bytes = BytesIO()
            pil_image.save(img_bytes, format='PNG')
            image_data = img_bytes.getvalue()

            # 이미지 데이터 크기 전송 (4 bytes, big-endian)
            image_size = len(image_data)
            self.socket.sendall(image_size.to_bytes(4, byteorder='big'))
            print(f"📤 이미지 크기 전송: {image_size} bytes")

            # 이미지 데이터 전송
            self.socket.sendall(image_data)
            print(f"📤 이미지 데이터 전송 완료: {img_path.name}")

            return True

        except Exception as e:
            print(f"❌ 이미지 전송 실패: {e}")
            return False

    def close(self):
        """연결 종료"""
        if self.socket:
            self.socket.close()
            print("🔌 연결 종료")


def print_analysis_result(result: dict):
    """분석 결과 출력 (단순화된 형식)"""
    print()
    print("=" * 80)
    print("📊 분석 결과")
    print("=" * 80)
    print()
    print(f"👁️  eye_shape: {result.get('eye_shape', 'unknown')}")
    print(f"🎭 face_shape: {result.get('face_shape', 'unknown')}")
    print()
    print("=" * 80)
    print()


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='얼굴 분석 TCP 클라이언트 테스트')
    parser.add_argument(
        '--host',
        default='localhost',
        help='서버 호스트 (기본: localhost)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='서버 포트 (기본: 5000)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("얼굴 분석 TCP 클라이언트")
    print("=" * 80)
    print()

    # 클라이언트 생성 및 연결
    client = FaceAnalysisClient(host=args.host, port=args.port)

    if not client.connect():
        return

    try:
        # 대화형 모드
        print()
        print("명령어:")
        print("  list                  - 이미지 목록 조회")
        print("  analyze <index>       - 이미지 분석 (예: analyze 0)")
        print("  analyze_raw <path>    - raw 이미지 전송 후 분석 (예: analyze_raw data/sample_images/image_1.png)")
        print("  quit                  - 종료")
        print()

        while True:
            try:
                # 사용자 입력
                user_input = input(">>> ").strip()

                if not user_input:
                    continue

                # 명령 파싱
                parts = user_input.split()
                cmd = parts[0].lower()

                # 종료
                if cmd == 'quit':
                    client.send_command({'command': 'quit'})
                    break

                # 이미지 목록
                elif cmd == 'list':
                    client.send_command({'command': 'list'})
                    response = client.receive_response()

                    if response:
                        print()
                        print("📋 이미지 목록:")
                        for img in response.get('images', []):
                            print(f"   [{img['index']}] {img['filename']}")
                        print()

                # 이미지 분석 (파일 기반)
                elif cmd == 'analyze':
                    if len(parts) < 2:
                        print("⚠️  사용법: analyze <index>")
                        continue

                    try:
                        image_index = int(parts[1])
                    except ValueError:
                        print("⚠️  인덱스는 숫자여야 합니다")
                        continue

                    client.send_command({
                        'command': 'analyze',
                        'image_index': image_index
                    })

                    response = client.receive_response()

                    if response:
                        print_analysis_result(response)

                # raw 이미지 분석
                elif cmd == 'analyze_raw':
                    if len(parts) < 2:
                        print("⚠️  사용법: analyze_raw <image_path>")
                        continue

                    image_path = parts[1]

                    # analyze_raw 명령 전송
                    if not client.send_command({'command': 'analyze_raw'}):
                        continue

                    # raw 이미지 데이터 전송
                    if not client.send_raw_image(image_path):
                        continue

                    # 분석 결과 수신
                    response = client.receive_response()

                    if response:
                        if 'error' in response:
                            print(f"❌ 오류: {response['error']}")
                        else:
                            print_analysis_result(response)

                else:
                    print(f"⚠️  알 수 없는 명령: {cmd}")

            except KeyboardInterrupt:
                print("\n⚠️  중단됨")
                break

    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()

    finally:
        client.close()
        print()
        print("=" * 80)
        print("클라이언트 종료")
        print("=" * 80)


if __name__ == "__main__":
    main()
