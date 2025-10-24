"""TCP 서버 데모 - 얼굴 분석 결과 실시간 전송"""

import cv2
import logging
from pathlib import Path

from src.config.settings import DetectionConfig
from src.core.face_detector import FaceDetector
from src.processing.geometry import GeometryCalculator
from src.processing.face_analyzer import FaceAnalyzer
from src.network.tcp_server import FaceAnalysisTCPServer


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    """
    TCP 서버 데모 메인 함수

    동작 방식:
    1. TCP 서버 시작 (포트 5000)
    2. 클라이언트 연결 대기
    3. 클라이언트로부터 분석 요청 수신
    4. 얼굴 분석 수행
    5. 결과를 JSON으로 전송
    """
    import argparse

    parser = argparse.ArgumentParser(description='얼굴 분석 TCP 서버')
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='서버 호스트 (기본: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='서버 포트 (기본: 5000)'
    )
    parser.add_argument(
        '--image-dir',
        default='data/sample_images',
        help='이미지 디렉토리 (기본: data/sample_images)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("얼굴 분석 TCP 서버")
    print("=" * 80)
    print()

    # 얼굴 분석 초기화
    print("🔧 얼굴 분석 모듈 초기화 중...")
    config = DetectionConfig(static_image_mode=True)
    detector = FaceDetector(config)
    geometry_calc = GeometryCalculator()
    analyzer = FaceAnalyzer()
    print("✅ 초기화 완료")
    print()

    # 이미지 디렉토리 확인
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"❌ 이미지 디렉토리가 존재하지 않습니다: {image_dir}")
        return

    image_files = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg'))
    print(f"📂 이미지 디렉토리: {image_dir}")
    print(f"📷 사용 가능한 이미지: {len(image_files)}개")
    print()

    # TCP 서버 시작
    try:
        with FaceAnalysisTCPServer(host=args.host, port=args.port) as server:
            print("=" * 80)
            print("명령 프로토콜:")
            print("  - {'command': 'analyze', 'image_index': 0}  # 0번째 이미지 분석")
            print("  - {'command': 'analyze_raw'}                # raw 이미지 데이터 전송 후 분석")
            print("  - {'command': 'list'}                       # 이미지 목록 조회")
            print("  - {'command': 'quit'}                       # 연결 종료")
            print("=" * 80)
            print()

            while server.is_running:
                # 클라이언트 연결 대기
                client_socket, client_address = server.accept_connection()

                try:
                    # 클라이언트 요청 처리 루프
                    while True:
                        # 명령 수신
                        command = server.receive_command(client_socket)

                        if command is None:
                            print("⚠️  클라이언트 연결 끊김")
                            break

                        cmd_type = command.get('command', '')

                        # 이미지 목록 조회
                        if cmd_type == 'list':
                            image_list = {
                                'command': 'list_response',
                                'images': [
                                    {
                                        'index': i,
                                        'filename': img.name,
                                        'path': str(img)
                                    }
                                    for i, img in enumerate(image_files)
                                ]
                            }

                            # 응답 전송
                            import json
                            response = json.dumps(image_list, ensure_ascii=False)
                            client_socket.sendall(len(response).to_bytes(4, byteorder='big'))
                            client_socket.sendall(response.encode('utf-8'))
                            print(f"📋 이미지 목록 전송: {len(image_files)}개")

                        # 얼굴 분석 요청 (파일 기반)
                        elif cmd_type == 'analyze':
                            image_index = command.get('image_index', 0)

                            if image_index < 0 or image_index >= len(image_files):
                                print(f"❌ 잘못된 이미지 인덱스: {image_index}")
                                continue

                            image_path = image_files[image_index]
                            print(f"🔍 분석 중: {image_path.name} (인덱스: {image_index})")

                            # 이미지 로드
                            image = cv2.imread(str(image_path))
                            if image is None:
                                print(f"❌ 이미지 로드 실패")
                                continue

                            # 얼굴 검출
                            detection_result = detector.detect(image)

                            if not detection_result.success:
                                print(f"❌ 얼굴 검출 실패")
                                continue

                            # 얼굴 기하학 정보
                            face_geometry = geometry_calc.get_face_geometry(detection_result.landmarks)

                            # 상세 분석
                            detailed_analysis = analyzer.get_detailed_analysis(
                                detection_result.landmarks,
                                roll_angle=face_geometry.roll,
                                yaw_angle=face_geometry.yaw
                            )

                            # 결과 전송 (단순화된 형식: eye_shape, face_shape만)
                            success = server.send_analysis_result(
                                client_socket,
                                detailed_analysis
                            )

                            if success:
                                print(f"✅ 분석 완료 및 전송: {image_path.name}")
                                print(f"   - eye_shape: {detailed_analysis.eye_analysis.overall_eye_shape.value}")
                                print(f"   - face_shape: {detailed_analysis.face_shape_analysis.face_shape.value}")
                            else:
                                print(f"❌ 전송 실패")

                        # 얼굴 분석 요청 (raw 이미지 데이터)
                        elif cmd_type == 'analyze_raw':
                            print("📥 raw 이미지 데이터 수신 대기...")

                            # raw 이미지 데이터 수신
                            image = server.receive_raw_image(client_socket)

                            if image is None:
                                print(f"❌ 이미지 수신 실패")
                                # 에러 응답 전송
                                error_response = {
                                    "error": "이미지 수신 실패"
                                }
                                import json
                                error_json = json.dumps(error_response, ensure_ascii=False)
                                client_socket.sendall(len(error_json).to_bytes(4, byteorder='big'))
                                client_socket.sendall(error_json.encode('utf-8'))
                                continue

                            print(f"🔍 얼굴 분석 중... (이미지 크기: {image.shape})")

                            # 얼굴 검출
                            detection_result = detector.detect(image)

                            if not detection_result.success:
                                print(f"❌ 얼굴 검출 실패")
                                # 에러 응답 전송
                                error_response = {
                                    "error": "얼굴 검출 실패"
                                }
                                import json
                                error_json = json.dumps(error_response, ensure_ascii=False)
                                client_socket.sendall(len(error_json).to_bytes(4, byteorder='big'))
                                client_socket.sendall(error_json.encode('utf-8'))
                                continue

                            # 얼굴 기하학 정보
                            face_geometry = geometry_calc.get_face_geometry(detection_result.landmarks)

                            # 상세 분석
                            detailed_analysis = analyzer.get_detailed_analysis(
                                detection_result.landmarks,
                                roll_angle=face_geometry.roll,
                                yaw_angle=face_geometry.yaw
                            )

                            # 결과 전송 (단순화된 형식: eye_shape, face_shape만)
                            success = server.send_analysis_result(
                                client_socket,
                                detailed_analysis
                            )

                            if success:
                                print(f"✅ raw 이미지 분석 완료 및 전송")
                                print(f"   - eye_shape: {detailed_analysis.eye_analysis.overall_eye_shape.value}")
                                print(f"   - face_shape: {detailed_analysis.face_shape_analysis.face_shape.value}")
                            else:
                                print(f"❌ 전송 실패")

                        # 종료 요청
                        elif cmd_type == 'quit':
                            print("👋 클라이언트 종료 요청")
                            break

                        else:
                            print(f"⚠️  알 수 없는 명령: {cmd_type}")

                except KeyboardInterrupt:
                    print("\n⚠️  사용자 중단")
                    break

                except Exception as e:
                    print(f"❌ 오류 발생: {e}")

                finally:
                    # 클라이언트 연결 종료
                    server.close_client(client_socket, client_address)

    except KeyboardInterrupt:
        print("\n⚠️  서버 중단 (Ctrl+C)")

    except Exception as e:
        print(f"❌ 서버 오류: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 정리
        detector.release()
        print()
        print("=" * 80)
        print("서버 종료")
        print("=" * 80)


if __name__ == "__main__":
    main()
