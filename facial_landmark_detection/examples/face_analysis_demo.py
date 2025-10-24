"""
얼굴 특징 분석 데모
- 눈 형태 분류 (눈꼬리 올라감/내려감/기본)
- 얼굴형 분류 (계란형/둥근형/사각형/하트형/긴형)
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import DetectionConfig
from src.core.face_detector import FaceDetector
from src.core.landmark_extractor import LandmarkExtractor
from src.processing.geometry import GeometryCalculator
from src.processing.face_analyzer import FaceAnalyzer


def analyze_face_from_image(image_path: str):
    """
    이미지에서 얼굴 특징 분석

    Args:
        image_path: 이미지 파일 경로
    """
    print("=" * 70)
    print(f"얼굴 분석: {Path(image_path).name}")
    print("=" * 70)
    print()

    # 1. 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지를 불러올 수 없습니다: {image_path}")
        return

    print(f"✅ 이미지 로드 완료: {image.shape[1]}x{image.shape[0]}")
    print()

    # 2. 얼굴 검출
    config = DetectionConfig(static_image_mode=True)
    detector = FaceDetector(config)
    extractor = LandmarkExtractor()
    geometry_calc = GeometryCalculator()
    analyzer = FaceAnalyzer()

    result = detector.detect(image)

    if not result.success:
        print("❌ 얼굴이 검출되지 않았습니다.")
        detector.release()
        return

    print(f"✅ 얼굴 검출 성공!")
    print(f"   - Landmarks: {len(result.landmarks)}개")
    print(f"   - Confidence: {result.confidence:.2f}")
    print(f"   - 처리 시간: {result.processing_time:.1f}ms")
    print()

    # 3. 얼굴 기하학 정보 계산
    face_geometry = geometry_calc.get_face_geometry(result.landmarks)
    print("📐 얼굴 각도:")
    print(f"   - Pitch (상하): {face_geometry.pitch:.1f}°")
    print(f"   - Yaw (좌우): {face_geometry.yaw:.1f}°")
    print(f"   - Roll (기울기): {face_geometry.roll:.1f}°")
    print()

    # 4. 눈 형태 분석 (Roll, Yaw 보정 적용)
    eye_analysis = analyzer.analyze_eye_shape(
        result.landmarks,
        roll_angle=face_geometry.roll,
        yaw_angle=face_geometry.yaw
    )

    print("👁️  눈 형태 분석:")
    print(f"   - 왼쪽 눈: {eye_analysis.left_eye_shape.value} (기울기: {eye_analysis.left_eye_angle:.3f})")
    print(f"   - 오른쪽 눈: {eye_analysis.right_eye_shape.value} (기울기: {eye_analysis.right_eye_angle:.3f})")
    print(f"   - 평균 기울기: {eye_analysis.average_eye_angle:.3f}")
    print(f"   - 신뢰도: {eye_analysis.confidence:.2%}")

    # 눈 형태 설명
    eye_shape_desc = {
        'upturned': '눈꼬리가 올라간 형태 (상승형)',
        'downturned': '눈꼬리가 내려간 형태 (하강형)',
        'neutral': '눈꼬리가 평행한 기본 형태'
    }
    print(f"   → {eye_shape_desc[eye_analysis.left_eye_shape.value]}")
    print()

    # 5. 얼굴형 분석
    face_shape_analysis = analyzer.analyze_face_shape(result.landmarks)

    print("🎭 얼굴형 분석:")
    print(f"   - 얼굴형: {face_shape_analysis.face_shape.value.upper()}")
    print(f"   - 종횡비 (세로/가로): {face_shape_analysis.aspect_ratio:.3f}")
    print(f"   - 얼굴 너비: {face_shape_analysis.face_width:.1f}px")
    print(f"   - 얼굴 높이: {face_shape_analysis.face_height:.1f}px")
    print()
    print("   📏 너비 측정:")
    print(f"   - 이마 너비: {face_shape_analysis.forehead_width:.1f}px")
    print(f"   - 광대 너비: {face_shape_analysis.cheekbone_width:.1f}px")
    print(f"   - 턱선 너비: {face_shape_analysis.jawline_width:.1f}px")
    print(f"   - 신뢰도: {face_shape_analysis.confidence:.2%}")

    # 얼굴형 설명
    face_shape_desc = {
        'oval': '계란형 (긴 타원형, 이상적인 얼굴형)',
        'round': '둥근형 (원형에 가까운, 부드러운 인상)',
        'square': '사각형 (각진 턱선, 남성적인 인상)',
        'heart': '하트형 (넓은 이마, 뾰족한 턱)',
        'long': '긴형 (매우 긴 얼굴, 날씬한 인상)'
    }
    print(f"   → {face_shape_desc[face_shape_analysis.face_shape.value]}")
    print()

    # 6. JSON 출력
    print("📄 JSON 출력:")
    detailed_analysis = analyzer.get_detailed_analysis(
        result.landmarks,
        roll_angle=face_geometry.roll
    )
    import json
    print(json.dumps(detailed_analysis.to_dict(), indent=2, ensure_ascii=False))
    print()

    # 7. 정리
    detector.release()

    print("=" * 70)
    print("✅ 분석 완료!")
    print("=" * 70)
    print()


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='얼굴 특징 분석 데모')
    parser.add_argument('image', help='분석할 이미지 파일 경로')

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"❌ 파일이 존재하지 않습니다: {args.image}")
        print()
        print("사용법:")
        print("  python examples/face_analysis_demo.py <image_path>")
        print()
        print("예시:")
        print("  python examples/face_analysis_demo.py data/sample_images/face1.jpg")
        sys.exit(1)

    analyze_face_from_image(args.image)


if __name__ == "__main__":
    main()
