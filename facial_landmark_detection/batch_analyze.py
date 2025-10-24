"""배치 이미지 분석 스크립트"""

import sys
import cv2
import json
from pathlib import Path
from typing import List, Dict, Any

from src.config.settings import DetectionConfig
from src.core.face_detector import FaceDetector
from src.core.landmark_extractor import LandmarkExtractor
from src.processing.geometry import GeometryCalculator
from src.processing.face_analyzer import FaceAnalyzer


def analyze_images_in_directory(directory: str) -> List[Dict[str, Any]]:
    """
    디렉토리 내 모든 이미지 분석

    Args:
        directory: 이미지 디렉토리 경로

    Returns:
        분석 결과 리스트
    """
    # 초기화
    config = DetectionConfig(static_image_mode=True)
    detector = FaceDetector(config)
    extractor = LandmarkExtractor()
    geometry_calc = GeometryCalculator()
    analyzer = FaceAnalyzer()

    # 이미지 파일 찾기
    image_dir = Path(directory)
    image_files = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg'))

    results = []

    print("=" * 80)
    print(f"배치 얼굴 분석 - {len(image_files)}개 이미지")
    print("=" * 80)
    print()

    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] 분석 중: {image_path.name}")
        print("-" * 80)

        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"   ❌ 이미지 로드 실패\n")
            continue

        # 얼굴 검출
        detection_result = detector.detect(image)

        if not detection_result.success:
            print(f"   ❌ 얼굴 검출 실패\n")
            results.append({
                'filename': image_path.name,
                'success': False,
                'error': '얼굴이 검출되지 않음'
            })
            continue

        # 얼굴 기하학 정보
        face_geometry = geometry_calc.get_face_geometry(detection_result.landmarks)

        # 상세 분석 (Roll, Yaw 보정 적용)
        detailed_analysis = analyzer.get_detailed_analysis(
            detection_result.landmarks,
            roll_angle=face_geometry.roll,
            yaw_angle=face_geometry.yaw
        )

        # 결과 출력
        eye_analysis = detailed_analysis.eye_analysis
        face_shape_analysis = detailed_analysis.face_shape_analysis

        print(f"   ✅ 얼굴 검출 성공 (Confidence: {detection_result.confidence:.2f})")
        print(f"   📐 얼굴 각도: Pitch={face_geometry.pitch:.1f}°, Yaw={face_geometry.yaw:.1f}°, Roll={face_geometry.roll:.1f}°")
        print()
        print(f"   👁️  눈 형태:")
        print(f"      - 왼쪽: {eye_analysis.left_eye_shape.value.upper()} (기울기: {eye_analysis.left_eye_angle:.3f})")
        print(f"      - 오른쪽: {eye_analysis.right_eye_shape.value.upper()} (기울기: {eye_analysis.right_eye_angle:.3f})")
        print(f"      - 평균 기울기: {eye_analysis.average_eye_angle:.3f}")
        print(f"      ⭐ 전체 눈 형태: {eye_analysis.overall_eye_shape.value.upper()}")
        print()
        print(f"   🎭 얼굴형: {face_shape_analysis.face_shape.value.upper()}")
        print(f"      - 종횡비: {face_shape_analysis.aspect_ratio:.3f}")
        print(f"      - 크기: {face_shape_analysis.face_width:.0f}px × {face_shape_analysis.face_height:.0f}px")
        print(f"      - 이마: {face_shape_analysis.forehead_width:.0f}px")
        print(f"      - 광대: {face_shape_analysis.cheekbone_width:.0f}px")
        print(f"      - 턱선: {face_shape_analysis.jawline_width:.0f}px")
        print()

        # 결과 저장
        result_data = {
            'filename': image_path.name,
            'success': True,
            'image_size': {
                'width': image.shape[1],
                'height': image.shape[0]
            },
            'detection': {
                'confidence': detection_result.confidence,
                'processing_time_ms': detection_result.processing_time,
                'num_landmarks': len(detection_result.landmarks)
            },
            'face_geometry': {
                'pitch': round(face_geometry.pitch, 2),
                'yaw': round(face_geometry.yaw, 2),
                'roll': round(face_geometry.roll, 2)
            },
            'analysis': detailed_analysis.to_dict()
        }

        results.append(result_data)

    detector.release()

    return results


def print_summary(results: List[Dict[str, Any]]):
    """결과 요약 출력"""
    print("=" * 80)
    print("📊 분석 결과 요약")
    print("=" * 80)
    print()

    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    print(f"✅ 성공: {len(successful)}개")
    print(f"❌ 실패: {len(failed)}개")
    print()

    if successful:
        # 눈 형태 통계 (전체 눈 형태 기준)
        print("👁️  눈 형태 분포 (전체 눈 형태):")
        eye_shapes = {}
        for r in successful:
            shape = r['analysis']['eye_analysis']['overall_eye_shape']
            eye_shapes[shape] = eye_shapes.get(shape, 0) + 1

        for shape, count in sorted(eye_shapes.items()):
            print(f"   - {shape.upper()}: {count}개")
        print()

        # 얼굴형 통계
        print("🎭 얼굴형 분포:")
        face_shapes = {}
        for r in successful:
            shape = r['analysis']['face_shape_analysis']['face_shape']
            face_shapes[shape] = face_shapes.get(shape, 0) + 1

        for shape, count in sorted(face_shapes.items()):
            print(f"   - {shape.upper()}: {count}개")
        print()

        # 평균 종횡비
        avg_aspect_ratio = sum(
            r['analysis']['face_shape_analysis']['aspect_ratio']
            for r in successful
        ) / len(successful)
        print(f"📏 평균 종횡비 (세로/가로): {avg_aspect_ratio:.3f}")
        print()


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='배치 얼굴 분석')
    parser.add_argument(
        '--directory',
        default='data/sample_images',
        help='분석할 이미지 디렉토리 (기본: data/sample_images)'
    )
    parser.add_argument(
        '--output',
        default='analysis_results.json',
        help='결과 저장 파일 (기본: analysis_results.json)'
    )

    args = parser.parse_args()

    # 분석 실행
    results = analyze_images_in_directory(args.directory)

    # 요약 출력
    print_summary(results)

    # JSON 저장
    output_path = Path(args.output)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"💾 결과 저장: {output_path}")
    print()


if __name__ == "__main__":
    main()
