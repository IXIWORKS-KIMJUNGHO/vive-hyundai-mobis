"""
Basic usage example for Facial Landmark Detection System
Layer 1 & 2 통합 사용 예제

Note: MediaPipe requires Python 3.9-3.11
"""

import cv2
import numpy as np
from src.config.settings import DetectionConfig
from src.core.face_detector import FaceDetector
from src.core.landmark_extractor import LandmarkExtractor
from src.processing.frame_processor import FrameProcessor
from src.processing.geometry import GeometryCalculator


def example_basic_detection():
    """기본 얼굴 검출 예제"""
    print("=" * 60)
    print("Example 1: Basic Face Detection")
    print("=" * 60)

    # 설정 생성
    config = DetectionConfig(
        model_complexity=1,
        min_detection_confidence=0.5,
        static_image_mode=True,
        enable_face_geometry=True
    )

    # 검출기 초기화
    detector = FaceDetector(config)

    # 이미지 로드
    image = cv2.imread('data/sample_images/face.jpg')

    if image is None:
        print("⚠️  Sample image not found. Please add face.jpg to data/sample_images/")
        return

    # 얼굴 검출
    result = detector.detect(image)

    if result.success:
        print(f"✅ Face detected!")
        print(f"📊 Landmarks: {len(result.landmarks)} points")
        print(f"⏱️  Processing time: {result.processing_time:.2f}ms")
        print(f"📦 Bounding box: {result.bounding_box}")

        if result.face_geometry:
            geo = result.face_geometry
            print(f"\n📐 Face Geometry:")
            print(f"   Pitch: {geo.pitch:.2f}°")
            print(f"   Yaw: {geo.yaw:.2f}°")
            print(f"   Roll: {geo.roll:.2f}°")
            print(f"   Face size: {geo.face_width:.0f} x {geo.face_height:.0f} px")
    else:
        print("❌ No face detected")

    detector.release()
    print()


def example_frame_processor():
    """프레임 프로세서 사용 예제"""
    print("=" * 60)
    print("Example 2: Frame Processor Pipeline")
    print("=" * 60)

    # 컴포넌트 초기화
    config = DetectionConfig(
        model_complexity=1,
        static_image_mode=True,
        enable_face_geometry=True
    )

    detector = FaceDetector(config)
    extractor = LandmarkExtractor()

    # 프레임 프로세서 생성
    processor = FrameProcessor(detector, extractor, enable_smoothing=False)

    # 이미지 처리
    try:
        result = processor.process_image('data/sample_images/face.jpg')

        if result.detection_result and result.detection_result.success:
            print("✅ Processing successful!")
            print(f"📊 Detection result available")
            print(f"🖼️  Image shape: {result.metadata['image_shape']}")

            # 특정 얼굴 영역 추출
            landmarks = result.detection_result.landmarks
            left_eye = extractor.get_facial_region(landmarks, 'left_eye')
            print(f"\n👁️  Left eye landmarks: {len(left_eye)} points")
        else:
            print("❌ Processing failed")

    except Exception as e:
        print(f"⚠️  Error: {e}")

    detector.release()
    print()


def example_geometry_calculator():
    """기하학 계산 예제"""
    print("=" * 60)
    print("Example 3: Geometry Calculator")
    print("=" * 60)

    # 더미 landmarks 생성 (테스트용)
    from src.models import Landmark

    dummy_landmarks = []
    for i in range(468):
        x = 0.5 + (i % 10) * 0.01
        y = 0.5 + (i // 10) * 0.01
        dummy_landmarks.append(Landmark(
            x=x, y=y, z=0.0,
            pixel_x=int(x * 640), pixel_y=int(y * 480)
        ))

    # 기하학 정보 계산
    calculator = GeometryCalculator()
    geometry = calculator.get_face_geometry(dummy_landmarks)

    print("📐 Calculated Face Geometry:")
    print(f"   Pitch: {geometry.pitch:.2f}°")
    print(f"   Yaw: {geometry.yaw:.2f}°")
    print(f"   Roll: {geometry.roll:.2f}°")
    print(f"   Face width: {geometry.face_width:.1f} px")
    print(f"   Face height: {geometry.face_height:.1f} px")
    print(f"   Estimated distance: {geometry.estimated_distance:.1f} mm")
    print()


def example_facial_regions():
    """얼굴 영역 사용 예제"""
    print("=" * 60)
    print("Example 4: Facial Regions")
    print("=" * 60)

    from src.config.constants import FACIAL_REGIONS

    print("📋 Available facial regions:")
    for region_name, indices in FACIAL_REGIONS.items():
        print(f"   • {region_name}: {len(indices)} landmarks")

    print(f"\n✨ Total regions: {len(FACIAL_REGIONS)}")
    print()


if __name__ == "__main__":
    print("\n🎯 Facial Landmark Detection - Basic Usage Examples\n")

    # Run examples
    try:
        example_facial_regions()
        example_geometry_calculator()

        # These require MediaPipe (Python 3.9-3.11)
        # example_basic_detection()
        # example_frame_processor()

        print("=" * 60)
        print("✅ Examples completed!")
        print("=" * 60)
        print("\n⚠️  Note: Full detection examples require MediaPipe")
        print("   Install in Python 3.9-3.11 environment:")
        print("   $ pip install mediapipe opencv-python numpy\n")

    except ImportError as e:
        print(f"\n⚠️  Import Error: {e}")
        print("   Please ensure you're in Python 3.9-3.11 environment")
        print("   with mediapipe installed.\n")
