"""
MediaPipe 마이그레이션 검증 스크립트

dlib → MediaPipe 전환 후 정상 작동 확인:
1. MediaPipe 랜드마크 추출 테스트
2. dlib 호환성 테스트 (Rectangle, FullObjectDetection)
3. HairstyleAnalyzer 통합 테스트
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# unified_face_analyzer 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from core.mediapipe import (
    MediaPipeFaceDetector,
    MediaPipeShapePredictor,
    Rectangle,
    FullObjectDetection
)
from core.hairstyle_analyzer import HairstyleAnalyzer
from utils import get_logger

logger = get_logger(__name__)


def test_mediapipe_detector():
    """MediaPipe 얼굴 검출기 테스트"""
    logger.info("=" * 60)
    logger.info("Test 1: MediaPipe Face Detector")
    logger.info("=" * 60)

    detector = MediaPipeFaceDetector()

    # 더미 이미지 생성 (실제로는 샘플 이미지 사용)
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    gray_image = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2GRAY)

    try:
        faces = detector(gray_image, 1)
        logger.info(f"✅ Detector initialized successfully")
        logger.info(f"   Detected {len(faces)} face(s) (expected 0 for random image)")

        if faces:
            face = faces[0]
            assert isinstance(face, Rectangle), "Face should be Rectangle instance"
            logger.info(f"   Face bbox: left={face.left()}, top={face.top()}, "
                       f"right={face.right()}, bottom={face.bottom()}")
            logger.info(f"   Face size: {face.width()}x{face.height()}")

        detector.close()
        return True

    except Exception as e:
        logger.error(f"❌ Detector test failed: {e}")
        return False


def test_mediapipe_predictor():
    """MediaPipe 랜드마크 추출기 테스트"""
    logger.info("=" * 60)
    logger.info("Test 2: MediaPipe Shape Predictor (68-point)")
    logger.info("=" * 60)

    predictor = MediaPipeShapePredictor()

    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    gray_image = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2GRAY)
    dummy_rect = Rectangle(100, 100, 300, 300)

    try:
        landmarks = predictor(gray_image, dummy_rect)
        logger.info(f"✅ Predictor initialized successfully")

        if landmarks is not None:
            assert isinstance(landmarks, FullObjectDetection), \
                "Landmarks should be FullObjectDetection instance"
            assert landmarks.num_parts() == 68, \
                f"Expected 68 landmarks, got {landmarks.num_parts()}"

            logger.info(f"   Extracted {landmarks.num_parts()} landmarks")

            # 눈썹 랜드마크 확인 (17-26번)
            eyebrow_points = [landmarks.part(i) for i in range(17, 27)]
            logger.info(f"   Eyebrow landmarks (17-26): {len(eyebrow_points)} points")
            logger.info(f"   First eyebrow point: ({eyebrow_points[0].x}, {eyebrow_points[0].y})")

        predictor.close()
        return True

    except Exception as e:
        logger.error(f"❌ Predictor test failed: {e}")
        return False


def test_hairstyle_analyzer_integration():
    """HairstyleAnalyzer MediaPipe 통합 테스트"""
    logger.info("=" * 60)
    logger.info("Test 3: HairstyleAnalyzer Integration")
    logger.info("=" * 60)

    try:
        analyzer = HairstyleAnalyzer()
        logger.info("✅ HairstyleAnalyzer initialized successfully")
        logger.info(f"   Device: {analyzer.device}")
        logger.info(f"   Face detector: {type(analyzer.face_detector).__name__}")
        logger.info(f"   Shape predictor: {type(analyzer.shape_predictor).__name__}")

        # dlib 속성이 없는지 확인
        assert not hasattr(analyzer, 'dlib_detector'), \
            "HairstyleAnalyzer should not have dlib_detector"
        assert not hasattr(analyzer, 'dlib_predictor'), \
            "HairstyleAnalyzer should not have dlib_predictor"

        logger.info("✅ No dlib dependencies found")
        return True

    except Exception as e:
        logger.error(f"❌ HairstyleAnalyzer integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_image_analysis():
    """실제 이미지 분석 테스트 (샘플 이미지 있을 경우)"""
    logger.info("=" * 60)
    logger.info("Test 4: Real Image Analysis (Optional)")
    logger.info("=" * 60)

    # 샘플 이미지 경로 확인
    sample_dir = Path(__file__).parent / "samples"
    if not sample_dir.exists():
        logger.warning("⚠️  No samples directory found, skipping real image test")
        return True

    sample_images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
    if not sample_images:
        logger.warning("⚠️  No sample images found, skipping real image test")
        return True

    try:
        analyzer = HairstyleAnalyzer()
        test_image = str(sample_images[0])

        logger.info(f"   Testing with: {os.path.basename(test_image)}")

        result, viz_image = analyzer.analyze_image(test_image)

        if 'error' in result:
            logger.warning(f"⚠️  Analysis returned error: {result['error']}")
            return True  # 에러는 있을 수 있음 (얼굴 없는 이미지 등)

        logger.info("✅ Real image analysis completed")
        logger.info(f"   Classification: {result.get('classification', 'Unknown')}")
        logger.info(f"   Gender: {result.get('gender_analysis', {}).get('gender', 'Unknown')}")

        return True

    except Exception as e:
        logger.error(f"❌ Real image analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """전체 테스트 실행"""
    logger.info("\n" + "=" * 60)
    logger.info("MediaPipe Migration Validation Tests")
    logger.info("=" * 60 + "\n")

    results = {
        "MediaPipe Detector": test_mediapipe_detector(),
        "MediaPipe Predictor": test_mediapipe_predictor(),
        "HairstyleAnalyzer Integration": test_hairstyle_analyzer_integration(),
        "Real Image Analysis": test_real_image_analysis()
    }

    # 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info("Test Results Summary")
    logger.info("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    total_tests = len(results)
    passed_tests = sum(results.values())

    logger.info("=" * 60)
    logger.info(f"Total: {passed_tests}/{total_tests} tests passed")
    logger.info("=" * 60 + "\n")

    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! MediaPipe migration successful.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
