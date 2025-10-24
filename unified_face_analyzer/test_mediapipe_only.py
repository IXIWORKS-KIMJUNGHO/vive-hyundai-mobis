#!/usr/bin/env python3
"""
MediaPipe Only Test
MediaPipe 모듈만 테스트
"""

import sys
sys.path.insert(0, '.')

def test_mediapipe_imports():
    """MediaPipe import 테스트"""
    print("=== MediaPipe Import Test ===\n")

    try:
        from models.landmark_models import Landmark, FaceGeometry, DetectionResult
        print("✅ Landmark Models imported")

        from core.mediapipe import FaceDetector
        print("✅ MediaPipe FaceDetector imported")

        from utils import get_config, get_logger
        print("✅ Utils imported")

        print("\n=== All MediaPipe imports successful! ===\n")
        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mediapipe_detector_creation():
    """MediaPipe Detector 생성 테스트"""
    print("=== MediaPipe Detector Creation Test ===\n")

    try:
        from core.mediapipe import FaceDetector

        detector = FaceDetector()
        print(f"✅ FaceDetector created successfully")
        print(f"   Config loaded from: {detector.config.config_path}")

        print("\n=== Detector creation successful! ===\n")
        return True

    except Exception as e:
        print(f"❌ Detector creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Config 테스트"""
    print("=== Config Test ===\n")

    try:
        from utils import get_config

        config = get_config()

        # MediaPipe 설정 확인
        print("✅ MediaPipe config:")
        print(f"   - static_image_mode: {config.mediapipe.detection.static_image_mode}")
        print(f"   - max_num_faces: {config.mediapipe.detection.max_num_faces}")
        print(f"   - min_detection_confidence: {config.mediapipe.detection.min_detection_confidence}")

        # Unified Analysis 설정 확인
        print("\n✅ Unified Analysis config:")
        print(f"   - enable_mediapipe: {config.unified_analysis.enable_mediapipe}")
        print(f"   - enable_hairstyle: {config.unified_analysis.enable_hairstyle}")
        print(f"   - output_format: {config.unified_analysis.output_format}")

        print("\n=== Config test passed! ===\n")
        return True

    except Exception as e:
        print(f"❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 테스트 실행"""
    print("\n" + "="*60)
    print("  MediaPipe Module Test")
    print("="*60 + "\n")

    results = []

    # Test 1: Imports
    results.append(("Imports", test_mediapipe_imports()))

    # Test 2: Config
    results.append(("Config", test_config()))

    # Test 3: Detector Creation
    results.append(("Detector Creation", test_mediapipe_detector_creation()))

    # Summary
    print("="*60)
    print("  Test Summary")
    print("="*60)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:25} {status}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    print("="*60 + "\n")

    if total_passed == total_tests:
        print("🎉 All MediaPipe tests passed!")
        print("\nNote: Full system test requires dlib installation.")
        print("To install dlib: brew install cmake && pip install dlib\n")

    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
