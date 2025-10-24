#!/usr/bin/env python3
"""
Unified Face Analyzer Integration Test
통합 시스템 테스트 스크립트
"""

import sys
sys.path.insert(0, '.')

def test_imports():
    """Import 테스트"""
    print("=== Import Test ===\n")

    tests = [
        ("Utils", "from utils import get_config, get_logger"),
        ("Landmark Models", "from models.landmark_models import Landmark, FaceGeometry, DetectionResult"),
        ("MediaPipe", "from core.mediapipe import FaceDetector"),
        ("Hairstyle Analyzer", "from core.hairstyle_analyzer import HairstyleAnalyzer"),
        ("Unified Analyzer", "from core.unified_analyzer import UnifiedFaceAnalyzer"),
    ]

    passed = 0
    failed = 0

    for name, import_statement in tests:
        try:
            exec(import_statement)
            print(f"✅ {name:20} OK")
            passed += 1
        except Exception as e:
            print(f"❌ {name:20} FAILED: {e}")
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===\n")
    return failed == 0


def test_config():
    """Config 테스트"""
    print("=== Config Test ===\n")

    try:
        from utils import get_config

        config = get_config()

        # MediaPipe 설정 확인
        assert hasattr(config, 'mediapipe'), "MediaPipe config not found"
        assert hasattr(config.mediapipe, 'detection'), "MediaPipe detection config not found"

        print("✅ MediaPipe config loaded")
        print(f"   - static_image_mode: {config.mediapipe.detection.static_image_mode}")
        print(f"   - max_num_faces: {config.mediapipe.detection.max_num_faces}")

        # Unified Analysis 설정 확인
        assert hasattr(config, 'unified_analysis'), "Unified analysis config not found"

        print("✅ Unified Analysis config loaded")
        print(f"   - enable_mediapipe: {config.unified_analysis.enable_mediapipe}")
        print(f"   - enable_hairstyle: {config.unified_analysis.enable_hairstyle}")

        print("\n=== Config Test Passed ===\n")
        return True

    except Exception as e:
        print(f"❌ Config test failed: {e}\n")
        return False


def test_analyzer_creation():
    """Analyzer 생성 테스트"""
    print("=== Analyzer Creation Test ===\n")

    try:
        from core.unified_analyzer import UnifiedFaceAnalyzer

        analyzer = UnifiedFaceAnalyzer()
        print(f"✅ UnifiedFaceAnalyzer created: {analyzer}")

        print("\n=== Analyzer Creation Passed ===\n")
        return True

    except Exception as e:
        print(f"❌ Analyzer creation failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 테스트 실행"""
    print("\n" + "="*60)
    print("  Unified Face Analyzer - Integration Test")
    print("="*60 + "\n")

    results = []

    # Test 1: Imports
    results.append(("Imports", test_imports()))

    # Test 2: Config
    results.append(("Config", test_config()))

    # Test 3: Analyzer Creation
    results.append(("Analyzer Creation", test_analyzer_creation()))

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

    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
