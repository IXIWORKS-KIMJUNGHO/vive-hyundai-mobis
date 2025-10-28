"""
실제 샘플 이미지로 MediaPipe 마이그레이션 테스트
"""
import sys
import os
from pathlib import Path

# unified_face_analyzer 경로 추가
sys.path.insert(0, str(Path(__file__).parent / "unified_face_analyzer"))

try:
    import cv2
    import numpy as np
    from core.hairstyle_analyzer import HairstyleAnalyzer
    from utils import get_logger

    logger = get_logger(__name__)

    def test_with_sample_images():
        """샘플 이미지로 실제 분석 테스트"""
        logger.info("=" * 80)
        logger.info("MediaPipe Migration Test with Real Sample Images")
        logger.info("=" * 80)

        # 샘플 이미지 경로
        sample_dir = Path(__file__).parent / "unified_face_analyzer" / "sample_images"

        if not sample_dir.exists():
            logger.error(f"Sample directory not found: {sample_dir}")
            return False

        sample_images = list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpg"))

        if not sample_images:
            logger.error("No sample images found")
            return False

        logger.info(f"Found {len(sample_images)} sample images")
        logger.info("")

        # HairstyleAnalyzer 초기화
        try:
            logger.info("Initializing HairstyleAnalyzer with MediaPipe...")
            analyzer = HairstyleAnalyzer()
            logger.info(f"✅ Analyzer initialized successfully")
            logger.info(f"   Device: {analyzer.device}")
            logger.info(f"   Face detector: {type(analyzer.face_detector).__name__}")
            logger.info(f"   Shape predictor: {type(analyzer.shape_predictor).__name__}")
            logger.info("")

        except Exception as e:
            logger.error(f"❌ Failed to initialize analyzer: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 각 샘플 이미지 테스트
        results = []
        for idx, img_path in enumerate(sample_images[:3], 1):  # 처음 3개만 테스트
            logger.info("-" * 80)
            logger.info(f"Test {idx}/{min(3, len(sample_images))}: {img_path.name}")
            logger.info("-" * 80)

            try:
                result, viz_image = analyzer.analyze_image(str(img_path))

                if 'error' in result:
                    logger.warning(f"⚠️  Analysis error: {result['error']}")
                    results.append({'image': img_path.name, 'status': 'error', 'error': result['error']})
                    continue

                # 결과 출력
                classification = result.get('classification', 'Unknown')
                gender = result.get('gender_analysis', {}).get('gender', 'Unknown')
                gender_conf = result.get('gender_analysis', {}).get('confidence', 0)
                glasses = result.get('glasses_analysis', {}).get('has_glasses', False)
                glasses_conf = result.get('glasses_analysis', {}).get('confidence', 0)
                beard = result.get('beard_analysis', {}).get('has_beard', False)
                beard_conf = result.get('beard_analysis', {}).get('confidence', 0)

                logger.info(f"✅ Analysis completed successfully")
                logger.info(f"   Hairstyle: {classification}")
                logger.info(f"   Gender: {gender} (confidence: {gender_conf:.1%})")
                logger.info(f"   Glasses: {'Yes' if glasses else 'No'} (confidence: {glasses_conf:.1%})")
                logger.info(f"   Beard: {'Yes' if beard else 'No'} (confidence: {beard_conf:.1%})")

                results.append({
                    'image': img_path.name,
                    'status': 'success',
                    'classification': classification,
                    'gender': gender,
                    'glasses': glasses,
                    'beard': beard
                })

            except Exception as e:
                logger.error(f"❌ Test failed: {e}")
                import traceback
                traceback.print_exc()
                results.append({'image': img_path.name, 'status': 'exception', 'error': str(e)})

            logger.info("")

        # 최종 결과 요약
        logger.info("=" * 80)
        logger.info("Test Results Summary")
        logger.info("=" * 80)

        success_count = sum(1 for r in results if r['status'] == 'success')
        error_count = sum(1 for r in results if r['status'] == 'error')
        exception_count = sum(1 for r in results if r['status'] == 'exception')

        logger.info(f"Total images tested: {len(results)}")
        logger.info(f"✅ Successful: {success_count}")
        logger.info(f"⚠️  Analysis errors: {error_count}")
        logger.info(f"❌ Exceptions: {exception_count}")
        logger.info("")

        if success_count > 0:
            logger.info("Sample results:")
            for result in results:
                if result['status'] == 'success':
                    logger.info(f"  {result['image']}: {result['classification']} "
                              f"({result['gender']}, "
                              f"Glasses: {'Yes' if result['glasses'] else 'No'}, "
                              f"Beard: {'Yes' if result['beard'] else 'No'})")

        logger.info("=" * 80)

        if exception_count == 0:
            logger.info("🎉 MediaPipe migration test completed successfully!")
            logger.info("   All images were processed without exceptions.")
            return True
        else:
            logger.warning("⚠️  Some tests failed with exceptions.")
            return False

    if __name__ == "__main__":
        success = test_with_sample_images()
        sys.exit(0 if success else 1)

except ImportError as e:
    print(f"Error: Required modules not found: {e}")
    print("\nPlease activate the virtual environment first:")
    print("  cd unified_face_analyzer")
    print("  .\\venv\\Scripts\\activate  (Windows)")
    print("  source venv/bin/activate   (Linux/macOS)")
    sys.exit(1)
