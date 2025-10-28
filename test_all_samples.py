"""
모든 샘플 이미지로 MediaPipe 마이그레이션 전체 테스트
"""
import sys
import os
from pathlib import Path
import time

# unified_face_analyzer 경로 추가
sys.path.insert(0, str(Path(__file__).parent / "unified_face_analyzer"))

try:
    import cv2
    import numpy as np
    from core.hairstyle_analyzer import HairstyleAnalyzer
    from utils import get_logger

    logger = get_logger(__name__)

    def test_all_sample_images():
        """모든 샘플 이미지로 전체 테스트"""
        logger.info("=" * 80)
        logger.info("Complete MediaPipe Migration Test - All Sample Images")
        logger.info("=" * 80)

        # 샘플 이미지 경로
        sample_dir = Path(__file__).parent / "unified_face_analyzer" / "sample_images"

        if not sample_dir.exists():
            logger.error(f"Sample directory not found: {sample_dir}")
            return False

        sample_images = sorted(list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpg")))

        if not sample_images:
            logger.error("No sample images found")
            return False

        logger.info(f"Found {len(sample_images)} sample images")
        logger.info("")

        # HairstyleAnalyzer 초기화
        try:
            start_time = time.time()
            logger.info("Initializing HairstyleAnalyzer with MediaPipe...")
            analyzer = HairstyleAnalyzer()
            init_time = time.time() - start_time

            logger.info(f"✅ Analyzer initialized successfully ({init_time:.2f}s)")
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
        total_analysis_time = 0

        for idx, img_path in enumerate(sample_images, 1):
            logger.info("-" * 80)
            logger.info(f"Test {idx}/{len(sample_images)}: {img_path.name}")
            logger.info("-" * 80)

            try:
                start_time = time.time()
                result, viz_image = analyzer.analyze_image(str(img_path))
                analysis_time = time.time() - start_time
                total_analysis_time += analysis_time

                if 'error' in result:
                    logger.warning(f"⚠️  Analysis error: {result['error']}")
                    results.append({
                        'image': img_path.name,
                        'status': 'error',
                        'error': result['error'],
                        'time': analysis_time
                    })
                    continue

                # 결과 추출
                classification = result.get('classification', 'Unknown')
                gender = result.get('gender_analysis', {}).get('gender', 'Unknown')
                gender_conf = result.get('gender_analysis', {}).get('confidence', 0)
                glasses = result.get('glasses_analysis', {}).get('has_glasses', False)
                glasses_conf = result.get('glasses_analysis', {}).get('confidence', 0)
                beard = result.get('beard_analysis', {}).get('has_beard', False)
                beard_conf = result.get('beard_analysis', {}).get('confidence', 0)

                logger.info(f"✅ Analysis completed successfully ({analysis_time:.2f}s)")
                logger.info(f"   Hairstyle: {classification}")
                logger.info(f"   Gender: {gender} (confidence: {gender_conf:.1%})")
                logger.info(f"   Glasses: {'Yes' if glasses else 'No'} (confidence: {glasses_conf:.1%})")
                logger.info(f"   Beard: {'Yes' if beard else 'No'} (confidence: {beard_conf:.1%})")

                results.append({
                    'image': img_path.name,
                    'status': 'success',
                    'classification': classification,
                    'gender': gender,
                    'gender_conf': gender_conf,
                    'glasses': glasses,
                    'glasses_conf': glasses_conf,
                    'beard': beard,
                    'beard_conf': beard_conf,
                    'time': analysis_time
                })

            except Exception as e:
                logger.error(f"❌ Test failed: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'image': img_path.name,
                    'status': 'exception',
                    'error': str(e),
                    'time': 0
                })

            logger.info("")

        # 최종 결과 요약
        logger.info("=" * 80)
        logger.info("Complete Test Results Summary")
        logger.info("=" * 80)

        success_count = sum(1 for r in results if r['status'] == 'success')
        error_count = sum(1 for r in results if r['status'] == 'error')
        exception_count = sum(1 for r in results if r['status'] == 'exception')

        logger.info(f"Total images tested: {len(results)}")
        logger.info(f"✅ Successful: {success_count}")
        logger.info(f"⚠️  Analysis errors: {error_count}")
        logger.info(f"❌ Exceptions: {exception_count}")
        logger.info("")

        # 성능 통계
        if success_count > 0:
            avg_time = total_analysis_time / success_count
            logger.info(f"Performance Statistics:")
            logger.info(f"  Initialization time: {init_time:.2f}s")
            logger.info(f"  Average analysis time: {avg_time:.2f}s/image")
            logger.info(f"  Total analysis time: {total_analysis_time:.2f}s")
            logger.info("")

        # 결과 상세 출력
        if success_count > 0:
            logger.info("Detailed Results:")
            logger.info("")

            # 헤어스타일 별로 그룹화
            hairstyles = {}
            for result in results:
                if result['status'] == 'success':
                    style = result['classification']
                    if style not in hairstyles:
                        hairstyles[style] = []
                    hairstyles[style].append(result)

            for style, items in sorted(hairstyles.items()):
                logger.info(f"  {style} ({len(items)} images):")
                for item in items:
                    gender_str = f"{item['gender']} ({item['gender_conf']:.0%})"
                    glasses_str = "👓 Glasses" if item['glasses'] else "No Glasses"
                    beard_str = "🧔 Beard" if item['beard'] else "No Beard"
                    logger.info(f"    - {item['image']}: {gender_str}, {glasses_str}, {beard_str}")
                logger.info("")

        # 에러/예외 출력
        if error_count > 0 or exception_count > 0:
            logger.warning("Failed Tests:")
            for result in results:
                if result['status'] in ['error', 'exception']:
                    logger.warning(f"  ❌ {result['image']}: {result.get('error', 'Unknown error')}")
            logger.info("")

        logger.info("=" * 80)

        if exception_count == 0:
            logger.info("🎉 Complete MediaPipe migration test finished successfully!")
            logger.info(f"   {success_count}/{len(results)} images analyzed without exceptions")
            logger.info(f"   Success rate: {success_count/len(results)*100:.1f}%")
            return True
        else:
            logger.warning(f"⚠️  {exception_count} tests failed with exceptions.")
            return False

    if __name__ == "__main__":
        success = test_all_sample_images()
        sys.exit(0 if success else 1)

except ImportError as e:
    print(f"Error: Required modules not found: {e}")
    print("\nPlease activate the virtual environment first:")
    print("  cd unified_face_analyzer")
    print("  .\\venv\\Scripts\\activate  (Windows)")
    print("  source venv/bin/activate   (Linux/macOS)")
    sys.exit(1)
