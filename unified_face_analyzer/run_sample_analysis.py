#!/usr/bin/env python3
"""
Run unified face analysis on all sample images.
"""

import cv2
import json
from pathlib import Path
import time
from typing import List, Dict

from core.unified_analyzer import UnifiedFaceAnalyzer
from utils import get_logger

logger = get_logger(__name__)

def analyze_all_samples(sample_dir: Path) -> List[Dict]:
    """Analyze all sample images"""

    print("="*80)
    print("  Unified Face Analyzer - Sample Images Analysis")
    print("="*80)
    print()

    # Create analyzer
    print("[1/3] Initializing UnifiedFaceAnalyzer...")
    start_time = time.time()
    analyzer = UnifiedFaceAnalyzer()
    init_time = time.time() - start_time
    print(f"✅ Analyzer initialized in {init_time:.2f}s")
    print()

    # Find all image files
    image_files = sorted(list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpg")))

    if not image_files:
        print(f"❌ No images found in {sample_dir}")
        return []

    print(f"[2/3] Found {len(image_files)} sample images")
    print()

    # Analyze each image
    print("[3/3] Analyzing images...")
    print("-" * 80)

    results = []
    total_start = time.time()

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] {image_path.name}")
        print("-" * 80)

        try:
            # Run analysis
            start = time.time()
            result = analyzer.analyze_image(str(image_path))
            elapsed = time.time() - start

            # Store result
            result['image_path'] = str(image_path)
            result['image_name'] = image_path.name
            result['analysis_time'] = elapsed
            results.append(result)

            # Display results
            if result['success']:
                print(f"✅ Analysis completed in {elapsed:.2f}s")

                # MediaPipe results
                if 'mediapipe' in result:
                    mp = result['mediapipe']
                    print(f"\n📍 MediaPipe:")
                    print(f"   Landmarks: {mp.get('landmarks_count', 0)}")

                    if 'face_geometry' in mp:
                        geom = mp['face_geometry']
                        print(f"   Geometry: pitch={geom['pitch']:.1f}° yaw={geom['yaw']:.1f}° roll={geom['roll']:.1f}°")

                    # 눈꼬리 분석 결과
                    if 'eye_analysis' in mp:
                        eye = mp['eye_analysis']
                        print(f"   👁️  Eye Shape: {eye['overall_eye_shape']}")
                        print(f"       L={eye['left_eye_shape']} R={eye['right_eye_shape']} (conf={eye['confidence']:.2f})")

                    # 얼굴형 분석 결과
                    if 'face_shape_analysis' in mp:
                        face = mp['face_shape_analysis']
                        print(f"   😊 Face Shape: {face['face_shape']}")
                        print(f"       ratio={face['aspect_ratio']:.2f} (conf={face['confidence']:.2f})")

                # Hairstyle results
                if 'hairstyle' in result:
                    hs = result['hairstyle']
                    print(f"\n💇 Hairstyle:")
                    print(f"   Classification: {hs.get('classification', 'Unknown')}")

                    if 'clip_results' in hs:
                        clip = hs['clip_results']
                        print(f"   CLIP: gender={clip.get('gender', 'N/A')} "
                              f"glasses={clip.get('glasses', 'N/A')} "
                              f"beard={clip.get('beard', 'N/A')}")

            else:
                print(f"❌ Analysis failed")
                if 'error' in result:
                    print(f"   Error: {result['error']}")

        except Exception as e:
            print(f"❌ Exception: {e}")
            results.append({
                'image_path': str(image_path),
                'image_name': image_path.name,
                'success': False,
                'error': str(e)
            })

    total_time = time.time() - total_start

    # Summary
    print("\n" + "="*80)
    print("  Analysis Summary")
    print("="*80)

    successful = sum(1 for r in results if r.get('success', False))
    print(f"\nTotal images: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per image: {total_time/len(results):.2f}s")

    # MediaPipe stats
    mp_detected = sum(1 for r in results
                      if r.get('success') and r.get('mediapipe', {}).get('faces_detected', 0) > 0)
    print(f"\nMediaPipe face detection: {mp_detected}/{len(results)}")

    # Hairstyle stats
    hairstyles = {}
    for r in results:
        if r.get('success') and 'hairstyle' in r:
            classification = r['hairstyle'].get('classification', 'Unknown')
            hairstyles[classification] = hairstyles.get(classification, 0) + 1

    if hairstyles:
        print(f"\nHairstyle classifications:")
        for style, count in sorted(hairstyles.items(), key=lambda x: x[1], reverse=True):
            print(f"   {style}: {count}")

    print("\n" + "="*80)

    return results

def main():
    """Main execution"""
    sample_dir = Path("sample_images")

    if not sample_dir.exists():
        print(f"❌ Sample directory not found: {sample_dir}")
        return 1

    try:
        results = analyze_all_samples(sample_dir)

        # Save results to JSON
        output_file = Path("sample_analysis_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {output_file}")

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\n❌ Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
