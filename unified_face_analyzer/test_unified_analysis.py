#!/usr/bin/env python3
"""
Test script for unified face analysis with actual processing.
Tests the complete pipeline: MediaPipe + Hairstyle Analysis
"""

import cv2
import numpy as np
from pathlib import Path
import time

from core.unified_analyzer import UnifiedFaceAnalyzer
from utils import get_logger

logger = get_logger(__name__)

def create_test_image():
    """Create a simple test image (solid color)"""
    # Create a 640x480 BGR image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Fill with a face-like color (skin tone)
    image[:] = (180, 200, 220)  # BGR format
    return image

def test_unified_analysis():
    """Test the unified face analyzer with generated image"""

    print("="*60)
    print("  Unified Face Analyzer - Real Analysis Test")
    print("="*60)
    print()

    # Create analyzer
    print("[1/4] Creating UnifiedFaceAnalyzer...")
    start_time = time.time()
    analyzer = UnifiedFaceAnalyzer()
    init_time = time.time() - start_time
    print(f"✅ Analyzer created in {init_time:.2f}s")
    print()

    # Create test image
    print("[2/4] Creating test image...")
    test_image = create_test_image()
    test_image_path = Path("test_generated_image.png")
    cv2.imwrite(str(test_image_path), test_image)
    print(f"✅ Test image saved: {test_image_path}")
    print(f"   Image shape: {test_image.shape}")
    print()

    # Analyze image
    print("[3/4] Running unified analysis...")
    print("   This will test both MediaPipe and Hairstyle modules")
    print()

    start_time = time.time()
    result = analyzer.analyze_image(str(test_image_path))
    analysis_time = time.time() - start_time

    print(f"✅ Analysis completed in {analysis_time:.2f}s")
    print()

    # Display results
    print("[4/4] Analysis Results:")
    print("-" * 60)

    if result['success']:
        print("✅ Analysis Status: SUCCESS")
        print()

        # MediaPipe results
        if 'mediapipe' in result:
            mp_result = result['mediapipe']
            print("📍 MediaPipe Results:")
            print(f"   - Faces detected: {mp_result.get('faces_detected', 0)}")
            print(f"   - Landmarks count: {mp_result.get('landmarks_count', 0)}")

            if 'face_geometry' in mp_result:
                geom = mp_result['face_geometry']
                print(f"   - Face Geometry:")
                print(f"     • Pitch: {geom.get('pitch', 0):.2f}°")
                print(f"     • Yaw: {geom.get('yaw', 0):.2f}°")
                print(f"     • Roll: {geom.get('roll', 0):.2f}°")
            print()

        # Hairstyle results
        if 'hairstyle' in result:
            hs_result = result['hairstyle']
            print("💇 Hairstyle Analysis Results:")

            if 'classification' in hs_result:
                print(f"   - Classification: {hs_result['classification']}")

            if 'clip_results' in hs_result:
                clip_res = hs_result['clip_results']
                print(f"   - CLIP Results:")
                print(f"     • Gender: {clip_res.get('gender', 'N/A')}")
                print(f"     • Glasses: {clip_res.get('glasses', 'N/A')}")
                print(f"     • Beard: {clip_res.get('beard', 'N/A')}")
            print()

        # Metadata
        if 'metadata' in result:
            meta = result['metadata']
            print("⚙️  Metadata:")
            print(f"   - Total processing time: {meta.get('total_processing_time_ms', 0):.2f}ms")
            print(f"   - Timestamp: {meta.get('timestamp', 'N/A')}")
            print()

        # Compact result
        print("📦 Compact Result (for TCP):")
        compact = analyzer.get_compact_result(result)
        print(f"   Keys: {list(compact.keys())}")
        print()

    else:
        print("❌ Analysis Status: FAILED")
        if 'error' in result:
            print(f"   Error: {result['error']}")
        print()

    print("="*60)
    print()

    # Cleanup
    if test_image_path.exists():
        test_image_path.unlink()
        print("🧹 Cleaned up test image")

    return result

def main():
    """Main test execution"""
    try:
        result = test_unified_analysis()

        if result['success']:
            print("✅ Unified analysis test PASSED")
            return 0
        else:
            print("❌ Unified analysis test FAILED")
            return 1

    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        print(f"\n❌ Test failed with exception: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
