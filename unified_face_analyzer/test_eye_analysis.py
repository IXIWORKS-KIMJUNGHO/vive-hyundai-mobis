#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test eye shape analysis for specific image"""

import sys
import io
import cv2
import numpy as np
from core.unified_analyzer import UnifiedFaceAnalyzer

# Windows UTF-8 fix
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def test_image(image_path):
    print(f"\n{'='*80}")
    print(f"Testing image: {image_path}")
    print(f"{'='*80}\n")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    print(f"✅ Image loaded: {img.shape}")
    
    # Initialize analyzer
    print("Initializing UnifiedFaceAnalyzer...")
    analyzer = UnifiedFaceAnalyzer()
    
    # Analyze
    print("Analyzing image...")
    result = analyzer.analyze(img)
    
    # Print results
    print(f"\n{'='*80}")
    print("Analysis Results:")
    print(f"{'='*80}\n")
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"Classification: {result.get('classification', 'N/A')}")
    
    clip_results = result.get('clip_results', {})
    print(f"\nCLIP Results:")
    print(f"  Gender: {clip_results.get('gender', 'N/A')} (confidence: {clip_results.get('gender_confidence', 0):.2f})")
    print(f"  Glasses: {clip_results.get('glasses', 'N/A')} (confidence: {clip_results.get('glasses_confidence', 0):.2f})")
    print(f"  Beard: {clip_results.get('beard', 'N/A')} (confidence: {clip_results.get('beard_confidence', 0):.2f})")
    
    # Check for MediaPipe results
    mediapipe_results = result.get('mediapipe_results', {})
    print(f"\nMediaPipe Results:")
    
    if 'face_analysis' in mediapipe_results:
        face_analysis = mediapipe_results['face_analysis']
        
        # Eye analysis
        if 'eye_analysis' in face_analysis:
            eye_analysis = face_analysis['eye_analysis']
            print(f"\n  👁️ Eye Analysis:")
            print(f"    Overall shape: {eye_analysis.get('overall_eye_shape', 'N/A')}")
            print(f"    Left eye: {eye_analysis.get('left_eye_shape', 'N/A')} (angle: {eye_analysis.get('left_eye_angle', 0):.2f}°)")
            print(f"    Right eye: {eye_analysis.get('right_eye_shape', 'N/A')} (angle: {eye_analysis.get('right_eye_angle', 0):.2f}°)")
            print(f"    Average angle: {eye_analysis.get('average_eye_angle', 0):.2f}°")
            print(f"    Confidence: {eye_analysis.get('confidence', 0):.2f}")
        else:
            print("  ⚠️ No eye analysis found")
        
        # Face shape analysis
        if 'face_shape_analysis' in face_analysis:
            face_shape = face_analysis['face_shape_analysis']
            print(f"\n  👤 Face Shape Analysis:")
            print(f"    Shape: {face_shape.get('face_shape', 'N/A')}")
            print(f"    Aspect ratio: {face_shape.get('aspect_ratio', 0):.3f}")
            print(f"    Width: {face_shape.get('face_width', 0):.1f}px")
            print(f"    Height: {face_shape.get('face_height', 0):.1f}px")
            print(f"    Confidence: {face_shape.get('confidence', 0):.2f}")
        else:
            print("  ⚠️ No face shape analysis found")
    else:
        print("  ❌ No MediaPipe face analysis found!")
    
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    image_path = '../sample_ir_server/camera_capture_20250513_191504.png'
    test_image(image_path)
