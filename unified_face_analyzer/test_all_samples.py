#!/usr/bin/env python3
"""
모든 샘플 이미지를 분석해서 결과 확인
"""
import os
from pathlib import Path
from core.unified_analyzer import UnifiedFaceAnalyzer

def main():
    # 샘플 이미지 디렉토리
    sample_dir = Path("sample_images")
    images = sorted(sample_dir.glob("*.png"))

    print("=" * 100)
    print("  📸 Sample Image Analysis Test")
    print("=" * 100)
    print(f"Found {len(images)} images\n")

    # Analyzer 초기화
    print("🔧 Initializing UnifiedFaceAnalyzer...")
    analyzer = UnifiedFaceAnalyzer()
    print("✅ Initialization complete\n")

    print("=" * 100)

    # 각 이미지 분석
    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] 📸 {image_path.name}")
        print("-" * 100)

        try:
            result = analyzer.analyze_image(str(image_path))

            # MediaPipe 결과
            if 'mediapipe' in result and result['mediapipe'].get('success'):
                mp = result['mediapipe']
                print(f"  ✅ MediaPipe:")
                print(f"     Landmarks: {mp.get('landmarks_count', 0)}")

                if 'eye_analysis' in mp:
                    eye = mp['eye_analysis']
                    print(f"     👁️  Eye Shape: {eye.get('overall_eye_shape', 'Unknown')}")
                    print(f"         Left:  {eye.get('left_eye_shape', 'Unknown')} (slope: {eye.get('left_eye_slope', 0):.3f})")
                    print(f"         Right: {eye.get('right_eye_shape', 'Unknown')} (slope: {eye.get('right_eye_slope', 0):.3f})")

                if 'face_shape_analysis' in mp:
                    face = mp['face_shape_analysis']
                    print(f"     😊 Face Shape: {face.get('face_shape', 'Unknown')}")
                    print(f"         Aspect Ratio: {face.get('aspect_ratio', 0):.3f}")
            else:
                print(f"  ❌ MediaPipe failed")

            # Hairstyle 결과
            if 'hairstyle' in result:
                hs = result['hairstyle']
                print(f"  ✅ Hairstyle:")

                classification = hs.get('classification', 'Unknown')
                print(f"     💇 Style: {classification}")

                # Gender
                if 'gender_analysis' in hs:
                    gender = hs['gender_analysis']
                    gender_name = gender.get('gender', 'Unknown')
                    gender_conf = gender.get('confidence', 0)
                    print(f"     👤 Gender: {gender_name} ({gender_conf:.1f}%)")
                else:
                    print(f"     👤 Gender: Unknown")

                # Glasses
                if 'glasses_analysis' in hs:
                    glasses = hs['glasses_analysis']
                    has_glasses = glasses.get('has_glasses', False)
                    glasses_conf = glasses.get('confidence', 0)
                    glasses_icon = "👓" if has_glasses else "🚫"
                    glasses_text = "Yes" if has_glasses else "No"
                    print(f"     {glasses_icon} Glasses: {glasses_text} ({glasses_conf:.1f}%)")
                else:
                    print(f"     🚫 Glasses: Unknown")

                # Beard
                if 'beard_analysis' in hs:
                    beard = hs['beard_analysis']
                    has_beard = beard.get('has_beard', False)
                    beard_conf = beard.get('confidence', 0)
                    beard_icon = "🧔" if has_beard else "🚫"
                    beard_text = "Yes" if has_beard else "No"
                    print(f"     {beard_icon} Beard: {beard_text} ({beard_conf:.1f}%)")
                else:
                    print(f"     🚫 Beard: Unknown")
            else:
                print(f"  ❌ Hairstyle analysis failed")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n" + "=" * 100)
    print("  ✅ Analysis Complete")
    print("=" * 100)

if __name__ == "__main__":
    main()
