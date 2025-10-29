#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Viewer for Y8 Data
CameraClient.cs와 동일한 이미지 처리 로직으로 Y8 데이터를 시각화

사용법:
    python image_viewer.py <y8_file_path>

예시:
    python image_viewer.py received_image.raw
"""

import sys
import numpy as np
import cv2
from pathlib import Path


class Y8ImageViewer:
    """CameraClient.cs와 동일한 Y8 이미지 처리"""

    def __init__(self, width=1280, height=800):
        self.width = width
        self.height = height
        self.expected_size = width * height

    def process_y8_data(self, data: bytes) -> np.ndarray:
        """
        CameraClient.cs의 ConvertY8ToTextureAndroidStyle() 재현

        Args:
            data: Y8 raw bytes (1280x800 = 1,024,000 bytes)

        Returns:
            BGR 이미지 (OpenCV 형식)
        """
        if len(data) != self.expected_size:
            raise ValueError(
                f"잘못된 데이터 크기: {len(data)} bytes "
                f"(예상: {self.expected_size} bytes)"
            )

        # 1. Y8 데이터를 numpy array로 변환
        y8_array = np.frombuffer(data, dtype=np.uint8)
        y8_image = y8_array.reshape((self.height, self.width))

        # 2. Y축 뒤집기 (CameraClient.cs의 flippedY 로직)
        y8_flipped = np.flipud(y8_image)

        # 3. Grayscale → RGB 변환 (R=G=B=grayValue)
        rgb_image = cv2.cvtColor(y8_flipped, cv2.COLOR_GRAY2RGB)

        # 4. OpenCV는 BGR을 사용하므로 변환
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        return bgr_image

    def display_image(self, image: np.ndarray, window_name="Y8 Image Viewer"):
        """이미지를 화면에 표시"""
        cv2.imshow(window_name, image)
        print(f"\n이미지 표시 중 - '{window_name}'")
        print("아무 키나 누르면 종료됩니다...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image(self, image: np.ndarray, output_path: str):
        """이미지를 파일로 저장"""
        cv2.imwrite(output_path, image)
        print(f"✅ 이미지 저장: {output_path}")


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print("사용법: python image_viewer.py <y8_file_path>")
        print("예시: python image_viewer.py received_image.raw")
        sys.exit(1)

    input_file = Path(sys.argv[1])

    if not input_file.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
        sys.exit(1)

    print("=" * 80)
    print("  Y8 Image Viewer (CameraClient.cs 호환)")
    print("=" * 80)
    print()

    # Y8 데이터 로드
    print(f"📂 파일 로드 중: {input_file}")
    with open(input_file, 'rb') as f:
        y8_data = f.read()

    print(f"   파일 크기: {len(y8_data):,} bytes")

    # 이미지 처리
    viewer = Y8ImageViewer(width=1280, height=800)

    try:
        print(f"\n🎨 Y8 → BGR 변환 중...")
        print(f"   - Y축 뒤집기 (CameraClient.cs 호환)")
        print(f"   - Grayscale → RGB 변환")

        bgr_image = viewer.process_y8_data(y8_data)

        print(f"✅ 변환 완료!")
        print(f"   이미지 크기: {bgr_image.shape[1]}x{bgr_image.shape[0]}")
        print(f"   채널: {bgr_image.shape[2]}")

        # PNG로 저장
        output_file = input_file.with_suffix('.png')
        viewer.save_image(bgr_image, str(output_file))

        # 화면에 표시
        viewer.display_image(bgr_image)

    except Exception as e:
        print(f"❌ 처리 실패: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
