"""IR 이미지 전처리 모듈"""

import cv2
import numpy as np
from pathlib import Path


class IRImagePreprocessor:
    """IR 카메라 이미지 전처리 클래스"""

    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size

    def load_image(self, image_path: str) -> np.ndarray:
        """이미지 로드"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        return img

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """이미지 정규화 (0-255 범위)"""
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """대비 강화 (CLAHE)"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def resize(self, image: np.ndarray) -> np.ndarray:
        """이미지 리사이즈"""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)

    def to_rgb(self, image: np.ndarray) -> np.ndarray:
        """그레이스케일 → RGB 변환 (모델 입력용)"""
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    def process(self, image_path: str, output_path: str = None) -> np.ndarray:
        """전체 전처리 파이프라인"""
        img = self.load_image(image_path)
        img = self.normalize(img)
        img = self.enhance_contrast(img)
        img = self.resize(img)
        img_rgb = self.to_rgb(img)

        if output_path:
            cv2.imwrite(output_path, img_rgb)

        return img_rgb


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python preprocess_ir.py <이미지_경로> [출력_경로]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "processed_output.jpg"

    preprocessor = IRImagePreprocessor()
    processed = preprocessor.process(input_path, output_path)
    print(f"✅ 전처리 완료: {processed.shape} → {output_path}")