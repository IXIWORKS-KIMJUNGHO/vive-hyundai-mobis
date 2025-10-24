"""MediaPipe FaceMesh 기반 얼굴 검출기"""

import time
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any

from ..models import DetectionResult, Landmark
from ..config.settings import DetectionConfig
from ..utils.exceptions import DetectionError, ConfigurationError
from ..utils.validators import validate_image, validate_confidence
from .normalizer import CoordinateNormalizer
from .landmark_extractor import LandmarkExtractor


class FaceDetector:
    """MediaPipe FaceMesh 기반 얼굴 검출기"""

    def __init__(self, config: DetectionConfig):
        """
        초기화

        Args:
            config: 검출 설정

        Raises:
            ConfigurationError: 설정이 유효하지 않은 경우
        """
        self.config = config
        self.normalizer = CoordinateNormalizer()
        self.extractor = LandmarkExtractor()

        try:
            # Note: model_complexity parameter was removed in newer MediaPipe versions
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=config.static_image_mode,
                max_num_faces=config.max_num_faces,
                refine_landmarks=config.refine_landmarks,
                min_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize MediaPipe FaceMesh: {e}")

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        이미지에서 얼굴 검출 수행

        Args:
            image: BGR 형식 이미지 (H, W, 3)

        Returns:
            DetectionResult: 검출 결과

        Raises:
            InvalidImageError: 이미지가 유효하지 않은 경우
        """
        validate_image(image)

        start_time = time.time()

        # BGR → RGB 변환
        image_rgb = image.copy()
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image[:, :, ::-1]

        # MediaPipe 처리
        results = self.face_mesh.process(image_rgb)

        processing_time = (time.time() - start_time) * 1000  # ms

        # 검출 실패
        if not results or not results.multi_face_landmarks:
            return DetectionResult(
                success=False,
                processing_time=processing_time
            )

        # Landmark 추출
        height, width = image.shape[:2]
        try:
            landmarks = self.extractor.extract_landmarks(results, width, height)
        except Exception as e:
            raise DetectionError(f"Failed to extract landmarks: {e}")

        # Bounding box 계산
        bbox = self.normalizer.get_bounding_box(landmarks)

        return DetectionResult(
            success=True,
            landmarks=landmarks,
            confidence=1.0,  # MediaPipe는 개별 신뢰도 미제공
            bounding_box=bbox,
            processing_time=processing_time
        )

    def detect_batch(self, images: List[np.ndarray]) -> List[DetectionResult]:
        """
        배치 이미지 검출

        Args:
            images: 이미지 리스트

        Returns:
            검출 결과 리스트
        """
        return [self.detect(img) for img in images]

    def set_confidence_threshold(self, threshold: float):
        """
        검출 신뢰도 임계값 설정

        Args:
            threshold: 신뢰도 임계값 (0.0 ~ 1.0)
        """
        validate_confidence(threshold, "detection threshold")
        self.config.min_detection_confidence = threshold
        # MediaPipe 재초기화 필요
        self.__init__(self.config)

    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환

        Returns:
            모델 설정 및 상태 정보
        """
        return {
            'model_complexity': self.config.model_complexity,
            'max_num_faces': self.config.max_num_faces,
            'min_detection_confidence': self.config.min_detection_confidence,
            'min_tracking_confidence': self.config.min_tracking_confidence,
            'refine_landmarks': self.config.refine_landmarks,
            'static_image_mode': self.config.static_image_mode,
        }

    def release(self):
        """리소스 해제"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

    def __del__(self):
        """소멸자"""
        self.release()
