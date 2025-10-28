"""
MediaPipe Face Mesh wrapper with dlib-compatible interface.

Provides face detection and 68-point landmark extraction
using MediaPipe Face Mesh, with dlib-compatible output format.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Optional

from .dlib_compatible import Rectangle, FullObjectDetection
from .landmark_mapping import (
    convert_mediapipe_to_dlib68,
    get_face_bbox
)
from utils import get_logger

logger = get_logger(__name__)


class MediaPipeFaceDetector:
    """
    MediaPipe 기반 얼굴 검출기 (dlib.get_frontal_face_detector 호환).

    dlib 검출기를 대체하여 더 빠르고 정확한 얼굴 검출 제공.
    """
    def __init__(self,
                 static_image_mode: bool = True,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5):
        """
        Args:
            static_image_mode: 정적 이미지 모드 (True 권장)
            max_num_faces: 최대 검출 얼굴 수
            min_detection_confidence: 최소 검출 신뢰도
        """
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        logger.info(f"MediaPipe FaceDetector initialized (max_faces={max_num_faces})")

    def __call__(self, image: np.ndarray, upsample_num_times: int = 1) -> List[Rectangle]:
        """
        얼굴 검출 (dlib detector 호환 인터페이스).

        Args:
            image: 입력 이미지 (grayscale or BGR)
            upsample_num_times: 업샘플링 횟수 (MediaPipe에서는 무시)

        Returns:
            List[Rectangle]: 검출된 얼굴 바운딩 박스 리스트
        """
        # Grayscale → BGR 변환
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # BGR → RGB 변환 (MediaPipe 요구사항)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # MediaPipe 얼굴 검출
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            logger.debug("No faces detected by MediaPipe")
            return []

        # 바운딩 박스 계산
        rectangles = []
        for face_landmarks in results.multi_face_landmarks:
            bbox = get_face_bbox(face_landmarks.landmark, w, h)
            rect = Rectangle(
                left=bbox['x_min'],
                top=bbox['y_min'],
                right=bbox['x_max'],
                bottom=bbox['y_max']
            )
            rectangles.append(rect)

        logger.debug(f"Detected {len(rectangles)} face(s)")
        return rectangles

    def close(self):
        """리소스 정리"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


class MediaPipeShapePredictor:
    """
    MediaPipe 기반 68점 랜드마크 추출기 (dlib.shape_predictor 호환).

    468점 → 68점 변환을 통해 dlib 코드와 호환성 유지.
    """
    def __init__(self,
                 static_image_mode: bool = True,
                 max_num_faces: int = 1,
                 min_detection_confidence: float = 0.5):
        """
        Args:
            static_image_mode: 정적 이미지 모드
            max_num_faces: 최대 얼굴 수
            min_detection_confidence: 최소 검출 신뢰도
        """
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        logger.info("MediaPipe ShapePredictor initialized (68-point landmarks)")

    def __call__(self, image: np.ndarray, rect: Rectangle) -> Optional[FullObjectDetection]:
        """
        68점 랜드마크 추출 (dlib predictor 호환 인터페이스).

        Args:
            image: 입력 이미지 (grayscale or BGR)
            rect: 얼굴 바운딩 박스 (사용하지 않음, 호환성 유지용)

        Returns:
            FullObjectDetection: 68점 랜드마크 객체 (dlib 호환)
        """
        # Grayscale → BGR 변환
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # BGR → RGB 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # MediaPipe 랜드마크 추출
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            logger.warning("No face landmarks detected")
            return None

        # 첫 번째 얼굴 사용
        face_landmarks = results.multi_face_landmarks[0]

        # 468점 → 68점 변환
        dlib68_points = convert_mediapipe_to_dlib68(
            face_landmarks.landmark,
            w, h
        )

        # dlib 호환 객체 생성
        full_detection = FullObjectDetection(rect, dlib68_points)
        logger.debug(f"Extracted 68 landmarks for face at {rect}")

        return full_detection

    def close(self):
        """리소스 정리"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
