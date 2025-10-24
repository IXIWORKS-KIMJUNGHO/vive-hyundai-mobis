"""MediaPipe FaceMesh 기반 얼굴 검출기 (Unified Face Analyzer용)"""

import time
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any, Optional

from models.landmark_models import DetectionResult, Landmark, FaceGeometry
from utils import get_config, get_logger

logger = get_logger(__name__)


class FaceDetector:
    """MediaPipe FaceMesh 기반 얼굴 검출기"""

    def __init__(self):
        """초기화"""
        self.config = get_config()

        # MediaPipe 설정 가져오기
        mp_config = self.config.mediapipe.detection

        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=mp_config.static_image_mode,
                max_num_faces=mp_config.max_num_faces,
                min_detection_confidence=mp_config.min_detection_confidence,
                min_tracking_confidence=mp_config.min_tracking_confidence
            )
            logger.info("MediaPipe FaceMesh initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe FaceMesh: {e}")
            raise

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        이미지에서 얼굴 검출 수행

        Args:
            image: BGR 형식 이미지 (H, W, 3)

        Returns:
            DetectionResult with 468 landmarks
        """
        start_time = time.time()

        # BGR → RGB 변환 (MediaPipe 요구사항)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image

        # MediaPipe 검출
        results = self.face_mesh.process(image_rgb)

        processing_time = (time.time() - start_time) * 1000  # ms

        if not results.multi_face_landmarks:
            logger.debug("No face detected")
            return DetectionResult(
                success=False,
                landmarks=[],
                confidence=0.0,
                processing_time=processing_time
            )

        # 첫 번째 얼굴만 처리 (max_num_faces=1 설정)
        face_landmarks = results.multi_face_landmarks[0]

        # 468개 랜드마크 변환
        h, w = image.shape[:2]
        landmarks = []

        for lm in face_landmarks.landmark:
            landmark = Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility if hasattr(lm, 'visibility') else 1.0,
                pixel_x=int(lm.x * w),
                pixel_y=int(lm.y * h)
            )
            landmarks.append(landmark)

        logger.info(f"Detected face with {len(landmarks)} landmarks")

        return DetectionResult(
            success=True,
            landmarks=landmarks,
            confidence=0.95,  # MediaPipe doesn't provide confidence, use default
            processing_time=processing_time
        )

    def calculate_geometry(self, landmarks: List[Landmark]) -> FaceGeometry:
        """
        얼굴 기하학 정보 계산 (pitch, yaw, roll)

        Args:
            landmarks: 468개 MediaPipe 랜드마크

        Returns:
            FaceGeometry with angles
        """
        if len(landmarks) != 468:
            logger.warning(f"Expected 468 landmarks, got {len(landmarks)}")
            return FaceGeometry()

        # 주요 포인트 추출 (MediaPipe landmark indices)
        nose_tip = landmarks[1]      # 코끝
        chin = landmarks[152]         # 턱
        left_eye = landmarks[33]      # 왼쪽 눈
        right_eye = landmarks[263]    # 오른쪽 눈

        # Pitch 계산 (상하 회전)
        pitch = np.arctan2(nose_tip.y - chin.y, nose_tip.z - chin.z)
        pitch_deg = np.degrees(pitch)

        # Yaw 계산 (좌우 회전)
        eye_center_x = (left_eye.x + right_eye.x) / 2
        yaw = np.arctan2(nose_tip.x - eye_center_x, nose_tip.z)
        yaw_deg = np.degrees(yaw)

        # Roll 계산 (기울기)
        roll = np.arctan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x)
        roll_deg = np.degrees(roll)

        # 얼굴 크기 계산
        face_width = abs(right_eye.x - left_eye.x)
        face_height = abs(chin.y - nose_tip.y)

        return FaceGeometry(
            pitch=float(pitch_deg),
            yaw=float(yaw_deg),
            roll=float(roll_deg),
            face_width=float(face_width),
            face_height=float(face_height)
        )

    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
            logger.debug("MediaPipe FaceMesh closed")


# cv2 import 추가
import cv2
