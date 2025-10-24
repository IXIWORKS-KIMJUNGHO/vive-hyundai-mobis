"""얼굴 기하학 계산 유틸리티"""

import numpy as np
from typing import List, Tuple
import math

from ..models import Landmark, FaceGeometry


class GeometryCalculator:
    """얼굴 기하학 계산"""

    @staticmethod
    def calculate_face_angles(landmarks: List[Landmark]) -> Tuple[float, float, float]:
        """
        얼굴 각도 계산 (pitch, yaw, roll)

        Args:
            landmarks: 468개 landmarks

        Returns:
            (pitch, yaw, roll) 튜플 (도 단위)
        """
        # 주요 포인트 추출
        nose_tip = landmarks[4]
        chin = landmarks[152]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]

        # Yaw (좌우 회전) 계산
        eye_center_x = (left_eye.x + right_eye.x) / 2
        nose_x = nose_tip.x
        yaw = math.degrees(math.atan2(nose_x - eye_center_x, 0.5)) * 2

        # Pitch (상하 회전) 계산
        eye_center_y = (left_eye.y + right_eye.y) / 2
        nose_y = nose_tip.y
        pitch = math.degrees(math.atan2(nose_y - eye_center_y, 0.3)) * 1.5

        # Roll (기울기) 계산
        dx = right_eye.x - left_eye.x
        dy = right_eye.y - left_eye.y
        roll = math.degrees(math.atan2(dy, dx))

        return (pitch, yaw, roll)

    @staticmethod
    def calculate_face_size(landmarks: List[Landmark]) -> Tuple[float, float]:
        """
        얼굴 크기 계산

        Args:
            landmarks: 468개 landmarks

        Returns:
            (width, height) 튜플 (픽셀 단위)
        """
        if not landmarks or landmarks[0].pixel_x is None:
            return (0.0, 0.0)

        x_coords = [lm.pixel_x for lm in landmarks]
        y_coords = [lm.pixel_y for lm in landmarks]

        width = float(max(x_coords) - min(x_coords))
        height = float(max(y_coords) - min(y_coords))

        return (width, height)

    @staticmethod
    def estimate_distance(
        landmarks: List[Landmark],
        reference_face_width: float = 140.0  # mm
    ) -> float:
        """
        얼굴까지의 거리 추정 (간단한 핀홀 카메라 모델)

        Args:
            landmarks: 468개 landmarks
            reference_face_width: 평균 얼굴 너비 (mm)

        Returns:
            추정 거리 (mm)
        """
        face_width_pixels, _ = GeometryCalculator.calculate_face_size(landmarks)

        if face_width_pixels == 0:
            return 0.0

        # 간단한 거리 추정 (실제로는 카메라 파라미터 필요)
        # distance = (real_width * focal_length) / pixel_width
        # 여기서는 간단한 비율 사용
        estimated_distance = (reference_face_width * 1000) / face_width_pixels

        return estimated_distance

    @staticmethod
    def calculate_landmark_distance(lm1: Landmark, lm2: Landmark) -> float:
        """
        두 landmark 간 유클리드 거리 계산

        Args:
            lm1, lm2: 두 개의 Landmark

        Returns:
            거리 (정규화 좌표 기준)
        """
        dx = lm2.x - lm1.x
        dy = lm2.y - lm1.y
        dz = lm2.z - lm1.z
        return math.sqrt(dx**2 + dy**2 + dz**2)

    @staticmethod
    def get_face_geometry(landmarks: List[Landmark]) -> FaceGeometry:
        """
        전체 얼굴 기하학 정보 계산

        Args:
            landmarks: 468개 landmarks

        Returns:
            FaceGeometry 객체
        """
        pitch, yaw, roll = GeometryCalculator.calculate_face_angles(landmarks)
        width, height = GeometryCalculator.calculate_face_size(landmarks)
        distance = GeometryCalculator.estimate_distance(landmarks)

        return FaceGeometry(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            face_width=width,
            face_height=height,
            estimated_distance=distance
        )
