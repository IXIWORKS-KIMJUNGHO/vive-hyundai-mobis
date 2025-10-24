"""시스템 설정 클래스 정의"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DetectionConfig:
    """얼굴 검출 설정"""

    # MediaPipe FaceMesh 설정
    model_complexity: int = 1  # 0: 빠름, 1: 균형, 2: 정확함
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_num_faces: int = 1
    refine_landmarks: bool = True  # 눈/입 주변 정밀 검출

    # 처리 모드
    static_image_mode: bool = False  # True: 이미지, False: 비디오
    enable_face_geometry: bool = True

    def __post_init__(self):
        """설정 값 검증"""
        if not 0 <= self.model_complexity <= 2:
            raise ValueError("model_complexity must be 0, 1, or 2")
        if not 0.0 <= self.min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be between 0 and 1")
        if not 0.0 <= self.min_tracking_confidence <= 1.0:
            raise ValueError("min_tracking_confidence must be between 0 and 1")
        if self.max_num_faces < 1:
            raise ValueError("max_num_faces must be >= 1")


@dataclass
class VisualizationStyle:
    """시각화 스타일 설정"""

    # 색상 (BGR 형식)
    landmark_color: Tuple[int, int, int] = (0, 255, 0)  # 녹색
    connection_color: Tuple[int, int, int] = (255, 0, 0)  # 파란색
    bbox_color: Tuple[int, int, int] = (0, 0, 255)  # 빨간색

    # 두께
    landmark_thickness: int = 2
    connection_thickness: int = 1
    bbox_thickness: int = 2

    # 크기
    landmark_radius: int = 2

    # 투명도 (0.0 ~ 1.0)
    connection_alpha: float = 0.5
