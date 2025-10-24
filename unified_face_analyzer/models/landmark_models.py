"""데이터 모델 정의"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import numpy as np


class EyeShape(Enum):
    """눈 형태 분류"""
    UPTURNED = "upturned"      # 눈꼬리 올라감
    DOWNTURNED = "downturned"  # 눈꼬리 내려감
    NEUTRAL = "neutral"        # 기본형 (평행)


class FaceShape(Enum):
    """얼굴형 분류 (단순화: OVAL vs ROUND)"""
    OVAL = "oval"              # 계란형 (세로가 긴 타원형)
    ROUND = "round"            # 둥근형 (원형에 가까운, 정사각형에 가까운)


@dataclass
class Landmark:
    """단일 랜드마크 포인트"""

    x: float  # 정규화 x 좌표 (0-1)
    y: float  # 정규화 y 좌표 (0-1)
    z: float  # 깊이 정보 (상대적)
    visibility: float = 1.0  # 가시성 점수 (0-1)

    # 픽셀 좌표 (계산 후 저장)
    pixel_x: Optional[int] = None
    pixel_y: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'visibility': self.visibility,
            'pixel_x': self.pixel_x,
            'pixel_y': self.pixel_y,
        }


@dataclass
class FaceGeometry:
    """얼굴 기하학 정보"""

    # 얼굴 각도 (도 단위)
    pitch: float = 0.0  # 상하 회전 (-90 ~ 90)
    yaw: float = 0.0    # 좌우 회전 (-90 ~ 90)
    roll: float = 0.0   # 기울기 (-180 ~ 180)

    # 크기 및 거리
    face_width: float = 0.0
    face_height: float = 0.0
    estimated_distance: Optional[float] = None


@dataclass
class DetectionResult:
    """얼굴 검출 결과"""

    success: bool
    landmarks: List[Landmark] = field(default_factory=list)
    confidence: float = 0.0
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h)
    face_geometry: Optional[FaceGeometry] = None
    processing_time: float = 0.0  # ms

    def __post_init__(self):
        """검출 성공 시 landmarks 개수 검증"""
        if self.success and len(self.landmarks) not in [0, 468]:
            # 경고: MediaPipe는 468개 landmarks 반환
            pass


@dataclass
class EyeAnalysis:
    """눈 형태 분석 결과"""

    left_eye_shape: EyeShape
    right_eye_shape: EyeShape
    left_eye_angle: float      # 왼쪽 눈꼬리 기울기
    right_eye_angle: float     # 오른쪽 눈꼬리 기울기
    average_eye_angle: float   # 평균 기울기
    overall_eye_shape: EyeShape  # 전체 눈 형태 (평균 기울기 기반)
    confidence: float = 0.95   # 분류 신뢰도

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'left_eye_shape': self.left_eye_shape.value,
            'right_eye_shape': self.right_eye_shape.value,
            'left_eye_angle': round(self.left_eye_angle, 3),
            'right_eye_angle': round(self.right_eye_angle, 3),
            'average_eye_angle': round(self.average_eye_angle, 3),
            'overall_eye_shape': self.overall_eye_shape.value,
            'confidence': self.confidence,
        }


@dataclass
class FaceShapeAnalysis:
    """얼굴형 분석 결과"""

    face_shape: FaceShape
    aspect_ratio: float        # 종횡비 (height/width)
    face_width: float          # 얼굴 너비 (픽셀)
    face_height: float         # 얼굴 높이 (픽셀)
    forehead_width: float = 0.0    # 이마 너비 (픽셀)
    cheekbone_width: float = 0.0   # 광대 너비 (픽셀)
    jawline_width: float = 0.0     # 턱 너비 (픽셀)
    confidence: float = 0.90   # 분류 신뢰도

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'face_shape': self.face_shape.value,
            'aspect_ratio': round(self.aspect_ratio, 3),
            'face_width': round(self.face_width, 1),
            'face_height': round(self.face_height, 1),
            'forehead_width': round(self.forehead_width, 1),
            'cheekbone_width': round(self.cheekbone_width, 1),
            'jawline_width': round(self.jawline_width, 1),
            'confidence': self.confidence,
        }


@dataclass
class DetailedFaceAnalysis:
    """상세 얼굴 분석 결과 (통합)"""

    eye_analysis: Optional[EyeAnalysis] = None
    face_shape_analysis: Optional[FaceShapeAnalysis] = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        result = {}
        if self.eye_analysis:
            result['eye_analysis'] = self.eye_analysis.to_dict()
        if self.face_shape_analysis:
            result['face_shape_analysis'] = self.face_shape_analysis.to_dict()
        return result


@dataclass
class ProcessedResult:
    """프레임 처리 결과"""

    original_image: np.ndarray
    annotated_image: Optional[np.ndarray] = None
    detection_result: Optional[DetectionResult] = None
    face_analysis: Optional[DetailedFaceAnalysis] = None  # 얼굴 분석 추가
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """메타데이터 초기화"""
        if 'timestamp' not in self.metadata:
            import time
            self.metadata['timestamp'] = time.time()
