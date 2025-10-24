"""얼굴 특징 분석 (눈 형태, 얼굴형 분류)"""

import math
from typing import List, Optional, Tuple
import numpy as np

from src.models import (
    Landmark,
    EyeShape,
    FaceShape,
    EyeAnalysis,
    FaceShapeAnalysis,
    DetailedFaceAnalysis,
)
from src.config.constants import EYE_LANDMARKS, FACE_SHAPE_LANDMARKS


class FaceAnalyzer:
    """
    얼굴 특징 분석 클래스

    기능:
    - 눈 형태 분류 (눈꼬리 올라감/내려감/기본)
    - 얼굴형 분류 (계란형/둥근형/사각형/하트형/긴형)
    """

    def __init__(
        self,
        eye_angle_threshold: float = 5.0,      # 눈 각도 분류 임계값 (도)
        aspect_ratio_thresholds: Optional[dict] = None,  # 얼굴형 분류 임계값
    ):
        """
        FaceAnalyzer 초기화

        Args:
            eye_angle_threshold: 눈꼬리 각도 분류 기준 (기본: 5도)
            aspect_ratio_thresholds: 얼굴형 분류 종횡비 임계값 딕셔너리
        """
        self.eye_angle_threshold = eye_angle_threshold

        # 얼굴형 분류 기준 (종횡비 기반, OVAL vs ROUND만 구분)
        self.aspect_ratio_thresholds = aspect_ratio_thresholds or {
            'oval': 1.16,      # 1.16 이상: 계란형 (세로가 긴 타원형)
            # 1.16 미만: 둥근형 (원형/정사각형에 가까운)
        }

    def _calculate_distance(self, p1: Landmark, p2: Landmark) -> float:
        """
        두 랜드마크 사이의 유클리드 거리 계산

        Args:
            p1: 첫 번째 랜드마크
            p2: 두 번째 랜드마크

        Returns:
            float: 거리 (픽셀 좌표 기준)
        """
        if p1.pixel_x is None or p2.pixel_x is None:
            # 정규화 좌표 사용
            dx = p2.x - p1.x
            dy = p2.y - p1.y
        else:
            # 픽셀 좌표 사용
            dx = p2.pixel_x - p1.pixel_x
            dy = p2.pixel_y - p1.pixel_y

        return math.sqrt(dx**2 + dy**2)

    def _calculate_angle(self, p1: Landmark, p2: Landmark, roll_correction: float = 0.0) -> float:
        """
        두 랜드마크를 연결하는 선의 수평선 대비 각도 계산

        Args:
            p1: 시작 랜드마크
            p2: 끝 랜드마크
            roll_correction: 얼굴 기울기 보정 각도 (도)

        Returns:
            float: 각도 (도 단위, -180 ~ 180)
        """
        if p1.pixel_x is None or p2.pixel_x is None:
            # 정규화 좌표 사용
            dx = p2.x - p1.x
            dy = p2.y - p1.y
        else:
            # 픽셀 좌표 사용
            dx = p2.pixel_x - p1.pixel_x
            dy = p2.pixel_y - p1.pixel_y

        # arctan2로 각도 계산 (라디안)
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        # 얼굴 기울기 보정
        angle_deg -= roll_correction

        # -180 ~ 180 범위로 정규화
        if angle_deg > 180:
            angle_deg -= 360
        elif angle_deg < -180:
            angle_deg += 360

        return angle_deg

    def _analyze_eye_shape_simple(
        self,
        eye_landmarks: dict,
        landmarks: List[Landmark],
        roll_angle: float = 0.0
    ) -> float:
        """
        간단한 눈 형태 분석: 내안각→외안각 기울기 + Roll 보정

        Args:
            eye_landmarks: 눈 landmark 인덱스 딕셔너리
            landmarks: 전체 얼굴 landmarks
            roll_angle: 얼굴 Roll 각도 (도 단위, 좌우 기울기 보정용)

        Returns:
            float: Roll 보정된 기울기
        """
        # 내안각과 외안각만 사용
        inner = landmarks[eye_landmarks['inner_corner']]
        outer = landmarks[eye_landmarks['outer_corner']]

        # 순수 기울기 계산: slope = dy / dx
        dx = outer.x - inner.x
        dy = outer.y - inner.y

        if dx == 0:
            return 0.0  # 수직선인 경우 0 반환

        slope = dy / dx

        # Roll 각도 보정 적용
        # 얼굴이 기울어져 있으면 눈도 함께 기울어지므로 보정 필요
        roll_rad = math.radians(roll_angle)
        roll_correction = math.tan(roll_rad)

        # 보정된 기울기 = 측정 기울기 - Roll 기울기
        corrected_slope = slope - roll_correction

        return corrected_slope

    def analyze_eye_shape(
        self,
        landmarks: List[Landmark],
        roll_angle: float = 0.0,
        yaw_angle: float = 0.0
    ) -> EyeAnalysis:
        """
        눈 형태 분류 (눈꼬리 올라감/내려감/기본) - Roll/Yaw 보정 적용

        Args:
            landmarks: 468개 얼굴 landmarks
            roll_angle: 얼굴 Roll 각도 (도, 좌우 기울기 보정)
            yaw_angle: 얼굴 Yaw 각도 (도, 좌우 방향 신뢰도)

        Returns:
            EyeAnalysis: 눈 형태 분석 결과
        """
        # 왼쪽 눈 분석 (내안각→외안각 기울기)
        left_eye_angle = self._analyze_eye_shape_simple(
            EYE_LANDMARKS['left_eye'],
            landmarks,
            roll_angle
        )

        # 오른쪽 눈 분석 (내안각→외안각 기울기)
        right_eye_angle = self._analyze_eye_shape_simple(
            EYE_LANDMARKS['right_eye'],
            landmarks,
            roll_angle
        )

        # Yaw 각도에 따른 신뢰도 계산
        # Yaw > 0: 오른쪽으로 돌림 → 왼쪽 눈 가려짐
        # Yaw < 0: 왼쪽으로 돌림 → 오른쪽 눈 가려짐
        left_confidence = 1.0 - max(0, yaw_angle / 45.0)  # Yaw가 양수면 왼쪽 신뢰도 하락
        right_confidence = 1.0 - max(0, -yaw_angle / 45.0)  # Yaw가 음수면 오른쪽 신뢰도 하락

        # 신뢰도가 0.3 이하면 해당 눈 분석 결과 가중치 감소
        if left_confidence < 0.3:
            left_weight = 0.3
        else:
            left_weight = 1.0

        if right_confidence < 0.3:
            right_weight = 0.3
        else:
            right_weight = 1.0

        # 눈 형태 분류
        # 왼쪽 눈: 양수(올라감) = UPTURNED, 음수(내려감) = DOWNTURNED
        # 오른쪽 눈: 음수(올라감) = UPTURNED, 양수(내려감) = DOWNTURNED (방향 반대!)

        def classify_left_eye(angle: float) -> EyeShape:
            """왼쪽 눈 형태 분류"""
            if angle > 0:
                return EyeShape.UPTURNED
            elif angle < 0:
                return EyeShape.DOWNTURNED
            else:
                return EyeShape.NEUTRAL

        def classify_right_eye(angle: float) -> EyeShape:
            """오른쪽 눈 형태 분류 (방향 반대)"""
            if angle < 0:  # 음수일 때 올라감
                return EyeShape.UPTURNED
            elif angle > 0:  # 양수일 때 내려감
                return EyeShape.DOWNTURNED
            else:
                return EyeShape.NEUTRAL

        left_eye_shape = classify_left_eye(left_eye_angle)
        right_eye_shape = classify_right_eye(right_eye_angle)

        # 평균 기울기: 절대값의 가중 평균 사용 (Yaw 신뢰도 반영)
        left_abs = abs(left_eye_angle)
        right_abs = abs(right_eye_angle)

        # 가중 평균 계산
        total_weight = left_weight + right_weight
        avg_angle = (left_abs * left_weight + right_abs * right_weight) / total_weight

        # 전체 눈 형태 분류 (평균 기울기 기반)
        def classify_overall_shape(avg_slope: float) -> EyeShape:
            """평균 기울기로 전체 눈 형태 분류"""
            if avg_slope < 0.05:
                return EyeShape.DOWNTURNED  # 평균 기울기 < 0.05
            elif avg_slope <= 0.1:
                return EyeShape.NEUTRAL     # 0.05 <= 평균 기울기 <= 0.1
            else:
                return EyeShape.UPTURNED    # 평균 기울기 > 0.1

        overall_shape = classify_overall_shape(avg_angle)

        # 전체 신뢰도: 양쪽 눈 신뢰도의 평균
        overall_confidence = (left_confidence + right_confidence) / 2.0
        # 최소 신뢰도 0.7 보장
        overall_confidence = max(0.7, overall_confidence)

        return EyeAnalysis(
            left_eye_shape=left_eye_shape,
            right_eye_shape=right_eye_shape,
            left_eye_angle=left_eye_angle,
            right_eye_angle=right_eye_angle,
            average_eye_angle=avg_angle,
            overall_eye_shape=overall_shape,
            confidence=overall_confidence,
        )

    def analyze_face_shape(
        self,
        landmarks: List[Landmark]
    ) -> FaceShapeAnalysis:
        """
        얼굴형 분류 (계란형/둥근형/사각형/하트형/긴형)

        Args:
            landmarks: 468개 얼굴 landmarks

        Returns:
            FaceShapeAnalysis: 얼굴형 분석 결과
        """
        # 1. 얼굴 높이 계산
        forehead_top = landmarks[FACE_SHAPE_LANDMARKS['forehead_top']]
        chin_bottom = landmarks[FACE_SHAPE_LANDMARKS['chin_bottom']]
        face_height = self._calculate_distance(forehead_top, chin_bottom)

        # 2. 얼굴 너비 계산 (3단계)
        # 2-1. 이마 너비
        forehead_left = landmarks[FACE_SHAPE_LANDMARKS['forehead_left']]
        forehead_right = landmarks[FACE_SHAPE_LANDMARKS['forehead_right']]
        forehead_width = self._calculate_distance(forehead_left, forehead_right)

        # 2-2. 광대뼈 너비 (가장 넓은 부분)
        cheekbone_left = landmarks[FACE_SHAPE_LANDMARKS['cheekbone_left']]
        cheekbone_right = landmarks[FACE_SHAPE_LANDMARKS['cheekbone_right']]
        cheekbone_width = self._calculate_distance(cheekbone_left, cheekbone_right)

        # 2-3. 턱선 너비
        jawline_left = landmarks[FACE_SHAPE_LANDMARKS['jawline_left']]
        jawline_right = landmarks[FACE_SHAPE_LANDMARKS['jawline_right']]
        jawline_width = self._calculate_distance(jawline_left, jawline_right)

        # 3. 대표 얼굴 너비 (광대뼈가 가장 넓은 부분)
        face_width = max(forehead_width, cheekbone_width, jawline_width)

        # 4. 종횡비 계산 (세로/가로)
        aspect_ratio = face_height / face_width if face_width > 0 else 0.0

        # 5. 얼굴형 분류
        face_shape = self._classify_face_shape(
            aspect_ratio,
            forehead_width,
            cheekbone_width,
            jawline_width
        )

        return FaceShapeAnalysis(
            face_shape=face_shape,
            aspect_ratio=aspect_ratio,
            face_width=face_width,
            face_height=face_height,
            forehead_width=forehead_width,
            cheekbone_width=cheekbone_width,
            jawline_width=jawline_width,
            confidence=0.90,  # 기하학적 분석이므로 높은 신뢰도
        )

    def _classify_face_shape(
        self,
        aspect_ratio: float,
        forehead_width: float,
        cheekbone_width: float,
        jawline_width: float
    ) -> FaceShape:
        """
        종횡비 기반 얼굴형 분류 (단순화: OVAL vs ROUND)

        Args:
            aspect_ratio: 얼굴 종횡비 (height/width)
            forehead_width: 이마 너비 (미사용, 호환성 유지)
            cheekbone_width: 광대뼈 너비 (미사용, 호환성 유지)
            jawline_width: 턱선 너비 (미사용, 호환성 유지)

        Returns:
            FaceShape: 분류된 얼굴형 (OVAL 또는 ROUND)
        """
        # 순수 종횡비 기반 단순 분류
        if aspect_ratio >= self.aspect_ratio_thresholds['oval']:
            return FaceShape.OVAL  # 계란형 (세로가 긴 타원형)
        else:
            return FaceShape.ROUND  # 둥근형 (원형/정사각형에 가까운)

    def get_detailed_analysis(
        self,
        landmarks: List[Landmark],
        roll_angle: float = 0.0,
        yaw_angle: float = 0.0
    ) -> DetailedFaceAnalysis:
        """
        통합 얼굴 분석 수행 (Roll/Yaw 보정 적용)

        Args:
            landmarks: 468개 얼굴 landmarks
            roll_angle: 얼굴 Roll 각도 (도, 좌우 기울기)
            yaw_angle: 얼굴 Yaw 각도 (도, 좌우 방향)

        Returns:
            DetailedFaceAnalysis: 눈 형태 + 얼굴형 통합 분석 결과
        """
        eye_analysis = self.analyze_eye_shape(landmarks, roll_angle, yaw_angle)
        face_shape_analysis = self.analyze_face_shape(landmarks)

        return DetailedFaceAnalysis(
            eye_analysis=eye_analysis,
            face_shape_analysis=face_shape_analysis
        )
