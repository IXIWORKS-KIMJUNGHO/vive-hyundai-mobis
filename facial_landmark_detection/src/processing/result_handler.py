"""검출 결과 후처리 및 검증"""

from typing import List, Optional
import numpy as np
from collections import deque

from ..models import DetectionResult, Landmark


class ResultHandler:
    """검출 결과 후처리 및 검증"""

    def __init__(self, smoothing_window: int = 5):
        """
        초기화

        Args:
            smoothing_window: Temporal smoothing을 위한 윈도우 크기
        """
        self.smoothing_window = smoothing_window
        self.landmark_history = deque(maxlen=smoothing_window)

    def validate_result(self, result: DetectionResult) -> bool:
        """
        검출 결과 유효성 검증

        Args:
            result: 검증할 검출 결과

        Returns:
            유효 여부
        """
        if not result.success:
            return False

        if len(result.landmarks) != 468:
            return False

        # 모든 landmark가 유효한 범위 내에 있는지 확인
        for lm in result.landmarks:
            if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
                return False
            if lm.visibility < 0.0:
                return False

        return True

    def filter_low_confidence_landmarks(
        self,
        landmarks: List[Landmark],
        threshold: float = 0.5
    ) -> List[Landmark]:
        """
        낮은 신뢰도의 landmark 필터링

        Args:
            landmarks: Landmark 리스트
            threshold: 가시성 임계값

        Returns:
            필터링된 Landmark 리스트
        """
        return [lm for lm in landmarks if lm.visibility >= threshold]

    def apply_temporal_smoothing(
        self,
        current_landmarks: List[Landmark]
    ) -> List[Landmark]:
        """
        Temporal smoothing 적용 (비디오 처리용)

        Args:
            current_landmarks: 현재 프레임의 landmarks

        Returns:
            평활화된 landmarks
        """
        self.landmark_history.append(current_landmarks)

        if len(self.landmark_history) < 2:
            return current_landmarks

        # 이동 평균 계산
        smoothed = []
        num_landmarks = len(current_landmarks)

        for i in range(num_landmarks):
            avg_x = np.mean([frame[i].x for frame in self.landmark_history])
            avg_y = np.mean([frame[i].y for frame in self.landmark_history])
            avg_z = np.mean([frame[i].z for frame in self.landmark_history])

            smoothed_lm = Landmark(
                x=float(avg_x),
                y=float(avg_y),
                z=float(avg_z),
                visibility=current_landmarks[i].visibility,
                pixel_x=current_landmarks[i].pixel_x,
                pixel_y=current_landmarks[i].pixel_y
            )
            smoothed.append(smoothed_lm)

        return smoothed

    def reset_history(self):
        """히스토리 초기화"""
        self.landmark_history.clear()
