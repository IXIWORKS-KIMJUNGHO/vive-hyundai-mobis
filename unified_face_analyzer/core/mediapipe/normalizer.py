"""좌표 정규화 유틸리티"""

import numpy as np
from typing import List, Tuple
from ..models import Landmark


class CoordinateNormalizer:
    """좌표계 변환 및 정규화"""

    @staticmethod
    def normalize_coordinates(
        landmarks: List[Tuple[float, float, float]],
        image_width: int,
        image_height: int
    ) -> List[Landmark]:
        """
        픽셀 좌표를 정규화 좌표로 변환

        Args:
            landmarks: [(x_pixel, y_pixel, z), ...] 형식의 좌표 리스트
            image_width: 이미지 너비
            image_height: 이미지 높이

        Returns:
            정규화된 Landmark 객체 리스트
        """
        normalized = []
        for x, y, z in landmarks:
            landmark = Landmark(
                x=x / image_width,
                y=y / image_height,
                z=z,
                visibility=1.0
            )
            normalized.append(landmark)
        return normalized

    @staticmethod
    def denormalize_coordinates(
        landmarks: List[Landmark],
        image_width: int,
        image_height: int
    ) -> List[Landmark]:
        """
        정규화 좌표를 픽셀 좌표로 변환

        Args:
            landmarks: 정규화된 Landmark 리스트
            image_width: 이미지 너비
            image_height: 이미지 높이

        Returns:
            픽셀 좌표가 설정된 Landmark 리스트
        """
        for landmark in landmarks:
            landmark.pixel_x = int(landmark.x * image_width)
            landmark.pixel_y = int(landmark.y * image_height)
        return landmarks

    @staticmethod
    def get_bounding_box(landmarks: List[Landmark]) -> Tuple[int, int, int, int]:
        """
        랜드마크로부터 bounding box 계산

        Args:
            landmarks: Landmark 리스트

        Returns:
            (x, y, width, height) 튜플
        """
        if not landmarks or landmarks[0].pixel_x is None:
            return (0, 0, 0, 0)

        x_coords = [lm.pixel_x for lm in landmarks if lm.pixel_x is not None]
        y_coords = [lm.pixel_y for lm in landmarks if lm.pixel_y is not None]

        if not x_coords or not y_coords:
            return (0, 0, 0, 0)

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        return (x_min, y_min, x_max - x_min, y_max - y_min)
