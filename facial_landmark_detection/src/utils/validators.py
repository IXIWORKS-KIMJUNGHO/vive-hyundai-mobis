"""입력 검증 유틸리티 함수"""

import numpy as np
from .exceptions import InvalidImageError


def validate_image(image: np.ndarray) -> None:
    """
    이미지 유효성 검증

    Args:
        image: 검증할 이미지 (numpy array)

    Raises:
        InvalidImageError: 이미지가 유효하지 않은 경우
    """
    if image is None:
        raise InvalidImageError("Image is None")

    if not isinstance(image, np.ndarray):
        raise InvalidImageError(f"Image must be numpy.ndarray, got {type(image)}")

    if image.size == 0:
        raise InvalidImageError("Image is empty")

    if len(image.shape) not in [2, 3]:
        raise InvalidImageError(f"Image must be 2D or 3D, got shape {image.shape}")

    if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
        raise InvalidImageError(f"Image channels must be 1, 3, or 4, got {image.shape[2]}")


def validate_confidence(confidence: float, param_name: str = "confidence") -> None:
    """신뢰도 값 검증 (0.0 ~ 1.0)"""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{param_name} must be between 0.0 and 1.0, got {confidence}")


def validate_landmark_index(index: int, max_landmarks: int = 468) -> None:
    """랜드마크 인덱스 검증"""
    if not 0 <= index < max_landmarks:
        raise ValueError(f"Landmark index must be between 0 and {max_landmarks-1}, got {index}")
