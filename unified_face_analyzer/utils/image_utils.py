# -*- coding: utf-8 -*-
"""
Image utility functions for Y8 format conversion
"""

import numpy as np
import cv2
from typing import Optional, Tuple
from . import get_logger

logger = get_logger(__name__)


def decode_y8_to_bgr(
    data: bytes,
    width: int = 1280,
    height: int = 800,
    flip_vertical: bool = False,
    auto_detect_resolution: bool = True
) -> Optional[np.ndarray]:
    """
    Raw Y8 (grayscale) 데이터를 BGR 이미지로 변환

    Args:
        data: Raw Y8 바이너리 데이터
        width: 이미지 너비 (기본값: 1280)
        height: 이미지 높이 (기본값: 800)
        flip_vertical: Y축 뒤집기 적용 여부 (기본값: False)
        auto_detect_resolution: 자동 해상도 감지 활성화 (기본값: True)

    Returns:
        numpy array (BGR 포맷) 또는 None

    Note:
        - Y8: 8-bit grayscale format (1 byte per pixel)
        - 일반적인 해상도: 1280x800, 1280x720, 1920x1080, 640x480
    """
    try:
        expected_size = width * height

        # 크기 검증 및 자동 해상도 감지
        if len(data) != expected_size and auto_detect_resolution:
            logger.warning(f"Y8 data size mismatch: expected {expected_size}, got {len(data)}")

            # 일반적인 해상도로 재시도
            common_resolutions = [
                (1280, 800),   # Default
                (1280, 720),   # HD Ready
                (1920, 1080),  # Full HD
                (640, 480),    # VGA
                (800, 600),    # SVGA
                (1024, 768)    # XGA
            ]

            for w, h in common_resolutions:
                if len(data) == w * h:
                    width, height = w, h
                    logger.info(f"Auto-detected resolution: {width}x{height}")
                    break
            else:
                logger.error(f"Cannot determine Y8 image resolution for {len(data)} bytes")
                return None

        # Y8 배열로 변환
        y8_array = np.frombuffer(data, dtype=np.uint8)

        # 2D 배열로 reshape
        y8_image = y8_array.reshape((height, width))

        # Y축 뒤집기 (옵션)
        if flip_vertical:
            y8_image = np.flipud(y8_image)

        # Grayscale → BGR 변환
        bgr_image = cv2.cvtColor(y8_image, cv2.COLOR_GRAY2BGR)

        logger.debug(f"Y8 decoded: {bgr_image.shape}, flip={flip_vertical}")
        return bgr_image

    except Exception as e:
        logger.error(f"Error decoding Y8 to BGR: {e}")
        return None


def get_y8_resolution(data_size: int) -> Optional[Tuple[int, int]]:
    """
    Y8 데이터 크기로부터 해상도 추정

    Args:
        data_size: Y8 데이터 크기 (bytes)

    Returns:
        (width, height) 튜플 또는 None

    Examples:
        >>> get_y8_resolution(1024000)
        (1280, 800)
        >>> get_y8_resolution(921600)
        (1280, 720)
    """
    common_resolutions = {
        1024000: (1280, 800),   # 1280 * 800
        921600: (1280, 720),    # 1280 * 720
        2073600: (1920, 1080),  # 1920 * 1080
        307200: (640, 480),     # 640 * 480
        480000: (800, 600),     # 800 * 600
        786432: (1024, 768)     # 1024 * 768
    }

    return common_resolutions.get(data_size)


def validate_y8_data(data: bytes, width: int, height: int) -> bool:
    """
    Y8 데이터 유효성 검증

    Args:
        data: Y8 바이너리 데이터
        width: 예상 너비
        height: 예상 높이

    Returns:
        유효하면 True, 아니면 False
    """
    expected_size = width * height
    return len(data) == expected_size
