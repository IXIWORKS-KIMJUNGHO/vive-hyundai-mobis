"""
MediaPipe 468-point to dlib 68-point landmark mapping.

This module provides conversion between MediaPipe Face Mesh (468 points)
and dlib facial landmarks (68 points) for backward compatibility.
"""

import numpy as np
from typing import List, Tuple

# MediaPipe → dlib 68점 매핑 테이블
MEDIAPIPE_TO_DLIB_68 = {
    # 얼굴 윤곽 (Jaw line) 0-16
    0: 152,   # 턱 중앙
    1: 234,
    2: 93,
    3: 132,
    4: 58,
    5: 172,
    6: 136,
    7: 150,
    8: 149,   # 턱 끝
    9: 176,
    10: 148,
    11: 152,
    12: 377,
    13: 400,
    14: 378,
    15: 379,
    16: 365,

    # 눈썹 (Eyebrows) 17-26 - 헤어스타일 분석의 핵심
    17: 70,   # 왼쪽 눈썹 시작
    18: 63,
    19: 105,
    20: 66,
    21: 107,  # 왼쪽 눈썹 끝
    22: 336,  # 오른쪽 눈썹 시작
    23: 296,
    24: 334,
    25: 293,
    26: 300,  # 오른쪽 눈썹 끝

    # 코 (Nose) 27-35
    27: 168,  # 코 브릿지 상단
    28: 6,
    29: 197,
    30: 195,  # 코 끝
    31: 5,
    32: 4,
    33: 1,
    34: 19,
    35: 94,

    # 눈 (Eyes) 36-47
    36: 33,   # 왼쪽 눈 외측
    37: 160,
    38: 159,
    39: 158,  # 왼쪽 눈 내측
    40: 133,
    41: 153,
    42: 263,  # 오른쪽 눈 내측
    43: 387,
    44: 386,
    45: 385,  # 오른쪽 눈 외측
    46: 362,
    47: 380,

    # 입 (Mouth) 48-67
    48: 61,   # 입 외곽 시작
    49: 146,
    50: 91,
    51: 181,
    52: 84,
    53: 17,
    54: 314,
    55: 405,
    56: 321,
    57: 375,
    58: 291,
    59: 409,  # 입 외곽 끝
    60: 78,   # 입 내부 시작
    61: 95,
    62: 88,
    63: 178,
    64: 87,
    65: 14,
    66: 317,
    67: 402   # 입 내부 끝
}

# 헤어스타일 분석에 중요한 눈썹 인덱스 (MediaPipe 468점)
MEDIAPIPE_EYEBROW_INDICES = {
    'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    'right_eyebrow': [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    'left_eyebrow_upper': [107, 66, 105, 63, 70],  # 상단 라인 (이마 경계)
    'right_eyebrow_upper': [336, 296, 334, 293, 300]  # 상단 라인 (이마 경계)
}


def convert_mediapipe_to_dlib68(mediapipe_landmarks, img_width: int, img_height: int) -> np.ndarray:
    """
    MediaPipe 468점 랜드마크를 dlib 68점 형식으로 변환.

    Args:
        mediapipe_landmarks: MediaPipe face_landmarks.landmark (정규화 좌표 0.0-1.0)
        img_width: 이미지 너비 (픽셀)
        img_height: 이미지 높이 (픽셀)

    Returns:
        np.ndarray: (68, 2) shape의 dlib 형식 랜드마크 좌표 (x, y)
    """
    dlib_points = []

    for dlib_idx in range(68):
        mp_idx = MEDIAPIPE_TO_DLIB_68[dlib_idx]
        lm = mediapipe_landmarks[mp_idx]

        # 정규화 좌표 → 픽셀 좌표 변환
        x = int(lm.x * img_width)
        y = int(lm.y * img_height)
        dlib_points.append([x, y])

    return np.array(dlib_points)


def get_eyebrow_points(mediapipe_landmarks, img_width: int, img_height: int) -> np.ndarray:
    """
    MediaPipe 468점에서 눈썹 좌표만 추출 (헤어스타일 분석용).

    dlib 17-26번 인덱스에 해당하는 10개 점을 반환.

    Args:
        mediapipe_landmarks: MediaPipe face_landmarks.landmark
        img_width: 이미지 너비
        img_height: 이미지 높이

    Returns:
        np.ndarray: (10, 2) shape의 눈썹 좌표 (x, y)
    """
    eyebrow_indices = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]  # dlib 눈썹 인덱스
    eyebrow_points = []

    for dlib_idx in eyebrow_indices:
        mp_idx = MEDIAPIPE_TO_DLIB_68[dlib_idx]
        lm = mediapipe_landmarks[mp_idx]

        x = int(lm.x * img_width)
        y = int(lm.y * img_height)
        eyebrow_points.append([x, y])

    return np.array(eyebrow_points)


def get_face_bbox(mediapipe_landmarks, img_width: int, img_height: int) -> dict:
    """
    MediaPipe 랜드마크에서 얼굴 바운딩 박스 계산.

    Args:
        mediapipe_landmarks: MediaPipe face_landmarks.landmark
        img_width: 이미지 너비
        img_height: 이미지 높이

    Returns:
        dict: {'x_min', 'y_min', 'x_max', 'y_max', 'width', 'height'}
    """
    x_coords = [lm.x * img_width for lm in mediapipe_landmarks]
    y_coords = [lm.y * img_height for lm in mediapipe_landmarks]

    x_min = int(min(x_coords))
    y_min = int(min(y_coords))
    x_max = int(max(x_coords))
    y_max = int(max(y_coords))

    return {
        'x_min': x_min,
        'y_min': y_min,
        'x_max': x_max,
        'y_max': y_max,
        'width': x_max - x_min,
        'height': y_max - y_min
    }
