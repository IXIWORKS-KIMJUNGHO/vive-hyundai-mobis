"""얼굴 랜드마크 인덱스 및 시스템 상수 정의"""

from typing import Dict, List

# MediaPipe FaceMesh 468 landmarks 영역별 인덱스
FACIAL_REGIONS: Dict[str, List[int]] = {
    # 얼굴 윤곽 (Face Oval)
    'face_oval': list(range(10, 338)),

    # 왼쪽 눈 (Left Eye)
    'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173,
                 157, 158, 159, 160, 161, 246],
    'left_eyebrow': [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],

    # 오른쪽 눈 (Right Eye)
    'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466,
                  388, 387, 386, 385, 384, 398],
    'right_eyebrow': [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],

    # 코 (Nose)
    'nose_bridge': [6, 197, 195, 5],
    'nose_tip': [4, 1, 19, 94, 2],
    'nostrils': [98, 97, 2, 326, 327],

    # 입 (Mouth)
    'lips_outer': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                   291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
    'lips_inner': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                   308, 324, 318, 402, 317, 14, 87, 178, 88, 95],

    # 턱 (Chin)
    'chin': [152, 175, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 32],

    # 이마 (Forehead)
    'forehead': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
}

# 중요 랜드마크 포인트
IMPORTANT_LANDMARKS: Dict[str, int] = {
    'left_eye_center': 468,
    'right_eye_center': 473,
    'nose_tip': 4,
    'mouth_center': 13,
    'left_ear_tragion': 234,
    'right_ear_tragion': 454,
}

# 눈 형태 분석용 주요 포인트
EYE_LANDMARKS: Dict[str, Dict[str, int]] = {
    'left_eye': {
        'inner_corner': 33,    # 내안각 (FIXED: 133 → 33)
        'outer_corner': 133,   # 외안각 (FIXED: 33 → 133)
        'top_center': 159,     # 상단 중앙
        'bottom_center': 145,  # 하단 중앙
        'top_outer': 46,       # 상단 외측
        'bottom_outer': 130,   # 하단 외측
    },
    'right_eye': {
        'inner_corner': 362,   # 내안각
        'outer_corner': 263,   # 외안각
        'top_center': 386,     # 상단 중앙
        'bottom_center': 374,  # 하단 중앙
        'top_outer': 276,      # 상단 외측
        'bottom_outer': 359,   # 하단 외측
    }
}

# 얼굴형 분석용 주요 포인트
FACE_SHAPE_LANDMARKS: Dict[str, int] = {
    # 세로 측정 (얼굴 높이)
    'forehead_top': 10,        # 이마 상단
    'chin_bottom': 152,        # 턱 끝

    # 가로 측정 (얼굴 너비 - 3단계)
    'forehead_left': 21,       # 이마 왼쪽
    'forehead_right': 251,     # 이마 오른쪽

    'cheekbone_left': 234,     # 광대뼈 왼쪽
    'cheekbone_right': 454,    # 광대뼈 오른쪽

    'jawline_left': 172,       # 턱선 왼쪽
    'jawline_right': 397,      # 턱선 오른쪽

    # 추가 참조 포인트
    'nose_bridge_top': 6,      # 코 다리 상단
    'left_temple': 127,        # 왼쪽 관자놀이
    'right_temple': 356,       # 오른쪽 관자놀이
}

# MediaPipe FaceMesh 연결선 (mesh 렌더링용)
# Note: MediaPipe에서 제공하는 FACEMESH_TESSELATION 사용 가능
FACE_CONNECTIONS = None  # mediapipe.solutions.face_mesh.FACEMESH_TESSELATION

# 시스템 상수
DEFAULT_IMAGE_SIZE = 640
MAX_NUM_FACES = 5
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
