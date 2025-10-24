"""얼굴 랜드마크 인덱스 및 시스템 상수 정의"""

from typing import Dict, List

# 눈 형태 분석용 주요 포인트
EYE_LANDMARKS: Dict[str, Dict[str, int]] = {
    'left_eye': {
        'inner_corner': 33,    # 내안각
        'outer_corner': 133,   # 외안각
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
