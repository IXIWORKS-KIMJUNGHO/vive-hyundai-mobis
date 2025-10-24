"""얼굴 랜드마크 추출 및 관리"""

from typing import List, Optional, Tuple
import mediapipe as mp

from ..models import Landmark, DetectionResult
from ..config.constants import FACIAL_REGIONS
from ..utils.exceptions import LandmarkExtractionError
from ..utils.validators import validate_landmark_index


class LandmarkExtractor:
    """얼굴 랜드마크 추출 및 관리"""

    def __init__(self):
        """초기화"""
        self.face_mesh_connections = mp.solutions.face_mesh.FACEMESH_TESSELATION

    def extract_landmarks(
        self,
        mediapipe_result,
        image_width: int,
        image_height: int
    ) -> List[Landmark]:
        """
        MediaPipe 결과에서 landmark 추출

        Args:
            mediapipe_result: MediaPipe FaceMesh 처리 결과
            image_width: 이미지 너비
            image_height: 이미지 높이

        Returns:
            468개 Landmark 리스트

        Raises:
            LandmarkExtractionError: 추출 실패 시
        """
        if not mediapipe_result or not mediapipe_result.multi_face_landmarks:
            raise LandmarkExtractionError("No face landmarks found in result")

        # 첫 번째 얼굴의 landmarks 추출
        face_landmarks = mediapipe_result.multi_face_landmarks[0]

        landmarks = []
        for landmark in face_landmarks.landmark:
            lm = Landmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=getattr(landmark, 'visibility', 1.0),
                pixel_x=int(landmark.x * image_width),
                pixel_y=int(landmark.y * image_height)
            )
            landmarks.append(lm)

        return landmarks

    def get_landmark_by_index(
        self,
        landmarks: List[Landmark],
        index: int
    ) -> Landmark:
        """
        특정 인덱스의 landmark 반환

        Args:
            landmarks: Landmark 리스트
            index: 랜드마크 인덱스 (0-467)

        Returns:
            해당 인덱스의 Landmark
        """
        validate_landmark_index(index)
        if index >= len(landmarks):
            raise IndexError(f"Landmark index {index} out of range")
        return landmarks[index]

    def get_facial_region(
        self,
        landmarks: List[Landmark],
        region: str
    ) -> List[Landmark]:
        """
        얼굴 영역별 landmark 반환

        Args:
            landmarks: 전체 Landmark 리스트
            region: 영역 이름 (예: 'left_eye', 'nose', 'mouth')

        Returns:
            해당 영역의 Landmark 리스트
        """
        if region not in FACIAL_REGIONS:
            available = ', '.join(FACIAL_REGIONS.keys())
            raise ValueError(f"Unknown region '{region}'. Available: {available}")

        indices = FACIAL_REGIONS[region]
        return [landmarks[i] for i in indices if i < len(landmarks)]

    def get_landmark_connections(self) -> List[Tuple[int, int]]:
        """
        Landmark 연결 정보 반환 (mesh 렌더링용)

        Returns:
            [(start_idx, end_idx), ...] 형식의 연결 정보
        """
        return list(self.face_mesh_connections)
