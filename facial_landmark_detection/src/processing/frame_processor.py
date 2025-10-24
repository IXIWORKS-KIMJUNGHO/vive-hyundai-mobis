"""프레임 처리 파이프라인"""

import time
import cv2
import numpy as np
from typing import Generator, List, Optional
from pathlib import Path

from ..core.face_detector import FaceDetector
from ..core.landmark_extractor import LandmarkExtractor
from ..models import ProcessedResult, DetectionResult
from ..utils.exceptions import InvalidImageError
from ..utils.validators import validate_image
from .result_handler import ResultHandler
from .geometry import GeometryCalculator


class FrameProcessor:
    """프레임 처리 파이프라인"""

    def __init__(
        self,
        detector: FaceDetector,
        extractor: LandmarkExtractor,
        enable_smoothing: bool = False
    ):
        """
        초기화

        Args:
            detector: FaceDetector 인스턴스
            extractor: LandmarkExtractor 인스턴스
            enable_smoothing: Temporal smoothing 활성화 여부
        """
        self.detector = detector
        self.extractor = extractor
        self.result_handler = ResultHandler() if enable_smoothing else None
        self.geometry_calculator = GeometryCalculator()

    def process_image(self, image_path: str) -> ProcessedResult:
        """
        단일 이미지 처리

        Args:
            image_path: 이미지 파일 경로

        Returns:
            ProcessedResult: 처리 결과

        Raises:
            InvalidImageError: 이미지 로드 실패
        """
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            raise InvalidImageError(f"Failed to load image: {image_path}")

        validate_image(image)

        # 얼굴 검출
        detection_result = self.detector.detect(image)

        # 기하학 정보 추가
        if detection_result.success and self.detector.config.enable_face_geometry:
            detection_result.face_geometry = self.geometry_calculator.get_face_geometry(
                detection_result.landmarks
            )

        return ProcessedResult(
            original_image=image,
            detection_result=detection_result,
            metadata={
                'image_path': str(image_path),
                'timestamp': time.time(),
                'image_shape': image.shape
            }
        )

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None
    ) -> Generator[ProcessedResult, None, None]:
        """
        비디오 처리 (제너레이터)

        Args:
            video_path: 비디오 파일 경로
            output_path: 출력 비디오 경로 (선택적)

        Yields:
            ProcessedResult: 각 프레임의 처리 결과
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise InvalidImageError(f"Failed to open video: {video_path}")

        frame_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 프레임 처리
                detection_result = self.detector.detect(frame)

                # Temporal smoothing 적용
                if detection_result.success and self.result_handler:
                    detection_result.landmarks = self.result_handler.apply_temporal_smoothing(
                        detection_result.landmarks
                    )

                # 기하학 정보 추가
                if detection_result.success and self.detector.config.enable_face_geometry:
                    detection_result.face_geometry = self.geometry_calculator.get_face_geometry(
                        detection_result.landmarks
                    )

                yield ProcessedResult(
                    original_image=frame,
                    detection_result=detection_result,
                    metadata={
                        'video_path': str(video_path),
                        'frame_number': frame_count,
                        'timestamp': time.time()
                    }
                )

                frame_count += 1

        finally:
            cap.release()

    def process_realtime(
        self,
        camera_id: int = 0,
        display: bool = True,
        max_frames: Optional[int] = None
    ) -> Generator[ProcessedResult, None, None]:
        """
        실시간 카메라 처리

        Args:
            camera_id: 카메라 디바이스 ID
            display: 결과 화면 표시 여부
            max_frames: 최대 처리 프레임 수 (None이면 무한)

        Yields:
            ProcessedResult: 각 프레임의 처리 결과
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise InvalidImageError(f"Failed to open camera {camera_id}")

        frame_count = 0

        try:
            while cap.isOpened():
                if max_frames and frame_count >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # 프레임 처리
                detection_result = self.detector.detect(frame)

                # Temporal smoothing
                if detection_result.success and self.result_handler:
                    detection_result.landmarks = self.result_handler.apply_temporal_smoothing(
                        detection_result.landmarks
                    )

                # 기하학 정보
                if detection_result.success and self.detector.config.enable_face_geometry:
                    detection_result.face_geometry = self.geometry_calculator.get_face_geometry(
                        detection_result.landmarks
                    )

                result = ProcessedResult(
                    original_image=frame,
                    detection_result=detection_result,
                    metadata={
                        'camera_id': camera_id,
                        'frame_number': frame_count,
                        'timestamp': time.time()
                    }
                )

                # 화면 표시
                if display:
                    display_frame = frame.copy()
                    if detection_result.success:
                        # 간단한 시각화 (bbox)
                        x, y, w, h = detection_result.bounding_box
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    cv2.imshow('Facial Landmark Detection', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                yield result
                frame_count += 1

        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

    def process_batch(self, image_paths: List[str]) -> List[ProcessedResult]:
        """
        배치 이미지 처리

        Args:
            image_paths: 이미지 경로 리스트

        Returns:
            처리 결과 리스트
        """
        return [self.process_image(path) for path in image_paths]
