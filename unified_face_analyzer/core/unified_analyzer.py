"""
Unified Face Analyzer
MediaPipe + BiSeNet + CLIP + dlib 통합 얼굴 분석 시스템
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from core.mediapipe import FaceDetector as MediaPipeDetector, FaceAnalyzer
from core.hairstyle_analyzer import HairstyleAnalyzer
from models.landmark_models import DetectionResult, FaceGeometry, DetailedFaceAnalysis
from utils import get_config, get_logger

logger = get_logger(__name__)


class UnifiedFaceAnalyzer:
    """
    통합 얼굴 분석 시스템

    Features:
    - MediaPipe: 468개 랜드마크 + 얼굴 각도 (pitch, yaw, roll)
    - HairstyleAnalyzer: BiSeNet + CLIP + dlib 68점 기반 헤어스타일 분석
    - 통합 결과 JSON 생성
    """

    def __init__(self):
        """초기화"""
        self.config = get_config()

        # 통합 분석 설정
        self.unified_config = self.config.unified_analysis

        # MediaPipe 초기화 (선택적)
        self.mediapipe_detector = None
        self.face_analyzer = None
        if self.unified_config.enable_mediapipe:
            try:
                self.mediapipe_detector = MediaPipeDetector()
                self.face_analyzer = FaceAnalyzer()
                logger.info("MediaPipe detector and FaceAnalyzer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MediaPipe: {e}")

        # HairstyleAnalyzer 초기화 (선택적)
        self.hairstyle_analyzer = None
        if self.unified_config.enable_hairstyle:
            try:
                self.hairstyle_analyzer = HairstyleAnalyzer()
                logger.info("Hairstyle analyzer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize HairstyleAnalyzer: {e}")

        logger.info("UnifiedFaceAnalyzer initialized successfully")

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        통합 이미지 분석

        Args:
            image_path: 분석할 이미지 경로

        Returns:
            통합 분석 결과 딕셔너리:
            {
                'success': bool,
                'mediapipe': {
                    'success': bool,
                    'landmarks_count': int,
                    'face_geometry': {
                        'pitch': float,
                        'yaw': float,
                        'roll': float,
                        'face_width': float,
                        'face_height': float
                    },
                    'processing_time_ms': float
                },
                'hairstyle': {
                    'classification': str,
                    'clip_results': {...},
                    'geometric_analysis': {...}
                },
                'metadata': {
                    'image_path': str,
                    'total_processing_time_ms': float,
                    'timestamp': str,
                    'enabled_modules': List[str]
                }
            }
        """
        start_time = time.time()

        # 이미지 로드
        if not Path(image_path).exists():
            logger.error(f"Image not found: {image_path}")
            return {
                'success': False,
                'error': f'Image not found: {image_path}'
            }

        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return {
                'success': False,
                'error': f'Failed to load image: {image_path}'
            }

        result = {
            'success': True,
            'metadata': {
                'image_path': image_path,
                'enabled_modules': []
            }
        }

        # MediaPipe 분석
        if self.mediapipe_detector and self.unified_config.enable_mediapipe:
            try:
                mediapipe_result = self._analyze_with_mediapipe(image)
                result['mediapipe'] = mediapipe_result
                result['metadata']['enabled_modules'].append('mediapipe')
                logger.info(f"MediaPipe analysis completed: {mediapipe_result['success']}")
            except Exception as e:
                logger.error(f"MediaPipe analysis failed: {e}", exc_info=True)
                result['mediapipe'] = {'success': False, 'error': str(e)}

        # Hairstyle 분석
        if self.hairstyle_analyzer and self.unified_config.enable_hairstyle:
            try:
                hairstyle_result, viz_image = self.hairstyle_analyzer.analyze_image(image_path)
                result['hairstyle'] = hairstyle_result
                result['visualization_image'] = viz_image  # numpy array
                result['metadata']['enabled_modules'].append('hairstyle')
                logger.info(f"Hairstyle analysis completed: {hairstyle_result.get('classification', 'Unknown')}")
            except Exception as e:
                logger.error(f"Hairstyle analysis failed: {e}", exc_info=True)
                result['hairstyle'] = {'success': False, 'error': str(e)}

        # 메타데이터 완성
        total_time = (time.time() - start_time) * 1000
        result['metadata']['total_processing_time_ms'] = round(total_time, 2)
        result['metadata']['timestamp'] = time.strftime("%Y-%m-%dT%H:%M:%S")

        logger.info(f"Unified analysis completed in {total_time:.2f}ms")

        return result

    def _analyze_with_mediapipe(self, image: np.ndarray) -> Dict[str, Any]:
        """
        MediaPipe로 얼굴 분석 (랜드마크 + 얼굴형 + 눈꼬리)

        Args:
            image: BGR 이미지 (numpy array)

        Returns:
            MediaPipe 분석 결과
        """
        # 얼굴 검출
        detection_result: DetectionResult = self.mediapipe_detector.detect(image)

        if not detection_result.success:
            return {
                'success': False,
                'landmarks_count': 0,
                'processing_time_ms': detection_result.processing_time
            }

        # 얼굴 기하학 계산
        face_geometry: FaceGeometry = self.mediapipe_detector.calculate_geometry(
            detection_result.landmarks
        )

        result = {
            'success': True,
            'landmarks_count': len(detection_result.landmarks),
            'confidence': detection_result.confidence,
            'face_geometry': {
                'pitch': round(face_geometry.pitch, 2),
                'yaw': round(face_geometry.yaw, 2),
                'roll': round(face_geometry.roll, 2),
                'face_width': round(face_geometry.face_width, 4),
                'face_height': round(face_geometry.face_height, 4)
            },
            'processing_time_ms': round(detection_result.processing_time, 2)
        }

        # FaceAnalyzer로 얼굴형 + 눈꼬리 분석
        if self.face_analyzer:
            try:
                detailed_analysis: DetailedFaceAnalysis = self.face_analyzer.get_detailed_analysis(
                    detection_result.landmarks,
                    roll_angle=face_geometry.roll,
                    yaw_angle=face_geometry.yaw
                )

                # 눈 형태 분석 결과 추가
                if detailed_analysis.eye_analysis:
                    result['eye_analysis'] = detailed_analysis.eye_analysis.to_dict()

                # 얼굴형 분석 결과 추가
                if detailed_analysis.face_shape_analysis:
                    result['face_shape_analysis'] = detailed_analysis.face_shape_analysis.to_dict()

            except Exception as e:
                logger.warning(f"FaceAnalyzer failed: {e}")

        return result

    def get_compact_result(self, full_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        압축된 결과 반환 (TCP 전송용)

        Args:
            full_result: analyze_image()의 전체 결과

        Returns:
            압축된 결과 (핵심 정보만)
        """
        compact = {
            'timestamp': full_result['metadata']['timestamp'],
            'success': full_result['success']
        }

        # MediaPipe 핵심 정보
        if 'mediapipe' in full_result and full_result['mediapipe'].get('success'):
            mp = full_result['mediapipe']
            compact['face_geometry'] = mp['face_geometry']

        # Hairstyle 핵심 정보
        if 'hairstyle' in full_result:
            hs = full_result['hairstyle']
            compact['hairstyle'] = {
                'classification': hs.get('classification', 'Unknown')
            }

            if 'clip_results' in hs:
                clip = hs['clip_results']
                compact['hairstyle'].update({
                    'gender': clip.get('gender', 'Unknown'),
                    'glasses': clip.get('glasses', 'Unknown'),
                    'beard': clip.get('beard', 'Unknown')
                })

        return compact

    def __repr__(self):
        enabled = self.config.unified_analysis
        return (
            f"UnifiedFaceAnalyzer("
            f"mediapipe={enabled.enable_mediapipe}, "
            f"hairstyle={enabled.enable_hairstyle})"
        )


if __name__ == "__main__":
    # 테스트 코드
    import sys

    if len(sys.argv) < 2:
        print("Usage: python unified_analyzer.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    analyzer = UnifiedFaceAnalyzer()
    print(f"\n{analyzer}\n")

    result = analyzer.analyze_image(image_path)

    print("=== Unified Analysis Result ===\n")

    if result['success']:
        # MediaPipe 결과
        if 'mediapipe' in result and result['mediapipe'].get('success'):
            mp = result['mediapipe']
            print("📐 MediaPipe Analysis:")
            print(f"  Landmarks: {mp['landmarks_count']}")
            print(f"  Face Geometry:")
            geo = mp['face_geometry']
            print(f"    Pitch: {geo['pitch']}°")
            print(f"    Yaw: {geo['yaw']}°")
            print(f"    Roll: {geo['roll']}°")
            print(f"  Processing Time: {mp['processing_time_ms']}ms\n")

        # Hairstyle 결과
        if 'hairstyle' in result:
            hs = result['hairstyle']
            print("💇 Hairstyle Analysis:")
            print(f"  Classification: {hs.get('classification', 'Unknown')}")

            if 'clip_results' in hs:
                clip = hs['clip_results']
                print(f"  Gender: {clip.get('gender', 'Unknown')}")
                print(f"  Glasses: {clip.get('glasses', 'Unknown')}")
                print(f"  Beard: {clip.get('beard', 'Unknown')}\n")

        # 메타데이터
        meta = result['metadata']
        print(f"⏱️  Total Time: {meta['total_processing_time_ms']}ms")
        print(f"📦 Enabled Modules: {', '.join(meta['enabled_modules'])}")
    else:
        print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
