"""
Analysis Model - HairstyleAnalyzer 래퍼 및 분석 로직 관리
"""
from core import HairstyleAnalyzer
from utils import get_logger

logger = get_logger(__name__)


class AnalysisModel:
    """
    헤어스타일 분석 모델
    HairstyleAnalyzer를 캡슐화하고 분석 로직을 관리합니다.
    """

    def __init__(self):
        """분석 모델 초기화"""
        self.analyzer = None
        self.is_initialized = False

    def initialize(self):
        """
        HairstyleAnalyzer 초기화 (별도 스레드에서 호출)

        Returns:
            bool: 초기화 성공 여부
        """
        try:
            logger.info("Initializing HairstyleAnalyzer...")
            self.analyzer = HairstyleAnalyzer()
            self.is_initialized = True
            logger.info("HairstyleAnalyzer initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize HairstyleAnalyzer: {e}", exc_info=True)
            self.is_initialized = False
            return False

    def analyze_image(self, image_path: str):
        """
        이미지 분석 실행

        Args:
            image_path: 분석할 이미지 경로

        Returns:
            tuple: (results dict, visualization image numpy array)
        """
        if not self.is_initialized or self.analyzer is None:
            logger.error("Analyzer not initialized")
            return {'error': 'Analyzer not initialized'}, None

        try:
            logger.info(f"Analyzing image: {image_path}")
            results, viz_image = self.analyzer.analyze_image(image_path)

            if 'error' in results:
                logger.error(f"Analysis failed: {results['error']}")
            else:
                classification = results.get('classification', 'Unknown')
                logger.info(f"Analysis complete: {classification}")

            return results, viz_image

        except Exception as e:
            logger.error(f"Analysis exception: {e}", exc_info=True)
            return {'error': str(e)}, None

    def is_ready(self) -> bool:
        """
        분석기 준비 상태 확인

        Returns:
            bool: 분석 가능 여부
        """
        return self.is_initialized and self.analyzer is not None
