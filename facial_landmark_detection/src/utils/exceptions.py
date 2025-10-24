"""커스텀 예외 클래스 정의"""


class FacialLandmarkException(Exception):
    """기본 예외 클래스"""
    pass


class DetectionError(FacialLandmarkException):
    """얼굴 검출 실패 예외"""
    pass


class InvalidImageError(FacialLandmarkException):
    """잘못된 이미지 입력 예외"""
    pass


class ConfigurationError(FacialLandmarkException):
    """설정 오류 예외"""
    pass


class LandmarkExtractionError(FacialLandmarkException):
    """랜드마크 추출 실패 예외"""
    pass
