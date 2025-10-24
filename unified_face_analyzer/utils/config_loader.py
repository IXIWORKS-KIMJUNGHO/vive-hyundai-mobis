"""
Configuration Loader Module
설정 파일(config.yaml)을 로드하고 관리하는 모듈
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """
    Configuration Manager

    config.yaml 파일을 로드하고 설정값에 접근할 수 있는 인터페이스 제공

    Usage:
        config = Config()
        threshold = config.get('bangs_detection.hair_coverage_threshold')
        # or
        threshold = config.bangs_detection.hair_coverage_threshold
    """

    def __init__(self, config_path: str = None):
        """
        Configuration 초기화

        Args:
            config_path: config.yaml 파일 경로 (None이면 자동 탐색)
        """
        if config_path is None:
            # 프로젝트 루트에서 config.yaml 찾기
            current_dir = Path(__file__).parent  # utils/
            project_root = current_dir.parent  # project root
            config_path = project_root / "config.yaml"

            # utils/config.yaml도 확인 (하위 호환성)
            if not config_path.exists():
                config_path = current_dir / "config.yaml"

            # 환경 변수로 오버라이드 가능
            if 'HAIRSTYLE_CONFIG_PATH' in os.environ:
                config_path = Path(os.environ['HAIRSTYLE_CONFIG_PATH'])

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """config.yaml 파일 로드"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Please create config.yaml or set HAIRSTYLE_CONFIG_PATH environment variable."
            )

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {self.config_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        점(.) 구분자로 중첩된 설정값 가져오기

        Args:
            key_path: 설정 키 경로 (예: 'bangs_detection.hair_coverage_threshold')
            default: 키가 없을 때 반환할 기본값

        Returns:
            설정값 또는 기본값

        Example:
            >>> config.get('clip.glasses_confidence_threshold')
            0.60
        """
        keys = key_path.split('.')
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def __getattr__(self, name: str):
        """
        속성 접근 방식으로 설정값 가져오기

        Example:
            >>> config.bangs_detection.hair_coverage_threshold
            0.88
        """
        if name.startswith('_'):
            # Private 속성은 일반 방식으로 처리
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

        if name in self._config:
            value = self._config[name]
            # dict면 ConfigSection으로 wrapping
            if isinstance(value, dict):
                return ConfigSection(value)
            return value

        raise AttributeError(f"Config has no key '{name}'")

    def reload(self):
        """설정 파일 다시 로드 (runtime에 변경사항 적용)"""
        self._load_config()

    def to_dict(self) -> Dict[str, Any]:
        """전체 설정을 딕셔너리로 반환"""
        return self._config.copy()

    def __repr__(self):
        return f"Config(path={self.config_path})"


class ConfigSection:
    """
    Config의 하위 섹션을 나타내는 헬퍼 클래스
    중첩된 딕셔너리를 속성 접근 방식으로 사용 가능
    """

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value

        raise AttributeError(f"ConfigSection has no key '{name}'")

    def get(self, key: str, default: Any = None) -> Any:
        """키로 값 가져오기 (기본값 지원)"""
        return self._data.get(key, default)

    def __repr__(self):
        return f"ConfigSection({list(self._data.keys())})"


# Singleton 인스턴스 (전역으로 사용)
_global_config: Config = None


def get_config() -> Config:
    """
    전역 Config 인스턴스 반환 (Singleton 패턴)

    Returns:
        Config 인스턴스

    Example:
        >>> from config_loader import get_config
        >>> config = get_config()
        >>> threshold = config.bangs_detection.hair_coverage_threshold
    """
    global _global_config

    if _global_config is None:
        _global_config = Config()

    return _global_config


# Convenience function
def reload_config():
    """전역 설정 다시 로드"""
    global _global_config
    if _global_config is not None:
        _global_config.reload()


if __name__ == "__main__":
    # 테스트 코드
    config = get_config()

    print("=== Config Test ===")
    print(f"Config path: {config.config_path}")
    print()

    # 1. get() 방식
    print("1. get() method:")
    print(f"  Bangs threshold: {config.get('bangs_detection.hair_coverage_threshold')}")
    print(f"  TCP port: {config.get('tcp.port')}")
    print()

    # 2. 속성 접근 방식
    print("2. Attribute access:")
    print(f"  Bangs threshold: {config.bangs_detection.hair_coverage_threshold}")
    print(f"  CLIP glasses threshold: {config.clip.glasses_confidence_threshold}")
    print(f"  TCP host: {config.tcp.host}")
    print()

    # 3. 전체 섹션 접근
    print("3. Section access:")
    print(f"  Models: {config.models.dlib_predictor}")
    print(f"  Models: {config.models.bisenet_weights}")
    print()

    print("✅ Config loaded successfully!")
