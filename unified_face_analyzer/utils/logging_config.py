"""
Logging configuration module for hairstyle analyzer.
Provides centralized logging setup with file and console handlers.
"""
import logging
import logging.handlers
import os
from pathlib import Path
from .config_loader import get_config


def setup_logging(name: str = None) -> logging.Logger:
    """
    로깅 시스템 설정 및 로거 반환

    Args:
        name: 로거 이름 (일반적으로 __name__ 사용)

    Returns:
        logging.Logger: 설정된 로거 객체
    """
    config = get_config()

    # 로거 생성
    logger = logging.getLogger(name or __name__)

    # 이미 핸들러가 설정되어 있으면 중복 설정 방지
    if logger.handlers:
        return logger

    # 로그 레벨 설정
    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # 로그 포맷 설정
    log_format = config.logging.format
    date_format = config.logging.date_format
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # 콘솔 핸들러 설정
    if config.logging.console.enabled:
        console_handler = logging.StreamHandler()
        console_level = getattr(logging, config.logging.console.level.upper(), logging.INFO)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 파일 핸들러 설정
    if config.logging.file.enabled:
        # 로그 디렉토리 생성
        log_dir = Path(config.logging.file.directory)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / config.logging.file.filename

        # 파일 핸들러 (RotatingFileHandler 사용)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=config.logging.file.max_bytes,
            backupCount=config.logging.file.backup_count,
            encoding='utf-8'
        )
        file_level = getattr(logging, config.logging.file.level.upper(), logging.DEBUG)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    로거 가져오기 (간편 함수)

    Args:
        name: 로거 이름 (일반적으로 __name__ 사용)

    Returns:
        logging.Logger: 설정된 로거 객체
    """
    return setup_logging(name)


# 테스트용 메인
if __name__ == "__main__":
    # 테스트 로거 생성
    test_logger = get_logger(__name__)

    print("=== Logging System Test ===")
    test_logger.debug("This is a DEBUG message")
    test_logger.info("This is an INFO message")
    test_logger.warning("This is a WARNING message")
    test_logger.error("This is an ERROR message")
    test_logger.critical("This is a CRITICAL message")
    print("\n✅ Logging system test complete!")
