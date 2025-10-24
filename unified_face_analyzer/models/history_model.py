"""
History Model - 분석 기록 관리
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from PIL import Image
import numpy as np
from utils import get_logger

logger = get_logger(__name__)


@dataclass
class HistoryItem:
    """분석 기록 아이템"""
    path: str  # 원본 이미지 경로
    results: Dict[str, Any]  # 분석 결과
    viz_image: np.ndarray  # 시각화 이미지
    unreal_screenshot: Optional[Image.Image] = None  # 언리얼 스크린샷


class HistoryModel:
    """
    분석 기록 모델
    분석된 이미지들의 히스토리를 관리합니다.
    """

    def __init__(self):
        """히스토리 모델 초기화"""
        self.items: List[HistoryItem] = []
        self.selected_path: Optional[str] = None

    def add_item(self, path: str, results: Dict[str, Any], viz_image: np.ndarray) -> bool:
        """
        새 분석 결과를 히스토리에 추가

        Args:
            path: 이미지 경로
            results: 분석 결과
            viz_image: 시각화 이미지

        Returns:
            bool: 추가 성공 여부 (중복이면 False)
        """
        # 중복 확인
        if any(item.path == path for item in self.items):
            logger.warning(f"Duplicate item ignored: {path}")
            return False

        item = HistoryItem(
            path=path,
            results=results,
            viz_image=viz_image,
            unreal_screenshot=None
        )
        self.items.insert(0, item)  # 최신 항목을 앞에 추가
        logger.info(f"Added to history: {path}")
        return True

    def get_item_by_path(self, path: str) -> Optional[HistoryItem]:
        """
        경로로 히스토리 아이템 조회

        Args:
            path: 이미지 경로

        Returns:
            Optional[HistoryItem]: 해당 아이템 (없으면 None)
        """
        for item in self.items:
            if item.path == path:
                return item
        return None

    def set_unreal_screenshot(self, path: str, screenshot: Image.Image) -> bool:
        """
        특정 아이템에 언리얼 스크린샷 추가

        Args:
            path: 이미지 경로
            screenshot: 언리얼 스크린샷

        Returns:
            bool: 추가 성공 여부
        """
        item = self.get_item_by_path(path)
        if item:
            item.unreal_screenshot = screenshot.copy()
            logger.info(f"Unreal screenshot added to: {path}")
            return True
        logger.warning(f"Item not found: {path}")
        return False

    def set_selected(self, path: Optional[str]):
        """
        선택된 아이템 설정

        Args:
            path: 선택할 이미지 경로 (None이면 선택 해제)
        """
        self.selected_path = path
        if path:
            logger.debug(f"Selected: {path}")

    def get_selected(self) -> Optional[HistoryItem]:
        """
        현재 선택된 아이템 조회

        Returns:
            Optional[HistoryItem]: 선택된 아이템 (없으면 None)
        """
        if self.selected_path:
            return self.get_item_by_path(self.selected_path)
        return None

    def get_all_items(self) -> List[HistoryItem]:
        """
        모든 히스토리 아이템 조회

        Returns:
            List[HistoryItem]: 히스토리 아이템 리스트
        """
        return self.items

    def clear(self):
        """모든 히스토리 삭제"""
        self.items.clear()
        self.selected_path = None
        logger.info("History cleared")

    def get_count(self) -> int:
        """
        히스토리 아이템 개수 조회

        Returns:
            int: 아이템 개수
        """
        return len(self.items)

    def is_selected(self, path: str) -> bool:
        """
        특정 경로가 선택되었는지 확인

        Args:
            path: 확인할 이미지 경로

        Returns:
            bool: 선택 여부
        """
        return self.selected_path == path
