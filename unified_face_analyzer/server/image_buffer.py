# -*- coding: utf-8 -*-
"""
Thread-safe circular image buffer for IR camera frames
"""

import time
import threading
import numpy as np
from collections import deque
from typing import Optional, Dict, Any, Union
from utils import get_logger

logger = get_logger(__name__)


class ImageBuffer:
    """
    Thread-safe 순환 이미지 버퍼
    - 최대 5개의 BGR 이미지 (numpy array) 저장
    - FIFO: 5개 초과 시 가장 오래된 것 자동 제거
    - Analyze 요청 시: 가장 최신 이미지 사용

    IRCameraReceiver에서 BGR 변환된 이미지를 저장하고,
    AnalysisServer에서 변환 없이 바로 사용
    """

    def __init__(self, max_size: int = 5):
        """
        Args:
            max_size: 버퍼에 저장할 최대 이미지 개수 (기본값: 5)
        """
        self.buffer = deque(maxlen=max_size)  # 자동 FIFO 순환 버퍼
        self.lock = threading.Lock()
        self.update_count = 0
        self.max_size = max_size

    def update(self, bgr_image: np.ndarray):
        """
        BGR 이미지를 버퍼에 추가

        Args:
            bgr_image: BGR 이미지 numpy array (height, width, 3)

        Note:
            - deque의 maxlen 설정으로 자동 FIFO 동작
            - 5개 초과 시 가장 오래된 이미지 자동 제거
        """
        with self.lock:
            timestamp = time.time()
            self.buffer.append({
                'image': bgr_image,
                'timestamp': timestamp,
                'index': self.update_count
            })
            self.update_count += 1

            logger.debug(
                f"ImageBuffer 업데이트 - "
                f"버퍼: {len(self.buffer)}/{self.max_size}, "
                f"인덱스: {self.update_count}, "
                f"shape: {bgr_image.shape}"
            )

    def get(self) -> Optional[np.ndarray]:
        """
        가장 최신 BGR 이미지 반환

        Returns:
            최신 BGR 이미지 (numpy array), 버퍼가 비어있으면 None
        """
        with self.lock:
            if len(self.buffer) > 0:
                latest = self.buffer[-1]  # 마지막(최신) 이미지
                logger.debug(
                    f"ImageBuffer 조회 - "
                    f"인덱스: {latest['index']}, "
                    f"나이: {(time.time() - latest['timestamp']) * 1000:.1f}ms"
                )
                return latest['image']
            return None

    def get_info(self) -> Dict[str, Any]:
        """
        버퍼 상태 정보 조회

        Returns:
            버퍼 통계 정보 딕셔너리
        """
        with self.lock:
            if len(self.buffer) == 0:
                return {
                    'buffer_count': 0,
                    'buffer_max': self.max_size,
                    'has_data': False,
                    'update_count': self.update_count
                }

            latest = self.buffer[-1]
            oldest = self.buffer[0]

            return {
                'buffer_count': len(self.buffer),
                'buffer_max': self.max_size,
                'buffer_full': len(self.buffer) == self.max_size,
                'has_data': True,
                'image_shape': latest['image'].shape,
                'data_size': latest['image'].nbytes,
                'latest_timestamp': latest['timestamp'],
                'latest_index': latest['index'],
                'latest_age_ms': (time.time() - latest['timestamp']) * 1000,
                'oldest_timestamp': oldest['timestamp'],
                'oldest_index': oldest['index'],
                'oldest_age_ms': (time.time() - oldest['timestamp']) * 1000,
                'update_count': self.update_count,
                'buffer_span_ms': (latest['timestamp'] - oldest['timestamp']) * 1000
            }
