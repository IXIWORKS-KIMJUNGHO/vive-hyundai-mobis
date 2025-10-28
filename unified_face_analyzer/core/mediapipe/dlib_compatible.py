"""
dlib-compatible wrapper classes for MediaPipe Face Mesh.

Provides backward compatibility with existing dlib-based code
by wrapping MediaPipe results in dlib-like interface.
"""

import numpy as np
from typing import List


class Point:
    """dlib.point 호환 클래스"""
    def __init__(self, x: int, y: int):
        self.x = int(x)
        self.y = int(y)

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


class Rectangle:
    """dlib.rectangle 호환 클래스"""
    def __init__(self, left: int, top: int, right: int, bottom: int):
        self._left = int(left)
        self._top = int(top)
        self._right = int(right)
        self._bottom = int(bottom)

    def left(self) -> int:
        return self._left

    def right(self) -> int:
        return self._right

    def top(self) -> int:
        return self._top

    def bottom(self) -> int:
        return self._bottom

    def width(self) -> int:
        return self._right - self._left

    def height(self) -> int:
        return self._bottom - self._top

    def __repr__(self):
        return f"Rectangle(({self._left},{self._top}),({self._right},{self._bottom}))"


class FullObjectDetection:
    """dlib.full_object_detection 호환 클래스 (68점 랜드마크)"""
    def __init__(self, rect: Rectangle, points: np.ndarray):
        """
        Args:
            rect: 얼굴 바운딩 박스
            points: (68, 2) shape의 랜드마크 좌표 numpy array
        """
        self.rect = rect
        self._points = [Point(int(x), int(y)) for x, y in points]

    def parts(self) -> List[Point]:
        """전체 68개 랜드마크 포인트 반환"""
        return self._points

    def part(self, idx: int) -> Point:
        """특정 인덱스의 랜드마크 포인트 반환"""
        if 0 <= idx < len(self._points):
            return self._points[idx]
        raise IndexError(f"Landmark index {idx} out of range (0-{len(self._points)-1})")

    def num_parts(self) -> int:
        """랜드마크 포인트 개수 (68)"""
        return len(self._points)

    def __repr__(self):
        return f"FullObjectDetection(rect={self.rect}, num_parts={self.num_parts()})"
