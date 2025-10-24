"""
Core analysis engine package.
"""
# Lazy imports to avoid dlib dependency when only using MediaPipe
# Modules can be imported directly when needed

__all__ = ['HairstyleAnalyzer', 'CLIPClassifier', 'GeometricAnalyzer', 'mediapipe']
