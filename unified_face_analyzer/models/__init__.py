"""
Models package for hairstyle analyzer MVC architecture.
"""
# Lazy imports to avoid circular dependencies and missing module issues
# Import landmark_models separately as it has no dependencies

__all__ = ['AnalysisModel', 'HistoryModel', 'HistoryItem', 'TCPModel', 'landmark_models']
