# -*- coding: utf-8 -*-
"""
Base TCP Server class with common lifecycle management
"""

import socket
import threading
from abc import ABC, abstractmethod
from utils import get_logger

logger = get_logger(__name__)


class BaseTCPServer(ABC):
    """
    Base class for all TCP servers

    Provides:
    - Lifecycle management (start/stop)
    - Thread safety
    - Common error handling
    """

    def __init__(self, name: str):
        """
        Args:
            name: Server name for logging
        """
        self.name = name
        self.is_running = False
        self.thread = None
        logger.info(f"{self.name} initialized")

    @abstractmethod
    def _run(self):
        """
        Main server loop - must be implemented by subclasses
        """
        pass

    def start(self):
        """Start server in background thread"""
        if self.is_running:
            logger.warning(f"{self.name} already running")
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"{self.name} started in background thread")

    def stop(self):
        """Stop server gracefully"""
        if not self.is_running:
            return

        logger.info(f"Stopping {self.name}...")
        self.is_running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)

        logger.info(f"{self.name} stopped")

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
