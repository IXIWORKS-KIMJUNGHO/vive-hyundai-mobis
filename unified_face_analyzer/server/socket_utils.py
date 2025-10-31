# -*- coding: utf-8 -*-
"""
Socket utility functions for non-blocking I/O operations
"""

import socket
import time
from typing import Optional
from utils import get_logger

logger = get_logger(__name__)


def setup_socket_buffers(sock: socket.socket, buffer_size_mb: int = 2):
    """
    소켓 버퍼 크기 설정 및 최적화

    Args:
        sock: 설정할 소켓
        buffer_size_mb: 버퍼 크기 (MB 단위, 기본값: 2MB)
    """
    buffer_size = buffer_size_mb * 1024 * 1024

    # 소켓 버퍼 크기 증가 - blocking 방지
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_size)

    # TCP_NODELAY 활성화 - Nagle 알고리즘 비활성화로 지연 최소화
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    logger.debug(f"Socket buffers configured: RCV={buffer_size:,}, SND={buffer_size:,}, TCP_NODELAY=1")


def recv_exactly(client_socket: socket.socket, size: int, timeout: float = 10.0, verbose: bool = False) -> Optional[bytes]:
    """
    정확한 크기만큼 데이터 수신 (Non-blocking I/O 최적화)

    ⚠️  개선사항:
    - 작은 chunk 크기 (65KB) 사용하여 blocking 최소화
    - 각 chunk마다 짧은 timeout (2초) 적용
    - 전체 timeout은 누적으로 관리

    Args:
        client_socket: 클라이언트 소켓
        size: 수신할 정확한 바이트 수
        timeout: 전체 타임아웃 (초)
        verbose: 상세 로그 출력 여부 (기본값: False)

    Returns:
        수신한 데이터 또는 None
    """
    CHUNK_SIZE = 65536  # 65KB - 작은 chunk로 blocking 최소화
    CHUNK_TIMEOUT = 2.0  # 각 chunk당 2초 timeout

    data = b''
    start_time = time.time()
    receiving_printed = False

    try:
        while len(data) < size:
            # 전체 timeout 체크
            elapsed = time.time() - start_time
            if elapsed > timeout:
                logger.error(f"Overall timeout: received {len(data)}/{size} bytes in {elapsed:.1f}s")
                print(f"\r⏱️  타임아웃: {len(data):,}/{size:,} bytes ({len(data)/size*100:.1f}%)")
                return None

            # 각 chunk마다 짧은 timeout 설정
            client_socket.settimeout(CHUNK_TIMEOUT)

            remaining = size - len(data)
            chunk = client_socket.recv(min(remaining, CHUNK_SIZE))

            if not chunk:
                # 연결 종료 - 에러 레벨을 warning으로 낮춤 (정상적인 종료일 수 있음)
                if len(data) == 0:
                    logger.debug(f"Connection closed before data received")
                else:
                    logger.warning(f"Connection closed: received {len(data)}/{size} bytes")
                return None

            data += chunk

            # 간단한 진행 상황 출력 (시작 시 한 번만)
            if not receiving_printed and verbose:
                print("📦 수신 중...", end='', flush=True)
                receiving_printed = True

    except socket.timeout:
        elapsed = time.time() - start_time
        logger.error(f"Chunk timeout: received {len(data)}/{size} bytes in {elapsed:.1f}s")
        print(f"\r⏱️  타임아웃: {len(data):,}/{size:,} bytes ({len(data)/size*100:.1f}%)")
        return None
    except Exception as e:
        logger.error(f"Error receiving data: {e}")
        return None
    finally:
        client_socket.settimeout(None)
        if receiving_printed and verbose:
            print(" ✅ 완료")

    return data


def clear_stale_data(client_socket: socket.socket):
    """
    소켓에서 오래된 데이터 제거

    Args:
        client_socket: 클라이언트 소켓
    """
    try:
        # 논블로킹 모드로 전환
        client_socket.setblocking(False)

        total_cleared = 0
        while True:
            try:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                total_cleared += len(chunk)
            except BlockingIOError:
                break

        # 블로킹 모드로 복원
        client_socket.setblocking(True)

        if total_cleared > 0:
            logger.info(f"Cleared {total_cleared:,} bytes of stale data")
            print(f"✅ 총 {total_cleared:,} bytes의 오래된 데이터 제거 완료")

    except Exception as e:
        logger.error(f"Error clearing stale data: {e}")
        client_socket.setblocking(True)
