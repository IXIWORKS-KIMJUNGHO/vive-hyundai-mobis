"""
TCP 통신 모듈: 분석 결과를 Unreal Engine으로 전송
[MODIFIED] Python이 TCP 서버로 작동, Unreal이 클라이언트로 연결
"""
import socket
import json
import threading
import time
from .json_exporter import to_unreal_json
from .logging_config import get_logger

logger = get_logger(__name__)


class TCPServer:
    def __init__(self, ip="0.0.0.0", port=5000, fps=25):
        """
        TCP Server 초기화 (Python이 서버, Unreal이 클라이언트)

        Args:
            ip: 바인딩할 IP (0.0.0.0 = 모든 인터페이스)
            port: 리스닝 포트
            fps: 전송 주기 (Hz)
        """
        self.ip = ip
        self.port = port
        self.interval = 1.0 / fps  # 25Hz = 0.04초 간격

        self.latest_result = None
        self.is_running = False
        self.server_thread = None
        self.server_socket = None
        self.client_connections = []  # 연결된 클라이언트 리스트
        self.connections_lock = threading.Lock()

        logger.info(f"[TCP Server] Initialized - Bind: {ip}:{port}, Rate: {fps}Hz ({self.interval*1000:.1f}ms)")

    def update_result(self, result, image_path=""):
        """
        최신 결과 업데이트 (브로드캐스트할 데이터)

        Args:
            result: 분석 결과
            image_path: 이미지 경로
        """
        self.latest_result = {
            'result': result,
            'image_path': image_path
        }
        classification = result.get('classification', 'Unknown')
        logger.debug(f"[TCP Server] Latest result updated: {classification}")

    def start_server(self):
        """TCP 서버 시작"""
        if self.is_running:
            logger.warning("[TCP Server] Already running!")
            return

        self.is_running = True
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
        logger.info(f"[TCP Server] Starting server on {self.ip}:{self.port}...")

    def stop_server(self):
        """TCP 서버 중지"""
        if not self.is_running:
            logger.warning("[TCP Server] Not running!")
            return

        self.is_running = False

        # 모든 클라이언트 연결 종료
        with self.connections_lock:
            for client_sock in self.client_connections:
                try:
                    client_sock.close()
                except:
                    pass
            self.client_connections.clear()

        # 서버 소켓 종료
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

        if self.server_thread:
            self.server_thread.join(timeout=2.0)

        logger.info("[TCP Server] Server stopped")

    def _server_loop(self):
        """서버 메인 루프"""
        try:
            # 서버 소켓 생성
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.ip, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)  # 1초마다 체크

            logger.info(f"[TCP Server] Server listening on {self.ip}:{self.port}")

            # 클라이언트 연결 수락 스레드
            accept_thread = threading.Thread(target=self._accept_clients, daemon=True)
            accept_thread.start()

            # 브로드캐스트 루프 (25Hz)
            while self.is_running:
                if self.latest_result is not None:
                    self._broadcast_result()

                time.sleep(self.interval)

        except Exception as e:
            logger.error(f"[TCP Server] Server error: {e}")
        finally:
            logger.info("[TCP Server] Server loop ended")

    def _accept_clients(self):
        """클라이언트 연결 수락 루프"""
        logger.info("[TCP Server] Accepting client connections...")

        while self.is_running:
            try:
                client_sock, client_addr = self.server_socket.accept()
                logger.info(f"[TCP Server] Client connected: {client_addr}")

                with self.connections_lock:
                    self.client_connections.append(client_sock)

                # 클라이언트 핸들러 스레드 시작
                threading.Thread(
                    target=self._handle_client,
                    args=(client_sock, client_addr),
                    daemon=True
                ).start()

            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    logger.error(f"[TCP Server] Accept error: {e}")
                break

        logger.info("[TCP Server] Accept loop ended")

    def _handle_client(self, client_sock, client_addr):
        """개별 클라이언트 핸들러 (연결 유지 감지)"""
        try:
            while self.is_running:
                # 연결 상태 체크 (recv로 확인)
                try:
                    data = client_sock.recv(1, socket.MSG_PEEK)
                    if not data:
                        # 클라이언트가 연결 종료
                        break
                except socket.timeout:
                    pass
                except:
                    break

                time.sleep(1.0)

        except Exception as e:
            logger.error(f"[TCP Server] Client {client_addr} error: {e}")
        finally:
            # 연결 종료 처리
            with self.connections_lock:
                if client_sock in self.client_connections:
                    self.client_connections.remove(client_sock)
            try:
                client_sock.close()
            except:
                pass
            logger.info(f"[TCP Server] Client disconnected: {client_addr}")

    def _broadcast_result(self):
        """모든 연결된 클라이언트에게 결과 전송"""
        if not self.latest_result:
            return

        # JSON 직렬화 (개행 구분자 사용, 길이 prefix 제거)
        json_data = to_unreal_json(
            self.latest_result['result'],
            self.latest_result['image_path']
        )
        json_string = json.dumps(json_data, ensure_ascii=False)

        # 개행 문자 추가 (메시지 구분자)
        message = (json_string + '\n').encode('utf-8')

        # 모든 클라이언트에게 전송
        with self.connections_lock:
            disconnected = []
            for client_sock in self.client_connections:
                try:
                    client_sock.sendall(message)
                except:
                    # 전송 실패한 클라이언트는 연결 종료
                    disconnected.append(client_sock)

            # 연결 종료된 클라이언트 제거
            for sock in disconnected:
                if sock in self.client_connections:
                    self.client_connections.remove(sock)
                try:
                    sock.close()
                except:
                    pass

    def is_server_running(self):
        """서버 실행 상태 확인"""
        return self.is_running

    def get_client_count(self):
        """연결된 클라이언트 수"""
        with self.connections_lock:
            return len(self.client_connections)


# 전역 TCP Server 인스턴스 (싱글톤)
_tcp_server = None

def get_tcp_server():
    """전역 TCP Server 인스턴스 반환"""
    global _tcp_server
    if _tcp_server is None:
        _tcp_server = TCPServer()
    return _tcp_server


if __name__ == "__main__":
    # 테스트 코드
    server = TCPServer()

    # 테스트 데이터
    test_result = {
        'classification': 'Center Part',
        'clip_results': {
            'glasses': 'No Glasses',
            'glasses_confidence': 0.8,
            'beard': 'No Beard',
            'beard_confidence': 0.9,
            'gender': 'Male',
            'gender_confidence': 0.7
        }
    }

    print("Starting server...")
    server.start_server()

    print("Updating result every 2 seconds for 20 seconds...")
    for i in range(10):
        server.update_result(test_result, f"test_{i}.png")
        print(f"[Test] Connected clients: {server.get_client_count()}")
        time.sleep(2)

    server.stop_server()
    print("Test complete!")
