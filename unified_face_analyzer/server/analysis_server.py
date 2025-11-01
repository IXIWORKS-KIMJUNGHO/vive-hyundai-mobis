# -*- coding: utf-8 -*-
"""
Analysis Server - Port 10000
Handles face analysis requests using UnifiedFaceAnalyzer
"""

import socket
import json
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from typing import Optional, Dict, Any
from .base_server import BaseTCPServer
from .image_buffer import ImageBuffer
from .socket_utils import recv_exactly, clear_stale_data
from core.unified_analyzer import UnifiedFaceAnalyzer
from utils import get_logger
from utils.color_log import ColorLog

logger = get_logger(__name__)


class AnalysisServer(BaseTCPServer):
    """
    얼굴 분석 요청 서버 (Port 10000)

    Features:
    - 'analyze' 명령 수신
    - ImageBuffer에서 최신 BGR 이미지 가져오기 (변환 없이 바로 사용)
    - UnifiedFaceAnalyzer로 분석
    - JSON 결과 반환 (TCP_SPEC 형식)
    """

    def __init__(
        self,
        image_buffer: ImageBuffer,
        analyzer: UnifiedFaceAnalyzer,
        host: str = '0.0.0.0',
        port: int = 10000
    ):
        """
        Args:
            image_buffer: 공유 ImageBuffer 인스턴스
            analyzer: UnifiedFaceAnalyzer 인스턴스
            host: 바인딩할 호스트
            port: 바인딩할 포트
        """
        super().__init__("AnalysisServer")
        self.image_buffer = image_buffer
        self.analyzer = analyzer
        self.host = host
        self.port = port
        self.server_socket = None

    def _run(self):
        """분석 서버 메인 루프"""
        try:
            # 서버 소켓 생성
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)

            ColorLog.header("Face Analysis Server")
            ColorLog.info(f"Listening on {self.host}:{self.port}")

            # 클라이언트 연결 수락
            while self.is_running:
                try:
                    self.server_socket.settimeout(1.0)
                    client_socket, client_address = self.server_socket.accept()

                    ColorLog.event(f"Analysis client connected: {client_address[0]}:{client_address[1]}")

                    # 클라이언트 요청 처리
                    self._handle_client(client_socket, client_address)

                except socket.timeout:
                    continue
                except Exception as e:
                    if self.is_running:
                        ColorLog.error(f"Error accepting client: {e}")

        except Exception as e:
            ColorLog.error(f"Analysis server error: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()
            ColorLog.info("Analysis server stopped")

    def _handle_client(self, client_socket: socket.socket, client_address: tuple):
        """
        클라이언트 요청 처리

        Protocol:
        1. Client → Server: 'analyze' 명령 (7 bytes)
        2. Server: ImageBuffer에서 최신 BGR 이미지 가져오기 (변환 없음)
        3. Server: UnifiedFaceAnalyzer로 분석
        4. Server → Client: JSON 결과
        """
        try:
            while self.is_running:
                # 명령 수신 (7 bytes: 'analyze')
                command_data = client_socket.recv(7)

                if not command_data:
                    logger.info(f"Client disconnected: {client_address}")
                    break

                command = command_data.decode('utf-8', errors='ignore').strip()

                if command == 'analyze':
                    # ImageBuffer에서 최신 BGR 이미지 가져오기 (변환 없이 바로 사용!)
                    bgr_image = self.image_buffer.get()

                    if bgr_image is None:
                        error_result = {
                            'success': False,
                            'error': 'No image data available in buffer'
                        }
                        self._send_json_result(client_socket, error_result)
                        ColorLog.warning("No image data in buffer")
                        continue

                    try:
                        analysis_result = self.analyzer.analyze(bgr_image)

                        # TCP_SPEC 형식으로 변환
                        tcp_result = self._convert_to_tcp_spec_format(analysis_result)

                    except Exception as analyze_error:
                        # 분석 실패 시에도 에러 응답 반환 (연결 유지)
                        logger.error(f"Analysis failed: {analyze_error}", exc_info=True)
                        tcp_result = {
                            'success': False,
                            'error': f'Analysis failed: {str(analyze_error)}',
                            'hairstyle': -1,
                            'hairstyle_name': 'Unknown',
                            'gender': -1,
                            'gender_name': 'Unknown',
                            'gender_confidence': 0.0,
                            'has_glasses': 0,
                            'glasses_confidence': 0.0,
                            'has_beard': 0,
                            'beard_confidence': 0.0,
                            'face_shape': -1,
                            'face_shape_name': 'Unknown',
                            'eye_shape': -1,
                            'eye_shape_name': 'Unknown',
                            'timestamp': '',
                            'image_path': ''
                        }
                        ColorLog.error(f"Analysis failed: {str(analyze_error)}")

                    # JSON 결과 전송
                    self._send_json_result(client_socket, tcp_result)

                    # 간결한 결과 로그
                    if tcp_result.get('hairstyle', -1) == -1:
                        ColorLog.warning("Face not detected")
                    else:
                        result_str = (
                            f"{tcp_result['gender_name']} | "
                            f"{tcp_result['hairstyle_name']} | "
                            f"{'Glasses' if tcp_result['has_glasses'] == 1 else 'No Glasses'} | "
                            f"{'Beard' if tcp_result['has_beard'] == 1 else 'No Beard'} | "
                            f"Face: {tcp_result['face_shape_name']} | "
                            f"Eyes: {tcp_result['eye_shape_name']}"
                        )

                        ColorLog.analysis(f"Result: {result_str}")

                else:
                    logger.warning(f"Unknown command from {client_address}: {command}")

        except Exception as e:
            logger.error(f"Error handling analysis client {client_address}: {e}")
        finally:
            client_socket.close()
            logger.info(f"Analysis client disconnected: {client_address}")

    def _sanitize_for_json(self, obj):
        """
        JSON 직렬화를 위한 객체 정제

        Args:
            obj: 정제할 객체

        Returns:
            JSON 직렬화 가능한 객체
        """
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._sanitize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        return obj

    def _convert_to_tcp_spec_format(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        UnifiedFaceAnalyzer 결과를 TCP_SPEC 형식으로 변환

        TCP_SPEC.md 참조:
        - hairstyle: 0-6 (남성: 0-4, 여성: 5-6), -1=Unknown
        - gender: 0=Female, 1=Male, -1=Unknown
        - has_glasses: 0=미착용, 1=착용
        - has_beard: 0=없음, 1=있음
        - face_shape: 0=Oval, 1=Round, -1=Unknown
        - eye_shape: 0=Upturned, 1=Downturned, 2=Neutral, -1=Unknown

        Args:
            result: UnifiedFaceAnalyzer.analyze() 결과

        Returns:
            TCP_SPEC 형식의 딕셔너리
        """
        import time

        # 기본 구조 (얼굴 미검출 시)
        tcp_result = {
            'hairstyle': -1,
            'hairstyle_name': 'Unknown',
            'gender': -1,
            'gender_name': 'Unknown',
            'gender_confidence': 0.0,
            'has_glasses': 0,
            'glasses_confidence': 0.0,
            'has_beard': 0,
            'beard_confidence': 0.0,
            'face_shape': -1,
            'face_shape_name': 'Unknown',
            'eye_shape': -1,
            'eye_shape_name': 'Unknown',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'image_path': ''
        }

        # 얼굴 미검출 시 즉시 반환
        if not result.get('face_detected', False):
            return tcp_result

        # Gender 변환 (0=Female, 1=Male, -1=Unknown)
        gender_str = result.get('gender', 'Unknown')
        if gender_str == 'Male':
            tcp_result['gender'] = 1
            tcp_result['gender_name'] = 'Male'
        elif gender_str == 'Female':
            tcp_result['gender'] = 0
            tcp_result['gender_name'] = 'Female'
        else:
            tcp_result['gender'] = -1
            tcp_result['gender_name'] = 'Unknown'

        # Gender confidence (임시 - 실제 confidence가 있으면 사용)
        tcp_result['gender_confidence'] = 0.85

        # Hairstyle 변환
        hairstyle_str = result.get('hairstyle', 'Unknown')
        # TCP_SPEC: Bangs=0, AllBack=1, CenterPart=2, RightSidePart=3, LeftSidePart=4, ShortHair=5, LongHair=6
        hairstyle_map = {
            'Bangs': 0,
            'All-Back': 1,
            'Center Part': 2,
            'Right Side Part': 3,
            'Left Side Part': 4,
            'Short Hair': 5,
            'Long Hair': 6,
            'Unknown': -1
        }
        tcp_result['hairstyle'] = hairstyle_map.get(hairstyle_str, -1)
        tcp_result['hairstyle_name'] = hairstyle_str

        # Glasses 변환 (0=미착용, 1=착용)
        glasses_str = result.get('glasses', 'None')
        tcp_result['has_glasses'] = 1 if glasses_str == 'Wearing Glasses' else 0

        # CLIP confidence 가져오기
        if 'hairstyle' in result and 'clip_results' in result['hairstyle']:
            tcp_result['glasses_confidence'] = result['hairstyle']['clip_results'].get('glasses_confidence', 0.0)
        else:
            tcp_result['glasses_confidence'] = 0.0

        # Beard 변환 (0=없음, 1=있음)
        beard_str = result.get('beard', 'None')
        tcp_result['has_beard'] = 1 if beard_str == 'With Beard' else 0

        # CLIP confidence 가져오기
        if 'hairstyle' in result and 'clip_results' in result['hairstyle']:
            tcp_result['beard_confidence'] = result['hairstyle']['clip_results'].get('beard_confidence', 0.0)
        else:
            tcp_result['beard_confidence'] = 0.0

        # Face Shape 변환 (MediaPipe 결과에서 가져오기)
        if 'mediapipe_results' in result and 'face_analysis' in result['mediapipe_results']:
            face_analysis = result['mediapipe_results']['face_analysis']

            if 'face_shape_analysis' in face_analysis:
                face_shape_data = face_analysis['face_shape_analysis']
                face_shape_str = face_shape_data.get('face_shape', 'Unknown')

                # TCP_SPEC: Oval=0, Round=1, -1=Unknown
                face_shape_map = {
                    'oval': 0,
                    'round': 1,
                    'Unknown': -1
                }
                tcp_result['face_shape'] = face_shape_map.get(face_shape_str.lower(), -1)
                tcp_result['face_shape_name'] = face_shape_str.capitalize()

        # Eye Shape 변환 (MediaPipe 결과에서 가져오기)
        if 'mediapipe_results' in result and 'face_analysis' in result['mediapipe_results']:
            face_analysis = result['mediapipe_results']['face_analysis']

            if 'eye_analysis' in face_analysis:
                eye_data = face_analysis['eye_analysis']
                eye_shape_str = eye_data.get('overall_eye_shape', 'Unknown')

                # TCP_SPEC: Upturned=0, Downturned=1, Neutral=2, -1=Unknown
                eye_shape_map = {
                    'upturned': 0,
                    'downturned': 1,
                    'neutral': 2,
                    'Unknown': -1
                }
                tcp_result['eye_shape'] = eye_shape_map.get(eye_shape_str.lower(), -1)
                tcp_result['eye_shape_name'] = eye_shape_str.capitalize()

        return tcp_result

    def _send_json_result(self, client_socket: socket.socket, result: Dict[str, Any]) -> bool:
        """
        JSON 결과를 클라이언트에게 전송

        Args:
            client_socket: 클라이언트 소켓
            result: 전송할 결과 딕셔너리

        Returns:
            전송 성공 여부
        """
        try:
            # JSON 직렬화를 위한 정제
            sanitized_result = self._sanitize_for_json(result)

            # JSON 문자열 생성
            json_str = json.dumps(sanitized_result, ensure_ascii=False)
            json_bytes = json_str.encode('utf-8')

            # 전송
            client_socket.sendall(json_bytes)

            logger.info(f"JSON result sent: {len(json_bytes)} bytes")
            return True

        except Exception as e:
            logger.error(f"Error sending JSON result: {e}")
            return False

    def stop(self):
        """서버 종료"""
        super().stop()
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
