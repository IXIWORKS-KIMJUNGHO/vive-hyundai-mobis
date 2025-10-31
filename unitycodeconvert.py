import numpy as np
import cv2
import socket

class IRCameraClient:
    def __init__(self, ip='127.0.0.1', port=5001, width=1280, height=800):
        self.ip = ip
        self.port = port
        self.width = width
        self.height = height
        self.expected_size = width * height  # 1,024,000
        
        # Unity 스타일: 고정 크기 버퍼
        self.single_buffer = bytearray(self.expected_size)
        self.buffer_position = 0
        
        self.sock = None
        
    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.ip, self.port))
        print(f"Connected to {self.ip}:{self.port}")
        
    def reset_buffer(self):
        """Unity의 ResetBuffer() - position만 0으로"""
        self.buffer_position = 0
        
    def handle_received_data(self, read_buf):
        """Unity의 handleReceivedDataAndroidStyle() 완전 재현"""
        chunk_len = len(read_buf)
        remaining = self.expected_size - self.buffer_position
        is_len_over = chunk_len > remaining
        
        # 현재 프레임에 필요한 만큼만 복사
        copy_size = remaining if is_len_over else chunk_len
        self.single_buffer[self.buffer_position:self.buffer_position + copy_size] = read_buf[:copy_size]
        
        # 프레임 완성 체크
        if self.buffer_position + copy_size >= self.expected_size:
            # 프레임 처리
            self.process_image_buffer()
        else:
            # 아직 미완성
            self.buffer_position += copy_size  # ✅ FIX: chunk_len → copy_size
            return
        
        # leftover 처리 (다음 프레임의 시작)
        if is_len_over:
            self.reset_buffer()  # ← 클리어!
            
            leftover = chunk_len - remaining
            # 남은 부분을 버퍼 시작에 다시 복사
            self.single_buffer[0:leftover] = read_buf[remaining:remaining + leftover]
            self.buffer_position = leftover
            
            print(f"프레임 완성, leftover: {leftover} bytes")
    
    def process_image_buffer(self):
        """Unity의 processImageBufferAndroidStyle()"""
        try:
            # Y8을 이미지로 변환
            frame_data = bytes(self.single_buffer[:self.expected_size])
            img = np.frombuffer(frame_data, dtype=np.uint8).reshape(self.height, self.width)
            
            # Y축 뒤집기
            img = cv2.flip(img, 0)
            
            # 표시
            cv2.imshow('IR Camera', img)
            cv2.waitKey(1)
            
        except Exception as e:
            print(f"이미지 처리 에러: {e}")
    
    def run(self):
        """메인 루프"""
        self.connect()
        
        try:
            while True:
                # 64KB씩 받기 (Unity와 동일)
                chunk = self.sock.recv(65536)
                
                if not chunk:
                    print("연결 종료")
                    break
                
                # Unity 스타일 처리
                self.handle_received_data(chunk)
                
        except KeyboardInterrupt:
            print("중지됨")
        finally:
            if self.sock:
                self.sock.close()
            cv2.destroyAllWindows()

# 사용
if __name__ == '__main__':
    client = IRCameraClient(ip='127.0.0.1', port=5001)
    client.run()