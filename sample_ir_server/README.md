# Sample IR Server

2개의 TCP 포트를 사용하는 듀얼 목적 서버

## 구조

### dual_server.py (추천)

```
dual_server.py
├── Port 5000: JSON 바이트 전송 (1회)
│   └── result.json → bytes
└── Port 5001: Raw Y8 스트리밍 (연속)
    └── camera_capture_20250513_185039.png → Y8 raw data
        ↓
Client (test_dual_client.py 또는 Unity/Unreal)
```

### dual_camera_server.py (구버전)

```
dual_camera_server.py
├── Port 5000: Raw Y8 스트리밍
└── Port 5001: Raw Y8 스트리밍
    ↓
Client (test_client.py)
```

## 필수 요구사항

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy

```bash
pip install opencv-python numpy
```

## 사용 방법

### A. dual_server.py (추천)

#### 1. 파일 준비

```
sample_ir_server/
  ├── result.json (샘플 제공됨)
  ├── camera_capture_20250513_185039.png (샘플 제공됨)
  ├── dual_server.py
  └── test_dual_client.py
```

#### 2. 서버 실행

```bash
# 기본 사용 (샘플 파일 사용)
python dual_server.py

# 커스텀 파일 사용
python dual_server.py --json your_result.json --image your_image.png

# 포트 및 옵션 변경
python dual_server.py \
  --json result.json \
  --image camera_capture.png \
  --json-port 5000 \
  --y8-port 5001 \
  --width 1280 \
  --height 800 \
  --fps 30
```

#### 3. 테스트 클라이언트 실행

```bash
# JSON + Y8 모두 테스트 (기본)
python test_dual_client.py

# JSON만 테스트
python test_dual_client.py --test json

# Y8만 테스트 (화면 표시)
python test_dual_client.py --test y8

# Y8 화면 표시 없이
python test_dual_client.py --test y8 --no-display
```

### B. dual_camera_server.py (구버전)

#### 1. IR 이미지 준비

```
sample_ir_server/
  ├── images/
  │   ├── driver_ir.jpg
  │   └── passenger_ir.jpg
  ├── dual_camera_server.py
  └── test_client.py
```

#### 2. 서버 실행

```bash
python dual_camera_server.py --image1 images/driver_ir.jpg --image2 images/passenger_ir.jpg
```

#### 3. 테스트

```bash
python test_client.py
```

## 명령줄 옵션

### dual_camera_server.py

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--port1` | 카메라 1 포트 | 5000 |
| `--port2` | 카메라 2 포트 | 5001 |
| `--width` | 이미지 너비 | 1280 |
| `--height` | 이미지 높이 | 800 |
| `--fps` | 전송 FPS | 30 |
| `--image1` | 카메라 1 IR 이미지 경로 | **필수** |
| `--image2` | 카메라 2 IR 이미지 경로 | **필수** |

### test_client.py

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--host` | 서버 주소 | 127.0.0.1 |
| `--port1` | 카메라 1 포트 | 5000 |
| `--port2` | 카메라 2 포트 | 5001 |
| `--width` | 이미지 너비 | 1280 |
| `--height` | 이미지 높이 | 800 |
| `--no-display` | 화면 표시 비활성화 | False |

## 프로토콜

### Raw Y8 스트리밍

```
서버 → 클라이언트: 연속적인 Raw Y8 데이터
- 크기: width × height bytes (예: 1280×800 = 1,024,000 bytes)
- 형식: Grayscale (Y8)
- 전송: 연속 스트리밍 (프레임 경계 없음)
```

클라이언트는 정확히 `width × height` 바이트를 받으면 하나의 프레임으로 처리합니다.

## 통합 예제

### Unity/Unreal과 연동

```csharp
// Unity C# 예제
TcpClient client = new TcpClient("127.0.0.1", 5000);
NetworkStream stream = client.GetStream();

byte[] buffer = new byte[1280 * 800];
int totalRead = 0;

while (totalRead < buffer.Length)
{
    int bytesRead = stream.Read(buffer, totalRead, buffer.Length - totalRead);
    totalRead += bytesRead;
}

// buffer를 Y8 이미지로 변환
```

### UnifiedFaceAnalyzer TCP Server와 연동

```bash
# Terminal 1: IR Camera Server
python dual_camera_server.py --image1 driver.jpg --image2 passenger.jpg

# Terminal 2: Face Analyzer Server (Port 5000 사용 시 포트 변경 필요)
cd ../unified_face_analyzer
python tcp_server.py --port 6000

# Terminal 3: 데이터 중계 (IR Server → Face Analyzer)
# 별도 스크립트 필요
```

## 종료

- 서버: `Ctrl+C`
- 클라이언트: `q` 키 또는 `Ctrl+C`

## 주의사항

1. **이미지 필수**: 서버 실행 시 `--image1`, `--image2` 옵션으로 IR 이미지를 반드시 제공해야 합니다.
2. **해상도**: 제공한 이미지는 자동으로 지정된 해상도로 리사이즈됩니다.
3. **포트 충돌**: 다른 프로그램이 5000, 5001 포트를 사용 중이면 다른 포트로 변경하세요.
4. **동시 클라이언트**: 각 카메라는 여러 클라이언트를 동시에 지원합니다 (브로드캐스트).

## 문제 해결

### "이미지 경로가 제공되지 않았습니다"

```bash
# 해결: --image1, --image2 옵션 추가
python dual_camera_server.py --image1 path/to/image1.jpg --image2 path/to/image2.jpg
```

### "이미지 파일을 찾을 수 없습니다"

```bash
# 파일 경로 확인
ls -la images/

# 절대 경로 사용
python dual_camera_server.py \
  --image1 /absolute/path/to/image1.jpg \
  --image2 /absolute/path/to/image2.jpg
```

### 포트 충돌 (Address already in use)

```bash
# 다른 포트 사용
python dual_camera_server.py \
  --image1 image1.jpg \
  --image2 image2.jpg \
  --port1 5002 \
  --port2 5003
```
