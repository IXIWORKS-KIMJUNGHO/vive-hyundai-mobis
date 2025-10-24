# 헤어스타일 분석 시스템 - TCP 통신 규격서

## 1. 연결 정보

| 항목 | 값 |
|------|-----|
| **프로토콜** | TCP/IP |
| **IP 주소** | `127.0.0.1` (로컬 테스트) / 분석 서버 IP |
| **포트** | `5001` (기본값, 변경 가능) |
| **전송 방식** | Request-Response (이미지 전송 → 분석 결과 수신) |
| **메시지 형식** | Binary (4-byte size header + data payload) |
| **인코딩** | UTF-8 (JSON) |
| **JSON 크기** | ~400 bytes (평균) |

**연결 방식:** Python이 서버, Unreal Engine이 클라이언트로 연결

## 2. JSON 메시지 구조

### 전송 예제
```json
{
  "hairstyle": 2,
  "hairstyle_name": "Center Part",
  "gender": 1,
  "gender_name": "Male",
  "gender_confidence": 0.73,
  "has_glasses": 0,
  "glasses_confidence": 0.85,
  "has_beard": 0,
  "beard_confidence": 0.92,
  "face_shape": 0,
  "face_shape_name": "oval",
  "eye_shape": 1,
  "eye_shape_name": "downturned",
  "timestamp": "2025-10-16T14:30:25.123456",
  "image_path": "camera_capture_20251016_143025.png"
}
```

### 필드 정의

#### hairstyle (Integer)
헤어스타일 분류 결과 (Enum)

| 값 | 의미 | 성별 |
|----|------|------|
| `0` | Bangs (앞머리) | 남성 |
| `1` | All-Back (올백) | 남성 |
| `2` | Center Part (가운데 가르마) | 남성 |
| `3` | Right Side Part (우측 가르마) | 남성 |
| `4` | Left Side Part (좌측 가르마) | 남성 |
| `5` | Short Hair (단발) | 여성 |
| `6` | Long Hair (장발) | 여성 |
| `-1` | Unknown (분석 실패) | - |

#### gender (Integer)
성별 분류 결과 (Enum)

| 값 | 의미 |
|----|------|
| `0` | Female (여성) |
| `1` | Male (남성) |
| `-1` | Unknown |

#### has_glasses (Integer)
안경 착용 여부 (0 = 미착용, 1 = 착용)

#### has_beard (Integer)
수염 유무 (0 = 없음, 1 = 있음)

#### face_shape (Integer)
얼굴 형태 분류 결과 (MediaPipe 기반 기하학 분석)

얼굴의 종횡비(aspect_ratio)를 분석하여 분류합니다.

| 값 | 의미 | 특징 |
|----|------|------|
| `0` | Oval (계란형) | 세로가 긴 타원형 (aspect_ratio ≥ 1.15) |
| `1` | Round (둥근형) | 원형/정사각형에 가까운 (aspect_ratio < 1.15) |
| `-1` | Unknown (분석 실패) | 얼굴 감지 실패 |

**분석 방법:**
- MediaPipe 468개 landmark 기반
- 얼굴 너비/높이, 이마/광대/턱선 너비 측정
- 종횡비(height/width)로 단순 이진 분류

#### eye_shape (Integer)
눈 모양 분류 결과 (MediaPipe landmark 기반 각도 분석)

눈꼬리와 눈머리의 상대적 높이 차이를 분석하여 분류합니다.

| 값 | 의미 | 특징 |
|----|------|------|
| `0` | Upturned (올라간 눈) | 눈꼬리가 눈머리보다 높음 (기울기 > 0.1) |
| `1` | Downturned (내려간 눈) | 눈꼬리가 눈머리보다 낮음 (기울기 < 0.05) |
| `2` | Neutral (기본형) | 수평에 가까운 형태 (0.05 ≤ 기울기 ≤ 0.1) |
| `-1` | Unknown (분석 실패) | 눈 landmark 감지 실패 |

**분석 방법:**
- 좌/우 눈 각각 분석 후 가중 평균
- 눈꼬리(outer corner) - 눈머리(inner corner) 기울기 계산
- Roll 각도 보정 적용 (얼굴 기울기 보정)

#### 보조 필드
- `hairstyle_name` (String) - 헤어스타일 텍스트 이름
- `gender_name` (String) - 성별 텍스트 ("Male", "Female")
- `gender_confidence` (Float) - 성별 신뢰도 (0.0 ~ 1.0)
- `glasses_confidence` (Float) - 안경 신뢰도 (0.0 ~ 1.0)
- `beard_confidence` (Float) - 수염 신뢰도 (0.0 ~ 1.0)
- `face_shape_name` (String) - 얼굴 형태 텍스트 이름
- `eye_shape_name` (String) - 눈 모양 텍스트 이름
- `timestamp` (String) - 분석 시각 (ISO 8601)
- `image_path` (String) - 원본 이미지 경로

## 3. 통신 프로토콜

### 요청 (Unreal Engine → Python Server)

**이미지 전송 프로토콜:**
```
1. 4 bytes: 이미지 데이터 크기 (uint32, little-endian)
2. N bytes: 이미지 바이너리 데이터 (JPEG/PNG)
```

**예시 (Python 코드):**
```python
import struct

# 1. 이미지 크기 전송
image_size = len(image_bytes)
size_header = struct.pack('<I', image_size)
socket.sendall(size_header)

# 2. 이미지 데이터 전송
socket.sendall(image_bytes)
```

### 응답 (Python Server → Unreal Engine)

**JSON 결과 전송 프로토콜:**
```
1. 4 bytes: JSON 데이터 크기 (uint32, little-endian)
2. N bytes: JSON 문자열 (UTF-8 인코딩)
```

**예시 (Unreal C++ 코드):**
```cpp
// 1. JSON 크기 수신
uint32 JsonSize;
Socket->Recv((uint8*)&JsonSize, 4);

// 2. JSON 데이터 수신
TArray<uint8> JsonData;
JsonData.SetNum(JsonSize);
Socket->Recv(JsonData.GetData(), JsonSize);

// 3. UTF-8 → FString 변환
FString JsonString = FString(UTF8_TO_TCHAR(JsonData.GetData()));
```

## 4. 실제 테스트 결과

### 샘플 1: 올백 헤어스타일
```json
{
  "hairstyle": 1,
  "hairstyle_name": "All-Back",
  "gender": 1,
  "gender_name": "Male",
  "gender_confidence": 0.0894,
  "has_glasses": 0,
  "glasses_confidence": 0.6771,
  "has_beard": 0,
  "beard_confidence": 0.0615,
  "face_shape": 0,
  "face_shape_name": "oval",
  "eye_shape": 1,
  "eye_shape_name": "downturned",
  "timestamp": "2025-10-24T00:08:27",
  "image_path": "/tmp/unreal_temp_image.jpg"
}
```
**분석:** 남성, 올백 헤어, 계란형 얼굴, 내려간 눈 (처리 시간: ~370ms)

### 샘플 2: 가운데 가르마
```json
{
  "hairstyle": 2,
  "hairstyle_name": "Center Part",
  "gender": 1,
  "gender_name": "Male",
  "gender_confidence": 0.0840,
  "has_glasses": 0,
  "glasses_confidence": 0.2395,
  "has_beard": 1,
  "beard_confidence": 0.3744,
  "face_shape": 1,
  "face_shape_name": "round",
  "eye_shape": 1,
  "eye_shape_name": "downturned",
  "timestamp": "2025-10-24T00:08:27",
  "image_path": "/tmp/unreal_temp_image.jpg"
}
```
**분석:** 남성, 가운데 가르마, 둥근형 얼굴, 내려간 눈, 수염 있음

## 5. 성능 지표

| 지표 | 값 |
|------|-----|
| **JSON 크기** | 411-417 bytes |
| **평균 처리 시간** | 350-420ms |
| **이미지 크기** | ~200KB (JPEG, 800x1280) |
| **전송 지연** | < 10ms (로컬) |
| **총 RTT** | ~400ms |

## 6. 주요 사항

- **프로토콜**: Binary protocol (4-byte size header + payload)
- **재연결**: 언리얼 Play 모드 재시작 시 자동 재연결 가능
- **순차 처리**: 이미지 전송 → 분석 → 결과 수신 (Request-Response)
- **에러 처리**: 분석 실패 시 enum 값 `-1` 반환

## 7. Unreal Engine Enum 참조표

언리얼 엔진에서 다음과 같이 Enum을 정의하여 사용하세요:

```cpp
// HairstyleEnum.h
UENUM(BlueprintType)
enum class EHairstyleType : uint8
{
    Bangs          = 0 UMETA(DisplayName = "Bangs (앞머리)"),
    AllBack        = 1 UMETA(DisplayName = "All-Back (올백)"),
    CenterPart     = 2 UMETA(DisplayName = "Center Part (가운데 가르마)"),
    RightSidePart  = 3 UMETA(DisplayName = "Right Side Part (우측 가르마)"),
    LeftSidePart   = 4 UMETA(DisplayName = "Left Side Part (좌측 가르마)"),
    ShortHair      = 5 UMETA(DisplayName = "Short Hair (단발)"),
    LongHair       = 6 UMETA(DisplayName = "Long Hair (장발)"),
    Unknown        = 255 UMETA(DisplayName = "Unknown")
};

UENUM(BlueprintType)
enum class EGenderType : uint8
{
    Female  = 0 UMETA(DisplayName = "Female"),
    Male    = 1 UMETA(DisplayName = "Male"),
    Unknown = 255 UMETA(DisplayName = "Unknown")
};

UENUM(BlueprintType)
enum class EFaceShapeType : uint8
{
    Oval    = 0 UMETA(DisplayName = "Oval (계란형)"),
    Round   = 1 UMETA(DisplayName = "Round (둥근형)"),
    Square  = 2 UMETA(DisplayName = "Square (사각형)"),
    Heart   = 3 UMETA(DisplayName = "Heart (하트형)"),
    Oblong  = 4 UMETA(DisplayName = "Oblong (긴형)"),
    Unknown = 255 UMETA(DisplayName = "Unknown")
};

UENUM(BlueprintType)
enum class EEyeShapeType : uint8
{
    Upturned   = 0 UMETA(DisplayName = "Upturned (올라간 눈)"),
    Downturned = 1 UMETA(DisplayName = "Downturned (내려간 눈)"),
    Almond     = 2 UMETA(DisplayName = "Almond (아몬드형)"),
    Round      = 3 UMETA(DisplayName = "Round (둥근 눈)"),
    Hooded     = 4 UMETA(DisplayName = "Hooded (깊은 눈)"),
    Unknown    = 255 UMETA(DisplayName = "Unknown")
};
```

### JSON 파싱 예시 (Unreal C++)

```cpp
// FaceAnalysisResult.h
USTRUCT(BlueprintType)
struct FFaceAnalysisResult
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    EHairstyleType Hairstyle;

    UPROPERTY(BlueprintReadOnly)
    FString HairstyleName;

    UPROPERTY(BlueprintReadOnly)
    EGenderType Gender;

    UPROPERTY(BlueprintReadOnly)
    float GenderConfidence;

    UPROPERTY(BlueprintReadOnly)
    bool bHasGlasses;

    UPROPERTY(BlueprintReadOnly)
    float GlassesConfidence;

    UPROPERTY(BlueprintReadOnly)
    bool bHasBeard;

    UPROPERTY(BlueprintReadOnly)
    float BeardConfidence;

    UPROPERTY(BlueprintReadOnly)
    EFaceShapeType FaceShape;

    UPROPERTY(BlueprintReadOnly)
    FString FaceShapeName;

    UPROPERTY(BlueprintReadOnly)
    EEyeShapeType EyeShape;

    UPROPERTY(BlueprintReadOnly)
    FString EyeShapeName;

    UPROPERTY(BlueprintReadOnly)
    FString Timestamp;
};
```

## 8. 서버 시작 방법

```bash
# 1. 가상환경 활성화
cd unified_face_analyzer
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

# 2. TCP 서버 시작
python tcp_server.py --port 5001

# 3. 테스트 (별도 터미널)
python tcp_client_test.py sample_images/test.png --port 5001
```
