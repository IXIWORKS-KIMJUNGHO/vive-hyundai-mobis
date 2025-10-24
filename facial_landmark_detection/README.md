# Facial Landmark Detection System

MediaPipe 기반 얼굴 랜드마크 검출 시스템 (Layer 1 & 2)

## 📋 현재 구현 상태

### ✅ Phase 0: Project Setup (완료)
- 프로젝트 디렉토리 구조
- requirements.txt
- .gitignore
- 가상환경 설정

### ✅ Phase 1: Data Models & Configuration (완료)
- **src/config/constants.py**: 12개 얼굴 영역 정의 (FACIAL_REGIONS)
- **src/config/settings.py**: DetectionConfig, VisualizationStyle
- **src/utils/exceptions.py**: 커스텀 예외 클래스
- **src/utils/validators.py**: 검증 함수
- **src/models.py**: Landmark, DetectionResult, ProcessedResult, FaceGeometry

### ✅ Phase 2: Core Detection Layer (완료)
- **src/core/normalizer.py**: CoordinateNormalizer (좌표 정규화)
- **src/core/landmark_extractor.py**: LandmarkExtractor (468개 landmark 추출)
- **src/core/face_detector.py**: FaceDetector (MediaPipe wrapper)

### ✅ Phase 3: Processing Layer (완료)
- **src/processing/result_handler.py**: ResultHandler (결과 후처리, temporal smoothing)
- **src/processing/geometry.py**: GeometryCalculator (얼굴 각도/크기 계산)
- **src/processing/frame_processor.py**: FrameProcessor (이미지/비디오/실시간 처리)

## 🚀 설치

### Python 버전 요구사항

**중요**: MediaPipe는 Python 3.9 ~ 3.11에서만 설치 가능합니다.

```bash
# Python 버전 확인
python3 --version

# Python 3.9-3.11이 필요합니다
```

### 의존성 설치

```bash
# 가상환경 생성 (Python 3.9-3.11 사용)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

## 📦 프로젝트 구조

```
facial_landmark_detection/
├── src/
│   ├── __init__.py
│   ├── models.py                   # 데이터 모델
│   ├── core/                       # Layer 1: Core Detection
│   │   ├── __init__.py
│   │   ├── normalizer.py           # 좌표 정규화
│   │   ├── landmark_extractor.py   # Landmark 추출
│   │   └── face_detector.py        # MediaPipe wrapper
│   ├── processing/                 # Layer 2: Processing
│   │   └── __init__.py
│   ├── interface/                  # Layer 3: Interface
│   │   └── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── constants.py            # 상수 및 FACIAL_REGIONS
│   │   └── settings.py             # 설정 클래스
│   └── utils/
│       ├── __init__.py
│       ├── exceptions.py           # 예외 클래스
│       └── validators.py           # 검증 함수
├── tests/
├── examples/
├── data/
│   ├── sample_images/
│   └── output/
├── docs/
├── requirements.txt
└── README.md
```

## 🎯 사용 예제 (Python 3.9-3.11에서)

### Phase 1 & 2 사용 예제

```python
from src.config.settings import DetectionConfig
from src.config.constants import FACIAL_REGIONS
from src.models import Landmark

# 설정 생성
config = DetectionConfig(
    model_complexity=1,
    min_detection_confidence=0.5,
    static_image_mode=True
)

# 얼굴 영역 확인
print(f"Available regions: {list(FACIAL_REGIONS.keys())}")
# ['face_oval', 'left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow',
#  'nose_bridge', 'nose_tip', 'nostrils', 'lips_outer', 'lips_inner',
#  'chin', 'forehead']

# Landmark 생성
landmark = Landmark(x=0.5, y=0.5, z=0.0, visibility=0.95)
```

### MediaPipe 기반 검출 (구현 완료, Python 3.9-3.11 필요)

```python
import cv2
from src.core.face_detector import FaceDetector
from src.config.settings import DetectionConfig

# 검출기 초기화
config = DetectionConfig(static_image_mode=True)
detector = FaceDetector(config)

# 이미지 로드
image = cv2.imread('face.jpg')

# 얼굴 검출
result = detector.detect(image)

if result.success:
    print(f"✅ 검출 성공!")
    print(f"Landmarks: {len(result.landmarks)}개")
    print(f"처리 시간: {result.processing_time:.2f}ms")
    print(f"Bounding box: {result.bounding_box}")

detector.release()
```

## 🔧 구현된 기능

### Layer 1 - Core Detection ✅
- ✅ **FaceDetector**: MediaPipe FaceMesh wrapper
- ✅ **LandmarkExtractor**: 468개 landmark 추출 및 영역별 그룹화
- ✅ **CoordinateNormalizer**: 좌표 정규화 및 bounding box 계산

### Layer 2 - Processing ✅
- ✅ **FrameProcessor**: 이미지/비디오/실시간 처리 파이프라인
- ✅ **ResultHandler**: 결과 검증, 신뢰도 필터링, temporal smoothing
- ✅ **GeometryCalculator**: 얼굴 각도(pitch, yaw, roll), 크기, 거리 계산

### 데이터 모델 ✅
- ✅ **Landmark**: 3D 좌표 + 가시성 + 픽셀 좌표
- ✅ **DetectionResult**: 검출 결과 (landmarks, confidence, bbox, 처리 시간)
- ✅ **FaceGeometry**: 얼굴 각도 및 크기 정보
- ✅ **ProcessedResult**: 프레임 처리 결과

### 설정 및 상수 ✅
- ✅ **DetectionConfig**: MediaPipe 설정 (model_complexity, confidence 등)
- ✅ **FACIAL_REGIONS**: 12개 얼굴 영역 정의
- ✅ **예외 처리**: 5개 커스텀 예외 클래스
- ✅ **검증 함수**: 이미지, 신뢰도, 인덱스 검증

## ⚠️ 중요 사항

### Python 버전 호환성

현재 개발 환경: Python 3.13.7
MediaPipe 지원: Python 3.9 ~ 3.11

**실행 테스트를 위해서는 Python 3.9-3.11 환경이 필요합니다.**

```bash
# pyenv 등으로 Python 3.11 설치 (권장)
pyenv install 3.11.0
pyenv local 3.11.0

# 가상환경 재생성
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 📚 다음 단계 (선택 사항)

### Phase 4: Testing & Validation (권장)
- 단위 테스트 작성 (pytest 기반)
- 통합 테스트 및 엣지 케이스 검증
- 성능 벤치마크 측정
- 테스트 커버리지 확인

### Phase 5: Examples & Documentation
- 실제 이미지를 사용한 데모 예제
- 비디오 처리 예제
- 실시간 카메라 처리 예제
- API 문서 자동 생성 (Sphinx 등)

### Phase 6: Interface Layer (Layer 3) - 선택 사항
- Visualizer: 랜드마크 시각화 및 오버레이
- CameraInterface: 카메라 관리 유틸리티
- DataExporter: JSON/CSV/NumPy 형식 내보내기

## 📄 라이선스

이 프로젝트는 Hyundai Mobis의 내부 프로젝트입니다.

## 🤝 기여

**Phase 0-3 구현 완료: 2025-10-23**

**구현 완료 항목:**
- ✅ Phase 0: Project Setup (디렉토리 구조, requirements.txt, venv)
- ✅ Phase 1: Data Models & Configuration (5개 파일, 12개 얼굴 영역 정의)
- ✅ Phase 2: Core Detection Layer (3개 파일, MediaPipe 통합)
- ✅ Phase 3: Processing Layer (3개 파일, 전체 처리 파이프라인)

**시스템 환경:**
- Python 3.12.12
- MediaPipe 0.10.21
- OpenCV 4.11.0
- NumPy 1.26.4

**총 구현된 파일:** 17개 Python 파일 (~1,500 라인)
**클래스 구현:** 12개 (FaceDetector, LandmarkExtractor, CoordinateNormalizer, FrameProcessor, ResultHandler, GeometryCalculator 등)
