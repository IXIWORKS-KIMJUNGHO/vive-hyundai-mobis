# 얼굴 특징 분석 가이드

MediaPipe FaceMesh 기반 얼굴 특징 분석 시스템

---

## 🎯 기능 개요

### 1. 눈 형태 분류 (Eye Shape Classification)
468개 facial landmarks를 활용해 눈꼬리 각도를 측정하여 눈 형태를 3가지로 분류합니다.

**분류 카테고리:**
- **UPTURNED (올라간 눈)**: 눈꼬리가 내안각보다 높은 위치 (각도 > 5°)
- **DOWNTURNED (내려간 눈)**: 눈꼬리가 내안각보다 낮은 위치 (각도 < -5°)
- **NEUTRAL (기본형)**: 눈꼬리와 내안각이 거의 평행 (-5° ~ 5°)

### 2. 얼굴형 분류 (Face Shape Classification)
얼굴의 종횡비(세로/가로)와 너비 분포를 분석하여 얼굴형을 5가지로 분류합니다.

**분류 카테고리:**
- **OVAL (계란형)**: 종횡비 1.35-1.55, 부드러운 곡선, 이상적인 얼굴형
- **ROUND (둥근형)**: 종횡비 1.15-1.35, 원형에 가까운, 부드러운 인상
- **SQUARE (사각형)**: 종횡비 < 1.15, 각진 턱선, 남성적인 인상
- **HEART (하트형)**: 넓은 이마, 좁은 턱, 뾰족한 턱 끝
- **LONG (긴형)**: 종횡비 > 1.55, 매우 긴 얼굴, 날씬한 인상

---

## 📐 분석 알고리즘

### 눈 형태 분석 알고리즘

```python
# 1. 주요 landmark 포인트 추출
내안각 = landmarks[133]  # 왼쪽 눈
외안각 = landmarks[33]

# 2. 눈꼬리 각도 계산
각도 = arctan2(외안각.y - 내안각.y, 외안각.x - 내안각.x) × 180/π

# 3. 얼굴 기울기 보정
보정된_각도 = 각도 - roll_angle

# 4. 분류
if 보정된_각도 > 5°:
    → UPTURNED (올라간 눈)
elif 보정된_각도 < -5°:
    → DOWNTURNED (내려간 눈)
else:
    → NEUTRAL (기본형)
```

**사용된 Landmark 인덱스:**
| 눈 | 포인트 | Landmark Index |
|---|--------|----------------|
| 왼쪽 | 내안각 | 133 |
| 왼쪽 | 외안각 | 33 |
| 오른쪽 | 내안각 | 362 |
| 오른쪽 | 외안각 | 263 |

### 얼굴형 분석 알고리즘

```python
# 1. 얼굴 높이 측정
이마_상단 = landmarks[10]
턱_끝 = landmarks[152]
얼굴_높이 = distance(이마_상단, 턱_끝)

# 2. 얼굴 너비 측정 (3단계)
이마_너비 = distance(landmarks[21], landmarks[251])
광대_너비 = distance(landmarks[234], landmarks[454])
턱선_너비 = distance(landmarks[172], landmarks[397])

얼굴_너비 = max(이마_너비, 광대_너비, 턱선_너비)

# 3. 종횡비 계산
종횡비 = 얼굴_높이 / 얼굴_너비

# 4. 분류
if 종횡비 >= 1.55:
    → LONG (긴형)
elif 종횡비 >= 1.35:
    → OVAL (계란형)
elif 종횡비 >= 1.15:
    → ROUND (둥근형)
else:
    # 너비 분포 추가 분석
    if 이마_너비 > 광대_너비 * 1.05 and 턱선_너비 < 광대_너비 * 0.85:
        → HEART (하트형)
    else:
        → SQUARE (사각형)
```

**사용된 Landmark 인덱스:**
| 측정 부위 | 왼쪽 | 오른쪽 | 설명 |
|----------|------|--------|------|
| 이마 상단 | - | 10 | 얼굴 높이 측정 |
| 턱 끝 | - | 152 | 얼굴 높이 측정 |
| 이마 너비 | 21 | 251 | 이마 가로 너비 |
| 광대 너비 | 234 | 454 | 얼굴 가장 넓은 부분 |
| 턱선 너비 | 172 | 397 | 턱 가로 너비 |

---

## 💻 사용법

### 1. 기본 사용 (Python API)

```python
from src.config.settings import DetectionConfig
from src.core.face_detector import FaceDetector
from src.core.landmark_extractor import LandmarkExtractor
from src.processing.face_analyzer import FaceAnalyzer
from src.processing.geometry import GeometryCalculator
import cv2

# 1. 초기화
config = DetectionConfig(static_image_mode=True)
detector = FaceDetector(config)
extractor = LandmarkExtractor()
geometry_calc = GeometryCalculator()
analyzer = FaceAnalyzer()

# 2. 이미지 로드 및 검출
image = cv2.imread('face.jpg')
result = detector.detect(image)

if result.success:
    # 3. 얼굴 각도 계산 (기울기 보정용)
    face_geometry = geometry_calc.get_face_geometry(result.landmarks)

    # 4. 눈 형태 분석
    eye_analysis = analyzer.analyze_eye_shape(
        result.landmarks,
        roll_angle=face_geometry.roll
    )

    # 5. 얼굴형 분석
    face_shape_analysis = analyzer.analyze_face_shape(result.landmarks)

    # 6. 결과 출력
    print(f"눈 형태: {eye_analysis.left_eye_shape.value}")
    print(f"얼굴형: {face_shape_analysis.face_shape.value}")

detector.release()
```

### 2. 통합 분석 사용

```python
# 통합 분석 (눈 형태 + 얼굴형)
detailed_analysis = analyzer.get_detailed_analysis(
    result.landmarks,
    roll_angle=face_geometry.roll
)

# JSON 변환
import json
print(json.dumps(detailed_analysis.to_dict(), indent=2, ensure_ascii=False))
```

**JSON 출력 예시:**
```json
{
  "eye_analysis": {
    "left_eye_shape": "upturned",
    "right_eye_shape": "upturned",
    "left_eye_angle": 7.5,
    "right_eye_angle": 6.8,
    "average_eye_angle": 7.15,
    "confidence": 0.95
  },
  "face_shape_analysis": {
    "face_shape": "oval",
    "aspect_ratio": 1.42,
    "face_width": 180.5,
    "face_height": 256.3,
    "forehead_width": 160.2,
    "cheekbone_width": 180.5,
    "jawline_width": 145.7,
    "confidence": 0.9
  }
}
```

### 3. CLI 데모 실행

```bash
# 이미지 파일로 분석
python examples/face_analysis_demo.py data/sample_images/face1.jpg
```

**출력 예시:**
```
======================================================================
얼굴 분석: face1.jpg
======================================================================

✅ 이미지 로드 완료: 640x480
✅ 얼굴 검출 성공!
   - Landmarks: 468개
   - Confidence: 0.98
   - 처리 시간: 28.5ms

📐 얼굴 각도:
   - Pitch (상하): -2.3°
   - Yaw (좌우): 5.1°
   - Roll (기울기): -1.2°

👁️  눈 형태 분석:
   - 왼쪽 눈: upturned (7.50°)
   - 오른쪽 눈: upturned (6.80°)
   - 평균 각도: 7.15°
   - 신뢰도: 95.00%
   → 눈꼬리가 올라간 형태 (상승형)

🎭 얼굴형 분석:
   - 얼굴형: OVAL
   - 종횡비 (세로/가로): 1.420
   - 얼굴 너비: 180.5px
   - 얼굴 높이: 256.3px

   📏 너비 측정:
   - 이마 너비: 160.2px
   - 광대 너비: 180.5px
   - 턱선 너비: 145.7px
   - 신뢰도: 90.00%
   → 계란형 (긴 타원형, 이상적인 얼굴형)

✅ 분석 완료!
```

---

## 🔧 커스터마이징

### Threshold 조정

```python
# 눈 각도 임계값 조정 (기본: 5도)
analyzer = FaceAnalyzer(
    eye_angle_threshold=7.0,  # 더 엄격한 기준
)

# 얼굴형 종횡비 임계값 조정
analyzer = FaceAnalyzer(
    aspect_ratio_thresholds={
        'long': 1.60,    # 긴형 기준 (기본: 1.55)
        'oval': 1.40,    # 계란형 기준 (기본: 1.35)
        'round': 1.20,   # 둥근형 기준 (기본: 1.15)
    }
)
```

---

## 📊 데이터 모델

### EyeAnalysis

```python
@dataclass
class EyeAnalysis:
    left_eye_shape: EyeShape          # 왼쪽 눈 형태
    right_eye_shape: EyeShape         # 오른쪽 눈 형태
    left_eye_angle: float             # 왼쪽 눈 각도 (도)
    right_eye_angle: float            # 오른쪽 눈 각도 (도)
    average_eye_angle: float          # 평균 각도 (도)
    confidence: float = 0.95          # 분류 신뢰도
```

### FaceShapeAnalysis

```python
@dataclass
class FaceShapeAnalysis:
    face_shape: FaceShape             # 얼굴형
    aspect_ratio: float               # 종횡비 (height/width)
    face_width: float                 # 얼굴 너비 (픽셀)
    face_height: float                # 얼굴 높이 (픽셀)
    forehead_width: float = 0.0       # 이마 너비 (픽셀)
    cheekbone_width: float = 0.0      # 광대 너비 (픽셀)
    jawline_width: float = 0.0        # 턱 너비 (픽셀)
    confidence: float = 0.90          # 분류 신뢰도
```

---

## ⚠️ 제한사항 및 개선 방향

### 현재 제한사항

1. **Threshold 기반 분류**
   - 경험적 임계값 사용 (5도, 1.35 비율 등)
   - 대규모 데이터셋 기반 통계적 조정 필요

2. **얼굴 각도 제한**
   - 극단적인 각도 (pitch > 30°, yaw > 45°)에서 정확도 저하
   - 정면에 가까운 얼굴에서 최적 성능

3. **단순 기하학적 분석**
   - ML 기반 분류 모델 대비 복잡한 형태 구분 어려움
   - 중간 형태 (oval과 round 경계) 분류 모호성

### 개선 방향

**Phase 2 (정밀도 개선):**
- [ ] 다중 포인트 기반 눈 형태 분석
- [ ] 통계적 threshold 조정 (데이터셋 수집)
- [ ] 얼굴 윤곽선 곡률 분석 추가
- [ ] 턱 각도 계산 추가

**Phase 3 (고도화):**
- [ ] ML 기반 분류 모델 통합 (TensorFlow/PyTorch)
- [ ] 추가 얼굴형 타입 (다이아몬드형, 삼각형)
- [ ] 눈 형태 세부 분류 (아몬드형, 둥근형 등)
- [ ] 실시간 분석 최적화

---

## 📚 참고 자료

- [MediaPipe Face Mesh 공식 문서](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md)
- [468 Facial Landmarks 시각화](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj)
- [얼굴형 분류 연구](https://en.wikipedia.org/wiki/Face_shape)

---

**구현 일자**: 2025-10-23
**버전**: 1.0.0 (Phase 1 MVP)
**프로젝트**: Hyundai Mobis Facial Landmark Detection System
