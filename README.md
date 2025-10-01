# IR 얼굴 특징 분석 시스템

Qwen2.5 Vision 모델을 활용한 IR(적외선) 카메라 얼굴 특징 자동 추출 시스템

## 📋 분석 항목

시스템은 IR 이미지에서 다음 특징들을 자동으로 추출합니다:

- ✅ **얼굴형**: 둥근형, 긴형, 각진형, 계란형
- ✅ **눈**: 크기, 모양, 위치
- ✅ **코**: 높이, 폭, 형태
- ✅ **입**: 크기, 형태, 위치
- ✅ **안경 착용 여부**: 착용/미착용
- ✅ **수염**: 유무 및 종류 (턱수염/콧수염)
- ✅ **성별**: 남성/여성
- ✅ **앞머리 스타일**: 덮힘머리, 올백, 2:8, 4:6, 5:5, 6:4, 8:2

---

## 🚀 빠른 시작 (클라이언트용)

### 사전 준비

시스템에 다음이 설치되어 있어야 합니다:
- **Python 3.8 이상**
- **NVIDIA GPU** (8GB+ VRAM 권장)
- **10GB 이상 디스크 여유 공간**

---

## 📥 1단계: Ollama 설치

### Windows
1. https://ollama.com/download/windows 에서 다운로드
2. 설치 프로그램 실행
3. 설치 완료 후 확인:
   ```cmd
   ollama --version
   ```

### Linux/macOS
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

---

## 📥 2단계: Python 환경 설정

### Python 가상환경 생성

**Windows:**
```cmd
# 프로젝트 폴더로 이동
cd d:\coding\vive-hyundai-mobis

# 가상환경 생성
python -m venv venv

# 가상환경 활성화
venv\Scripts\activate

# 필수 라이브러리 설치
pip install -r requirements.txt
```

**Linux/macOS:**
```bash
# 프로젝트 폴더로 이동
cd /path/to/vive-hyundai-mobis

# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 필수 라이브러리 설치
pip install -r requirements.txt
```

### 설치 확인

```bash
pip list | grep ollama
# 출력 예시: ollama    0.6.0
```

---

## 📥 3단계: AI 모델 다운로드

### Qwen2.5 Vision 모델 다운로드

```bash
# 7B 모델 다운로드 (약 6GB, 10-20분 소요)
ollama pull qwen2.5vl:7b

# 다운로드 확인
ollama list
```

**출력 예시:**
```
NAME              ID              SIZE      MODIFIED
qwen2.5vl:7b      5ced39dfa4ba    6.0 GB    2 hours ago
```

### 커스텀 얼굴 분석 모델 생성

```bash
# 커스텀 모델 생성
ollama create ir-face-analyzer -f models/ir-face-analyzer.modelfile

# 생성 확인
ollama list | grep ir-face-analyzer
```

---

## 🧪 4단계: 테스트

### 방법 1: 단일 이미지 분석

```bash
# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (Linux/macOS)
source venv/bin/activate

# 이미지 분석 실행
cd src
python face_analyzer.py ../test_images/your_image.png
```

**출력 예시:**
```json
{
  "face_shape": "긴형",
  "eyes": {"size": "중", "shape": "원형", "position": "중앙"},
  "nose": {"height": "중간", "width": "중", "shape": "직선형"},
  "mouth": {"size": "작고", "shape": "직선형", "position": "중간"},
  "glasses": "착용",
  "beard": {"has_beard": true, "type": "턱수염"},
  "gender": "남성",
  "hair_style": "올백",
  "confidence": 0.95
}
```

### 방법 2: 배치 분석 (여러 이미지)

```bash
# 가상환경 활성화
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# 폴더 내 모든 이미지 분석
cd src
python batch_analyze.py ../test_images ../results
```

**출력:**
- 개별 결과: `results/이미지이름_result.json`
- 통합 결과: `results/batch_analysis_summary.json`

---

## 📊 5단계: 결과 확인

### 개별 결과 파일

```bash
# Windows
type results\camera_capture_20250513_184758_result.json

# Linux/macOS
cat results/camera_capture_20250513_184758_result.json
```

### 통합 요약 보고서

```bash
# Windows
type results\batch_analysis_summary.json

# Linux/macOS
cat results/batch_analysis_summary.json
```

**통합 보고서 내용:**
- 총 분석 이미지 수
- 성공/실패 건수
- 평균 처리 시간
- 전체 결과 목록

---

## 📁 프로젝트 구조

```
vive-hyundai-mobis/
├── src/                    # 소스 코드
│   ├── face_analyzer.py    # 단일 이미지 분석
│   ├── batch_analyze.py    # 배치 분석
│   └── preprocess_ir.py    # 이미지 전처리
├── models/                 # Ollama Modelfile
│   └── ir-face-analyzer.modelfile
├── test_images/            # 테스트 이미지
├── results/                # 분석 결과 저장
├── docs/                   # 상세 문서
├── requirements.txt        # Python 의존성
└── README.md              # 이 파일
```

---

## 🔧 트러블슈팅

### 문제 1: "ollama: command not found"

**해결:**
- Windows: https://ollama.com/download/windows 에서 수동 설치
- Linux/macOS: `curl -fsSL https://ollama.com/install.sh | sh`

### 문제 2: "CUDA out of memory"

**해결:**
```bash
# 더 작은 양자화 모델 사용 (4GB VRAM)
ollama pull qwen2.5vl:7b-q4_0
```

### 문제 3: JSON 파싱 에러

**원인:** 모델이 JSON 형식을 정확히 출력하지 않음
**해결:** 스크립트가 자동으로 처리하지만, 실패 시 `raw_output` 필드 확인

### 문제 4: 느린 처리 속도

**개선 방법:**
- 이미지 해상도 낮추기 (512x512 권장)
- GPU 드라이버 업데이트
- 배치 처리 시 `--concurrency` 옵션 조정

---

## 📈 성능 벤치마크

**테스트 환경**: RTX 2080 8GB

| 항목 | 측정값 |
|------|--------|
| VRAM 사용량 | 6-7GB |
| 단일 이미지 분석 시간 | 1-2초 |
| 배치 42개 분석 시간 | 50초 (평균 1.19초/이미지) |
| JSON 파싱 성공률 | 100% |
| 평균 신뢰도 | 0.95 (95%) |

---

## 🆘 지원

문제가 발생하면:
1. `results/` 폴더의 로그 확인
2. GPU 메모리 확인: `nvidia-smi`
3. Ollama 상태 확인: `ollama list`

---

## 📝 라이센스

### Qwen 2.5 VL-7B: Apache 2.0 License

✅ **상업적 사용 가능**
- 무제한 상업적 이용 허용
- 수정 및 재배포 가능
- 사용자 수 제한 없음
- 추가 비용 없음

**준수 사항:**
- 원본 라이센스 고지 포함
- Apache 2.0 라이센스 텍스트 포함
- 수정 시 수정 사항 명시

**라이센스 전문:** https://github.com/QwenLM/Qwen2.5-VL/blob/main/LICENSE

### Ollama: MIT License

✅ **상업적 사용 가능**

---

## ⚖️ 상업적 사용 가이드

본 프로젝트는 **Apache 2.0 라이센스**로, 상업적 사용이 완전히 허용됩니다.

**허용 사항:**
- ✅ 상업적 서비스 제공
- ✅ 제품에 통합
- ✅ SaaS 형태 제공
- ✅ 클라이언트 프로젝트 사용
- ✅ 수정 및 재배포

**준수 사항:**
1. Apache 2.0 라이센스 고지 포함
2. 원저작자 명시
3. 수정 사항 문서화 (수정한 경우)

**참고:**
- 모델 크기별 라이센스 차이: 3B(연구용), 7B(상업용), 72B(제한적 상업용)
- 본 프로젝트는 **7B 모델** 사용으로 상업적 제약 없음

---

**버전**: 1.0.0
**업데이트**: 2025-09-30
**문의**: 프로젝트 관리자
