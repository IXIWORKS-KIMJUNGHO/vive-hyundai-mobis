# Unified Face Analyzer - 설치 및 실행 가이드

## 📋 사전 요구사항

### 1. Python 설치 (필수)

**권장 버전**: Python 3.9, 3.10, 또는 3.11

> ⚠️ **중요**: Python 3.12+는 일부 라이브러리 호환성 문제가 있을 수 있습니다.

#### Windows 설치 방법

1. Python 다운로드: https://www.python.org/downloads/
2. 설치 시 **"Add Python to PATH"** 옵션 체크 필수!
3. 설치 확인:
   ```bash
   python --version
   ```

### 2. BiSeNet 모델 파일 (필수)

**다운로드**: [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)

- 파일: `79999_iter.pth` (169MB)
- 저장 위치: `unified_face_analyzer/data/79999_iter.pth`

### 3. GPU 지원 (선택)

NVIDIA GPU를 사용하는 경우:
- CUDA 11.8 또는 12.1 설치
- 설치 후 requirements.txt의 torch 부분 수정 필요

---

## 🚀 빠른 시작 (자동 설치)

### Windows

1. **`start_server.bat` 더블 클릭**

배치 파일이 자동으로:
- ✅ Python 버전 확인
- ✅ 가상환경 생성 (없는 경우)
- ✅ pip 업그레이드
- ✅ 의존성 패키지 설치
- ✅ 모델 파일 확인
- ✅ TCP 서버 실행

---

## 🔧 수동 설치 (문제 발생 시)

### 1. 가상환경 생성

```bash
python -m venv venv
```

### 2. 가상환경 활성화

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. pip 업그레이드

```bash
python -m pip install --upgrade pip
```

### 4. 의존성 설치

#### CPU 버전 (기본)
```bash
pip install -r requirements.txt
```

#### GPU 버전 (CUDA 11.8)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### GPU 버전 (CUDA 12.1)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 5. 서버 실행

```bash
python tcp_server.py
```

---

## 📡 서버 포트

| 포트 | 용도 | 설명 |
|------|------|------|
| **5001** | IR 카메라 수신 | Y8 이미지 스트리밍 수신 |
| **10000** | 분석 요청 | 얼굴 분석 요청 및 결과 전송 |
| **7001** | 실시간 뷰어 | BGR 이미지 브로드캐스트 |

포트 변경이 필요한 경우: `config.yaml` 수정

---

## 🛠️ 문제 해결

### 1. "Python이 설치되어 있지 않습니다"

**해결**:
- Python 설치 확인: `python --version`
- PATH 환경변수에 Python 추가
- 터미널 재시작

### 2. 패키지 설치 실패

**해결**:
```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 개별 설치 시도
pip install torch torchvision
pip install mediapipe
pip install transformers
```

### 3. "BiSeNet 모델 파일이 없습니다"

**해결**:
1. [GitHub](https://github.com/zllrunning/face-parsing.PyTorch)에서 `79999_iter.pth` 다운로드
2. `unified_face_analyzer/data/` 폴더에 저장
3. 파일 크기 확인: 약 169MB

### 4. 포트 충돌 (Address already in use)

**해결**:
```bash
# Windows: 포트 사용 프로세스 확인
netstat -ano | findstr :5001
netstat -ano | findstr :10000

# 프로세스 종료
taskkill /PID [프로세스ID] /F
```

### 5. CUDA 메모리 부족

**해결**:
- GPU 메모리가 부족한 경우 CPU 버전 사용
- 다른 GPU 사용 프로그램 종료
- config.yaml에서 배치 크기 조정

### 6. ModuleNotFoundError

**해결**:
```bash
# 가상환경이 활성화되었는지 확인
# 프롬프트 앞에 (venv)가 표시되어야 함

# Windows
venv\Scripts\activate

# 다시 설치
pip install -r requirements.txt --force-reinstall
```

---

## 📊 설치 확인

### 테스트 실행

```bash
# 가상환경 활성화 후
python -c "import torch; import cv2; import mediapipe; print('✅ 모든 패키지 정상')"
```

성공 시 출력: `✅ 모든 패키지 정상`

---

## 🔄 업데이트

### 의존성 업데이트

```bash
# 가상환경 활성화
venv\Scripts\activate

# 최신 패키지로 업데이트
pip install -r requirements.txt --upgrade
```

### 가상환경 재생성

문제가 계속되는 경우:

```bash
# 1. 기존 가상환경 삭제
rmdir /s /q venv

# 2. 새로 생성
python -m venv venv

# 3. 활성화 및 설치
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📞 지원

문제가 해결되지 않는 경우:

1. **로그 확인**: `logs/tcp_server.log`
2. **에러 메시지 전체 복사**
3. **Python 버전 확인**: `python --version`
4. **설치된 패키지 확인**: `pip list > installed.txt`

---

## 📝 주요 파일 구조

```
unified_face_analyzer/
├── start_server.bat        # Windows 자동 실행 스크립트
├── tcp_server.py            # TCP 서버 메인 파일
├── requirements.txt         # Python 의존성 목록
├── config.yaml              # 서버 설정 파일
├── INSTALL.md               # 이 파일
├── data/
│   └── 79999_iter.pth       # BiSeNet 모델 (수동 다운로드 필요)
├── core/                    # 얼굴 분석 코어
├── server/                  # TCP 서버 모듈
└── venv/                    # 가상환경 (자동 생성)
```

---

## ✅ 체크리스트

설치 완료 전 확인:

- [ ] Python 3.9-3.11 설치됨
- [ ] `python --version` 명령 정상 동작
- [ ] BiSeNet 모델 파일 (`data/79999_iter.pth`) 다운로드
- [ ] `start_server.bat` 실행 시 자동으로 설치 진행
- [ ] TCP 서버 정상 시작 (포트 5001, 10000, 7001 오픈)
- [ ] IR 카메라 연결 테스트 성공

---

**버전**: 1.0.0
**최종 업데이트**: 2025-01-31
