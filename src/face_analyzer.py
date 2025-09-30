"""IR 얼굴 특징 분석 모듈"""

import ollama
import json
import sys
from pathlib import Path
from preprocess_ir import IRImagePreprocessor


class IRFaceAnalyzer:
    """IR 이미지 얼굴 특징 분석기"""

    def __init__(
        self,
        model_name: str = "ir-face-analyzer",
        ollama_path: str = None
    ):
        self.model = model_name
        self.preprocessor = IRImagePreprocessor()
        self.ollama_path = ollama_path or self._find_ollama()

    def _find_ollama(self) -> str:
        """Ollama 실행 파일 경로 찾기"""
        import platform
        import os

        if platform.system() == "Windows":
            # Windows 기본 설치 경로
            default_path = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
            if default_path.exists():
                return str(default_path)

        # PATH에서 찾기
        return "ollama"

    def encode_image(self, image_path: str) -> bytes:
        """이미지를 바이트로 인코딩"""
        with open(image_path, 'rb') as f:
            return f.read()

    def analyze(self, image_path: str, preprocess: bool = True) -> dict:
        """
        얼굴 특징 분석

        Args:
            image_path: IR 이미지 경로
            preprocess: 전처리 수행 여부

        Returns:
            얼굴 특징 분석 결과 (dict)
        """
        # 전처리
        temp_path = None
        if preprocess:
            temp_path = "temp_preprocessed.jpg"
            self.preprocessor.process(image_path, temp_path)
            image_path = temp_path

        try:
            # 이미지 로드
            image_data = self.encode_image(image_path)

            # Ollama API 호출
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'user',
                        'content': '이 IR 이미지에서 얼굴 특징을 분석해주세요.',
                        'images': [image_data]
                    }
                ],
                options={
                    'temperature': 0.2,
                    'num_predict': 1024
                }
            )

            # JSON 파싱
            content = response['message']['content']

            # 코드 블록 제거 (```json ... ```)
            if content.startswith('```'):
                lines = content.split('\n')
                content = '\n'.join(lines[1:-1])  # 첫 줄과 마지막 줄 제거

            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # 재시도: 중괄호 사이 추출
                import re
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    try:
                        result = json.loads(match.group(0))
                    except:
                        result = {
                            "raw_output": content,
                            "parse_error": True
                        }
                else:
                    result = {
                        "raw_output": content,
                        "parse_error": True
                    }

            return result

        finally:
            # 임시 파일 삭제
            if temp_path and Path(temp_path).exists():
                Path(temp_path).unlink()

    def batch_analyze(self, image_paths: list) -> list:
        """여러 이미지 일괄 분석"""
        results = []
        for i, path in enumerate(image_paths):
            print(f"분석 중... ({i+1}/{len(image_paths)}): {path}")
            result = self.analyze(path)
            result['image_path'] = path
            results.append(result)
        return results


def main():
    """CLI 인터페이스"""
    import sys
    import io

    # Windows 콘솔 UTF-8 인코딩 설정
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    if len(sys.argv) < 2:
        print("사용법: python face_analyzer.py <이미지_경로>")
        sys.exit(1)

    analyzer = IRFaceAnalyzer()
    result = analyzer.analyze(sys.argv[1])

    print("\n=== 얼굴 특징 분석 결과 ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()