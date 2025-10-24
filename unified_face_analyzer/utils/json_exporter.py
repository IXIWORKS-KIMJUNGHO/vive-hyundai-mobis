"""
분석 결과를 언리얼 엔진용 JSON으로 변환
"""
import json
from datetime import datetime


# Enum 정의 (언리얼 엔진과 동일하게 매칭)
HAIRSTYLE_ENUM = {
    "Bangs": 0,
    "All-Back": 1,
    "Center Part": 2,
    "Right Side Part": 3,
    "Left Side Part": 4,
    "Short Hair": 5,      # 단발 (여성)
    "Long Hair": 6,       # 장발 (여성)
    "Unknown": -1
}

GENDER_ENUM = {
    "Female": 0,
    "Male": 1,
    "Unknown": -1
}


def to_unreal_json(result, image_path=""):
    """
    분석 결과를 언리얼 엔진에서 사용할 JSON으로 변환
    - Hairstyle: enum (0~4)
    - Gender: enum (0=Female, 1=Male)
    - Glasses/Beard: bool (0 or 1)

    Args:
        result: HairstyleAnalyzer.analyze_image()의 결과
        image_path: 원본 이미지 경로 (선택)

    Returns:
        dict: JSON 직렬화 가능한 딕셔너리
    """

    # 기본 정보
    classification = result.get('classification', 'Unknown')
    clip_results = result.get('clip_results', {})

    # Hairstyle enum 변환
    hairstyle_enum = HAIRSTYLE_ENUM.get(classification, -1)

    # Gender enum 변환
    gender_str = clip_results.get('gender', 'Unknown')
    gender_enum = GENDER_ENUM.get(gender_str, -1)

    # Glasses bool 변환
    glasses_str = clip_results.get('glasses', 'No Glasses')
    has_glasses = 1 if "Wearing" in glasses_str else 0

    # Beard bool 변환
    beard_str = clip_results.get('beard', 'No Beard')
    has_beard = 1 if "With" in beard_str else 0

    # JSON 구조 생성
    output = {
        # Hairstyle (enum + 원본 문자열)
        "hairstyle": hairstyle_enum,
        "hairstyle_name": classification,

        # Gender (enum + 원본 문자열)
        "gender": gender_enum,
        "gender_name": gender_str,
        "gender_confidence": float(clip_results.get('gender_confidence', 0)),

        # Glasses (bool 0/1)
        "has_glasses": has_glasses,
        "glasses_confidence": float(clip_results.get('glasses_confidence', 0)),

        # Beard (bool 0/1)
        "has_beard": has_beard,
        "beard_confidence": float(clip_results.get('beard_confidence', 0)),

        # 메타데이터
        "timestamp": datetime.now().isoformat(),
        "image_path": image_path
    }

    return output


def save_json(result, output_path, image_path=""):
    """
    분석 결과를 JSON 파일로 저장
    json_output 폴더에 저장

    Args:
        result: HairstyleAnalyzer.analyze_image()의 결과
        output_path: 저장할 JSON 파일 경로
        image_path: 원본 이미지 경로 (선택)
    """
    import os

    # output_path에서 디렉토리와 파일명 분리
    dir_path = os.path.dirname(output_path)
    filename = os.path.basename(output_path)

    # json_output 폴더 경로 생성
    json_dir = os.path.join(dir_path, 'json_output')

    # 폴더가 없으면 생성
    os.makedirs(json_dir, exist_ok=True)

    # 새로운 경로로 저장
    new_output_path = os.path.join(json_dir, filename)

    json_data = to_unreal_json(result, image_path)

    with open(new_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"[JSON] Saved to: {new_output_path}")
    return json_data


def to_json_string(result, image_path=""):
    """
    분석 결과를 JSON 문자열로 변환 (언리얼로 직접 전송용)

    Args:
        result: HairstyleAnalyzer.analyze_image()의 결과
        image_path: 원본 이미지 경로 (선택)

    Returns:
        str: JSON 문자열
    """
    json_data = to_unreal_json(result, image_path)
    return json.dumps(json_data, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # 테스트
    from hairstyle_analyzer import HairstyleAnalyzer
    import os

    analyzer = HairstyleAnalyzer()

    test_image = "camera_capture_20250513_212617.png"

    if os.path.exists(test_image):
        print(f"Testing JSON export with: {test_image}")

        result, viz = analyzer.analyze_image(test_image)

        if 'error' not in result:
            # JSON 파일로 저장
            json_output = test_image.replace('.png', '.json')
            save_json(result, json_output, test_image)

            # JSON 문자열 출력
            print("\n" + "="*60)
            print("JSON Output:")
            print("="*60)
            print(to_json_string(result, test_image))
