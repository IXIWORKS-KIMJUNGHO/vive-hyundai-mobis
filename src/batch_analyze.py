"""배치 이미지 분석 스크립트"""

import sys
import json
import time
from pathlib import Path
from face_analyzer import IRFaceAnalyzer


def batch_analyze_images(image_dir: str, output_dir: str = "../results"):
    """
    디렉토리 내 모든 이미지를 분석하고 결과를 저장

    Args:
        image_dir: 이미지가 있는 디렉토리
        output_dir: 결과를 저장할 디렉토리
    """
    # 결과 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 이미지 파일 찾기
    image_dir_path = Path(image_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(image_dir_path.glob(f'*{ext}'))
        image_files.extend(image_dir_path.glob(f'*{ext.upper()}'))

    image_files = sorted(set(image_files))

    print(f"✅ 발견된 이미지: {len(image_files)}개")
    print(f"📁 결과 저장 위치: {output_path.absolute()}")
    print("=" * 60)

    # 분석기 초기화
    analyzer = IRFaceAnalyzer()

    # 배치 분석
    results = []
    start_time = time.time()

    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 분석 중: {image_file.name}")

        try:
            # 분석 수행
            result = analyzer.analyze(str(image_file))
            result['image_name'] = image_file.name
            result['status'] = 'success'

            # 개별 결과 저장
            result_file = output_path / f"{image_file.stem}_result.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # 간단한 요약 출력
            print(f"   성별: {result.get('gender', 'N/A')}")
            print(f"   안경: {result.get('glasses', 'N/A')}")
            print(f"   수염: {result.get('beard', {}).get('type', 'N/A')}")
            print(f"   앞머리: {result.get('hair_style', 'N/A')}")
            print(f"   신뢰도: {result.get('confidence', 0):.2f}")
            print(f"   ✅ 저장: {result_file.name}")

            results.append(result)

        except Exception as e:
            print(f"   ❌ 오류: {str(e)}")
            results.append({
                'image_name': image_file.name,
                'status': 'error',
                'error': str(e)
            })

    # 전체 소요 시간
    elapsed_time = time.time() - start_time

    # 통합 결과 저장
    summary = {
        'total_images': len(image_files),
        'successful': sum(1 for r in results if r.get('status') == 'success'),
        'failed': sum(1 for r in results if r.get('status') == 'error'),
        'elapsed_time_seconds': round(elapsed_time, 2),
        'average_time_per_image': round(elapsed_time / len(image_files), 2) if image_files else 0,
        'results': results
    }

    summary_file = output_path / "batch_analysis_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 최종 요약 출력
    print("\n" + "=" * 60)
    print("📊 배치 분석 완료")
    print("=" * 60)
    print(f"총 이미지: {summary['total_images']}개")
    print(f"성공: {summary['successful']}개")
    print(f"실패: {summary['failed']}개")
    print(f"총 소요 시간: {summary['elapsed_time_seconds']}초")
    print(f"평균 시간: {summary['average_time_per_image']}초/이미지")
    print(f"\n📁 통합 결과: {summary_file}")


if __name__ == "__main__":
    import sys
    import io

    # Windows 콘솔 UTF-8 인코딩 설정
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    image_dir = sys.argv[1] if len(sys.argv) > 1 else "../test_images"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "../results"

    batch_analyze_images(image_dir, output_dir)