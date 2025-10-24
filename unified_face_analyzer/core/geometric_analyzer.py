"""
Stage 2: Geometric analysis for non-bangs hairstyles (All-Back, Center Part, Side Part).
[MODIFIED] Uses BiSeNet hair mask instead of brightness-based detection.
"""
import cv2
import numpy as np
from utils import get_config
from utils import get_logger

logger = get_logger(__name__)

class GeometricAnalyzer:
    def __init__(self):
        self.config = get_config()
        self.results = {}

    def analyze(self, img_gray, face, landmarks, person_silhouette, hair_mask=None, eyebrow_mask=None):
        """
        Performs the detailed geometric analysis on the forehead region.
        [MODIFIED] Now accepts 'hair_mask' and 'eyebrow_mask' from BiSeNet for accurate detection.
        
        Args:
            img_gray: Grayscale image
            face: dlib face rectangle
            landmarks: dlib facial landmarks
            person_silhouette: Person silhouette mask
            hair_mask: BiSeNet hair segmentation mask
            eyebrow_mask: BiSeNet eyebrow segmentation mask
        """
        logger.info("[Stage 2] Performing geometric analysis with BiSeNet hair mask...")
        
        if person_silhouette is None:
            logger.error("Error: Received an empty silhouette mask.")
            return "Unknown", self.results

        # BiSeNet 헤어 마스크를 사용한 이마 검출
        hair_region, forehead_mask, coverage = self._detect_forehead_with_hair_mask(
            img_gray, person_silhouette, face, landmarks, hair_mask, eyebrow_mask
        )

        min_forehead_coverage = self.config.geometric.min_forehead_coverage
        if forehead_mask is None or coverage < min_forehead_coverage:
            logger.info(f"No forehead skin detected (coverage: {coverage:.1%}, hair covers entire forehead).")
            return "Bangs", self.results

        forehead_ratio = self.config.bangs_detection.forehead_height_ratio
        left_ratio, right_ratio, symmetry = self._analyze_left_right(
            forehead_mask, face, hair_mask, forehead_top=max(0, landmarks.parts()[17].y - int((face.height() * forehead_ratio)))
        )
        classification = self._classify_style(coverage, left_ratio, right_ratio, symmetry)
        
        # 상세 결과 저장
        self.results['hair'] = {'region': hair_region}
        self.results['forehead'] = {'mask': forehead_mask, 'coverage': coverage}
        self.results['left_right'] = {'left_ratio': left_ratio, 'right_ratio': right_ratio, 'symmetry': symmetry}
        self.results['classification'] = {'scores': classification['scores']}
        self.results['clean_silhouette'] = person_silhouette
        self.results['hair_mask'] = hair_mask

        return classification['top_style'], self.results

    def _detect_forehead_with_hair_mask(self, img_gray, person_mask, face, landmarks, hair_mask, eyebrow_mask):
        """
        [NEW] Uses BiSeNet hair mask and eyebrow mask to detect forehead area.
        Forehead = (Person silhouette) - (Hair mask) - (Eyebrow mask) in the forehead region
        """
        logger.debug("Step 1: Using BiSeNet hair mask for forehead detection...")
        
        # BiSeNet 눈썹 마스크로 정확한 이마 하단 경계 찾기
        x_min, x_max = face.left(), face.right()
        eyebrow_points = np.array([[p.x, p.y] for p in landmarks.parts()[17:27]])
        dlib_y_max = int(np.min(eyebrow_points[:, 1]))
        
        # 눈썹 영역에서 실제 눈썹의 최상단 찾기
        if eyebrow_mask is not None:
            search_top = max(0, dlib_y_max - int(face.height() * 0.2))
            search_bottom = min(eyebrow_mask.shape[0], dlib_y_max + int(face.height() * 0.1))
            eyebrow_region = eyebrow_mask[search_top:search_bottom, x_min:x_max]
            
            eyebrow_y_coords = np.where(eyebrow_region > 0)[0]
            
            if len(eyebrow_y_coords) > 0:
                eyebrow_top_in_region = np.min(eyebrow_y_coords)
                y_max = search_top + eyebrow_top_in_region
                logger.debug(f"BiSeNet eyebrow top found at y={y_max}")
            else:
                y_max = dlib_y_max
                logger.debug(f"No eyebrow found in BiSeNet, using dlib y={y_max}")
        else:
            y_max = dlib_y_max
            logger.debug(f"No eyebrow mask, using dlib y={y_max}")
        
        forehead_ratio = self.config.bangs_detection.forehead_height_ratio
        forehead_top = max(0, y_max - int((face.height() * forehead_ratio)))
        
        # 분석할 이마 영역만 잘라내기
        hair_region_crop = img_gray[forehead_top:y_max, x_min:x_max]
        person_mask_crop = person_mask[forehead_top:y_max, x_min:x_max]
        
        if hair_region_crop.size == 0:
            return None, None, 0.0
        
        # BiSeNet 헤어 마스크 사용
        if hair_mask is not None:
            # 전체 이미지의 hair_mask에서 forehead 영역만 crop
            hair_mask_crop = hair_mask[forehead_top:y_max, x_min:x_max]
            
            # 눈썹 마스크도 crop
            if eyebrow_mask is not None:
                eyebrow_mask_crop = eyebrow_mask[forehead_top:y_max, x_min:x_max]
                # 헤어와 눈썹을 합침
                combined_exclude = cv2.bitwise_or(hair_mask_crop, eyebrow_mask_crop)
            else:
                combined_exclude = hair_mask_crop
            
            # 이마 = 사람 영역 - (헤어 + 눈썹) [순수 BiSeNet 기반 판정]
            forehead_skin_mask = cv2.bitwise_and(person_mask_crop, cv2.bitwise_not(combined_exclude))

            # 최소한의 노이즈 제거만 (BiSeNet 마스크를 신뢰)
            kernel_size = self.config.morphology.forehead_mask.open_kernel_size
            iterations = self.config.morphology.forehead_mask.open_iterations
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            # Opening으로 작은 노이즈만 제거
            forehead_skin_mask = cv2.morphologyEx(forehead_skin_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
            
        else:
            # BiSeNet 마스크가 없으면 기존 방식 사용 (fallback)
            logger.warning("No BiSeNet hair mask provided, using fallback method...")
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast_enhanced_img = clahe.apply(hair_region_crop)
            _, skin_mask_otsu = cv2.threshold(contrast_enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            forehead_skin_mask = cv2.bitwise_and(skin_mask_otsu, person_mask_crop)
        
        # Coverage 계산
        person_pixels = np.sum(person_mask_crop > 0)
        forehead_pixels = np.sum(forehead_skin_mask > 0)
        coverage = forehead_pixels / (person_pixels or 1)
        
        logger.debug(f"Forehead coverage (BiSeNet-based): {coverage:.1%}")
        logger.debug(f"Forehead pixels: {forehead_pixels}, Person pixels: {person_pixels}")
        
        return hair_region_crop, forehead_skin_mask, coverage

    def _analyze_left_right(self, forehead_mask, face, hair_mask, forehead_top):
        """
        Analyzes left/right symmetry of the forehead mask.
        [MODIFIED] Now uses full hair mask extent instead of cropped forehead dimensions.

        Args:
            forehead_mask: Cropped forehead skin mask
            face: dlib face rectangle
            hair_mask: Full BiSeNet hair mask
            forehead_top: Y-coordinate of forehead top in original image
        """
        h, w = forehead_mask.shape

        # 피부가 있는 픽셀들의 x 좌표 추출
        skin_coords = np.where(forehead_mask > 0)

        if len(skin_coords[1]) == 0:
            return 0.5, 0.5, 0.0

        # 피부 픽셀들의 x좌표 중앙값 (가르마 중심선)
        skin_center_x = int(np.median(skin_coords[1]))

        # 전체 헤어 마스크에서 이마 영역의 실제 헤어 범위 계산
        x_min, x_max = face.left(), face.right()
        forehead_bottom = forehead_top + h

        if hair_mask is not None:
            # 이마 영역의 헤어 마스크 추출
            hair_forehead_region = hair_mask[forehead_top:forehead_bottom, :]

            # 헤어가 존재하는 x 좌표 범위 찾기
            hair_coords = np.where(hair_forehead_region > 0)

            if len(hair_coords[1]) > 0:
                hair_x_min = np.min(hair_coords[1])
                hair_x_max = np.max(hair_coords[1])

                # 헤어 범위의 중심 = 실제 얼굴 중심
                hair_center_x = (hair_x_min + hair_x_max) // 2

                # Cropped 좌표계에서 실제 얼굴 중심 위치 계산
                face_center_in_crop = hair_center_x - x_min

                logger.debug(f"Hair x-range in full image: {hair_x_min} ~ {hair_x_max}")
                logger.debug(f"Hair center (actual face center): {hair_center_x}")
                logger.debug(f"Face center in cropped coords: {face_center_in_crop}")
            else:
                # 헤어가 없으면 face bbox 중심 사용
                face_center_in_crop = w // 2
                logger.debug(f"No hair found, using face bbox center: {face_center_in_crop}")
        else:
            # hair_mask가 없으면 기존 방식 (face bbox 중심)
            face_center_in_crop = w // 2
            logger.debug(f"No hair mask, using face bbox center: {face_center_in_crop}")

        logger.debug(f"Skin pixel x-range (in crop): {np.min(skin_coords[1])} ~ {np.max(skin_coords[1])}")
        logger.debug(f"Skin center (parting, in crop): {skin_center_x}")
        logger.debug(f"Offset from actual face center: {skin_center_x - face_center_in_crop}px")

        # 실제 얼굴 중심 기준 좌우 분할
        left_pixels = np.sum(forehead_mask[:, 0:face_center_in_crop] > 0)
        right_pixels = np.sum(forehead_mask[:, face_center_in_crop:] > 0)
        total_pixels = left_pixels + right_pixels

        if total_pixels == 0:
            return 0.5, 0.5, 0.0

        left_ratio = left_pixels / total_pixels
        right_ratio = right_pixels / total_pixels
        symmetry = abs(left_ratio - right_ratio)

        logger.debug(f"Left (from actual face center): {left_ratio:.1%}, Right: {right_ratio:.1%}")
        logger.debug(f"Symmetry diff: {symmetry:.2f}")

        return left_ratio, right_ratio, symmetry

    def _classify_style(self, coverage, left_ratio, right_ratio, symmetry):
        """
        Classifies the hairstyle based on geometric properties.
        Differentiates between Left and Right Side Part.
        """
        logger.debug("Classifying style based on coverage and symmetry...")

        results = {'All-Back': 0.0, 'Center Part': 0.0, 'Left Side Part': 0.0, 'Right Side Part': 0.0}

        # 가르마 방향 결정
        side_part_style = 'Right Side Part' if right_ratio > left_ratio else 'Left Side Part'

        # Load thresholds from config
        allback_cov_min = self.config.geometric.style_classification.allback_coverage_min
        allback_sym_max = self.config.geometric.style_classification.allback_symmetry_max
        center_cov_min = self.config.geometric.style_classification.centerpart_coverage_min
        center_cov_max = self.config.geometric.style_classification.centerpart_coverage_max
        center_sym_max = self.config.geometric.style_classification.centerpart_symmetry_max
        side_cov_min = self.config.geometric.style_classification.sidepart_coverage_min
        side_cov_max = self.config.geometric.style_classification.sidepart_coverage_max
        side_sym_min = self.config.geometric.style_classification.sidepart_symmetry_min

        if coverage > allback_cov_min and symmetry < allback_sym_max:
            results['All-Back'] = 0.8
        elif symmetry < center_sym_max and center_cov_min <= coverage <= center_cov_max:
            results['Center Part'] = 0.8
        elif symmetry >= side_sym_min and side_cov_min <= coverage <= side_cov_max:
            results[side_part_style] = 0.6
        else:
            if coverage > 0.2:
                style = 'Center Part' if symmetry < center_sym_max else side_part_style
                results[style] = 0.5
            else:
                style = 'Center Part' if symmetry < center_sym_max else side_part_style
                results[style] = 0.3

        total = sum(results.values())
        if total > 0:
            results = {k: v / total for k, v in results.items() if v > 0}
        
        top_style = max(results, key=results.get) if results else "Unknown"
        logger.info(f"Geometric Conclusion: {top_style} (scores: {results})")
        return {'scores': results, 'top_style': top_style}

    def analyze_hair_length(self, face, landmarks, hair_mask):
        """
        Analyzes hair length for female subjects.
        Checks if hair extends across most of the image height (top to bottom coverage).

        Args:
            face: dlib face rectangle
            landmarks: dlib facial landmarks (68-point)
            hair_mask: BiSeNet hair segmentation mask

        Returns:
            str: "Short Hair" or "Long Hair"
        """
        logger.info("[Hair Length] Analyzing hair length for female subject...")

        if hair_mask is None:
            logger.warning("No hair mask available, defaulting to Short Hair")
            return "Short Hair"

        # 이미지 전체 높이
        image_height = hair_mask.shape[0]

        # 헤어 마스크의 상단과 하단 y 좌표 찾기
        hair_y_coords = np.where(hair_mask > 0)[0]

        if len(hair_y_coords) == 0:
            logger.warning("No hair detected in mask, defaulting to Short Hair")
            return "Short Hair"

        hair_bottom_y = np.max(hair_y_coords)

        # 이미지 전체 대비 헤어 최하단 위치 비율
        bottom_ratio = hair_bottom_y / image_height  # 1에 가까울수록 아래쪽

        logger.debug(f"Image height: {image_height}px")
        logger.debug(f"Hair bottom: y={hair_bottom_y} ({bottom_ratio:.1%} from top)")

        # 임계값: 헤어 하단이 이미지의 threshold% 아래까지 내려가면 장발
        long_hair_threshold = self.config.geometric.long_hair_bottom_threshold

        if bottom_ratio > long_hair_threshold:
            result = "Long Hair"
            logger.info(f"Classification: Long Hair (bottom {bottom_ratio:.1%} > {long_hair_threshold:.1%})")
        else:
            result = "Short Hair"
            logger.info(f"Classification: Short Hair (bottom {bottom_ratio:.1%} <= {long_hair_threshold:.1%})")

        return result