# hairstyle_analyzer.py

import os
import traceback
import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

from .bisenet import BiSeNet
from .clip_classifier import CLIPClassifier
from .geometric_analyzer import GeometricAnalyzer
from .mediapipe import MediaPipeFaceDetector, MediaPipeShapePredictor
from utils import get_config, get_logger

logger = get_logger(__name__)

class HairstyleAnalyzer:
    def __init__(self):
        logger.info("Loading all models for HairstyleAnalyzer...")

        # Load configuration
        self.config = get_config()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # MediaPipe face detector & landmark predictor (dlib 대체)
        logger.info("Initializing MediaPipe face detector (replacing dlib)...")
        self.face_detector = MediaPipeFaceDetector(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        self.shape_predictor = MediaPipeShapePredictor(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        logger.info("MediaPipe models initialized successfully (68-point landmarks)")

        self.clip_classifier = CLIPClassifier(self.device)

        # Load BiSeNet model from config
        logger.info("Loading BiSeNet face-parsing model...")
        bisenet_model_path = self.config.models.bisenet_weights
        if not os.path.exists(bisenet_model_path):
            raise FileNotFoundError(f"BiSeNet model not found at '{bisenet_model_path}'.")

        self.seg_model = BiSeNet(n_classes=self.config.bisenet.n_classes)
        self.seg_model.load_state_dict(torch.load(bisenet_model_path, map_location=self.device))
        self.seg_model.to(self.device)
        self.seg_model.eval()

        logger.info(f"All models loaded successfully on device: {self.device}")

    def run_segmentation(self, pil_img):
        logger.debug("Running BiSeNet model for face/hair segmentation...")

        # [추가] 전처리: 대비 향상 및 밝기 조정
        logger.debug("Applying preprocessing for better hair detection...")
        
        # PIL을 numpy로 변환
        img_np = np.array(pil_img)
        
        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
        # RGB 각 채널에 대해 적용
        clahe = cv2.createCLAHE(
            clipLimit=self.config.preprocessing.clahe.clip_limit,
            tileGridSize=tuple(self.config.preprocessing.clahe.tile_grid_size)
        )
        
        # RGB를 LAB 색공간으로 변환 (밝기 채널만 향상)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # L 채널에만 CLAHE 적용
        l_clahe = clahe.apply(l)
        
        # 다시 합치고 RGB로 변환
        lab_clahe = cv2.merge([l_clahe, a, b])
        img_enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        # 2. 선명도 향상 (Sharpening)
        kernel_sharpening = np.array([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]])
        img_sharpened = cv2.filter2D(img_enhanced, -1, kernel_sharpening)
        
        # 3. 원본과 블렌딩 (너무 강하지 않게)
        alpha = self.config.preprocessing.sharpen_alpha  # 전처리 강도
        img_preprocessed = cv2.addWeighted(img_enhanced, alpha, img_np, 1 - alpha, 0)
        
        # numpy를 다시 PIL로 변환
        pil_img_preprocessed = Image.fromarray(img_preprocessed.astype('uint8'))
        
        # BiSeNet 입력을 위한 transform
        transform = transforms.Compose([
            transforms.Resize((self.config.bisenet.input_size, self.config.bisenet.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.bisenet.normalize_mean,
                std=self.config.bisenet.normalize_std
            )
        ])
        
        original_size = pil_img.size  # 원본 크기 유지
        input_tensor = transform(pil_img_preprocessed).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.seg_model(input_tensor)[0]
            prediction = output.squeeze(0).cpu().numpy().argmax(0)
        
        person_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]
        person_mask_bool = np.isin(prediction, person_classes)
        
        person_mask_resized = cv2.resize(
            person_mask_bool.astype(np.uint8),
            original_size,
            interpolation=cv2.INTER_NEAREST
        )
        
        person_silhouette_mask = (person_mask_resized * 255).astype(np.uint8)
        
        hair_mask_bool = (prediction == 17)
        hair_mask_resized = cv2.resize(
            hair_mask_bool.astype(np.uint8),
            original_size,
            interpolation=cv2.INTER_NEAREST
        )
        self.hair_only_mask = (hair_mask_resized * 255).astype(np.uint8)
        
        # [추가] Hair mask 후처리: Morphological operations으로 노이즈 제거 및 빈 공간 채우기
        kernel_close = np.ones(
            (self.config.morphology.hair_mask.close_kernel_size,
             self.config.morphology.hair_mask.close_kernel_size),
            np.uint8
        )
        kernel_open = np.ones(
            (self.config.morphology.hair_mask.open_kernel_size,
             self.config.morphology.hair_mask.open_kernel_size),
            np.uint8
        )
        # Closing: 작은 구멍 메우기
        self.hair_only_mask = cv2.morphologyEx(
            self.hair_only_mask,
            cv2.MORPH_CLOSE,
            kernel_close,
            iterations=self.config.morphology.hair_mask.close_iterations
        )
        # Opening: 작은 노이즈 제거
        self.hair_only_mask = cv2.morphologyEx(
            self.hair_only_mask,
            cv2.MORPH_OPEN,
            kernel_open,
            iterations=self.config.morphology.hair_mask.open_iterations
        )
        
        # 눈썹 마스크 추출 (class 2, 3)
        eyebrow_mask_bool = np.isin(prediction, [2, 3])
        eyebrow_mask_resized = cv2.resize(
            eyebrow_mask_bool.astype(np.uint8),
            original_size,
            interpolation=cv2.INTER_NEAREST
        )
        self.eyebrow_mask = (eyebrow_mask_resized * 255).astype(np.uint8)
        
        return person_silhouette_mask

    def _detect_bangs_with_bisenet(self, face, landmarks):
        if self.hair_only_mask is None:
            logger.warning("[Bangs Detection] No hair mask available")
            return False
        
        # BiSeNet 눈썹 마스크로 정확한 이마 하단 경계 찾기
        x_min, x_max = face.left(), face.right()
        eyebrow_points = np.array([[p.x, p.y] for p in landmarks.parts()[17:27]])
        dlib_y_max = int(np.min(eyebrow_points[:, 1]))
        
        # 눈썹 영역에서 실제 눈썹의 최상단 찾기
        if hasattr(self, 'eyebrow_mask') and self.eyebrow_mask is not None:
            # 눈썹 위쪽 영역 검색 (dlib 기준에서 위아래로 여유 공간)
            search_top = max(0, dlib_y_max - int(face.height() * 0.2))
            search_bottom = min(self.eyebrow_mask.shape[0], dlib_y_max + int(face.height() * 0.1))
            eyebrow_region = self.eyebrow_mask[search_top:search_bottom, x_min:x_max]
            
            # 눈썹이 있는 y 좌표들 찾기
            eyebrow_y_coords = np.where(eyebrow_region > 0)[0]
            
            if len(eyebrow_y_coords) > 0:
                # 눈썹의 최상단 (가장 작은 y 값)
                eyebrow_top_in_region = np.min(eyebrow_y_coords)
                y_max = search_top + eyebrow_top_in_region
                logger.debug(f"[Bangs Detection] BiSeNet eyebrow top found at y={y_max}")
            else:
                # 눈썹을 못 찾으면 dlib 사용
                y_max = dlib_y_max
                logger.debug(f"[Bangs Detection] No eyebrow found in BiSeNet, using dlib y={y_max}")
        else:
            y_max = dlib_y_max
            logger.debug(f"[Bangs Detection] No eyebrow mask, using dlib y={y_max}")
        
        forehead_top = max(0, y_max - int((face.height() * self.config.bangs_detection.forehead_height_ratio)))
        
        forehead_region = self.hair_only_mask[forehead_top:y_max, x_min:x_max]
        
        if forehead_region.size == 0:
            logger.warning("[Bangs Detection] Empty forehead region")
            return False
        
        total_pixels = forehead_region.size
        hair_pixels = np.sum(forehead_region > 0)
        hair_ratio = hair_pixels / total_pixels

        bangs_threshold = self.config.bangs_detection.hair_coverage_threshold

        logger.debug(f"[Bangs Detection] Hair coverage in forehead: {hair_ratio:.1%}")
        logger.debug(f"[Bangs Detection] Threshold: {bangs_threshold:.1%}")

        if hair_ratio > bangs_threshold:
            logger.info(f"BiSeNet Conclusion: BANGS DETECTED (coverage: {hair_ratio:.1%})")
            return True
        else:
            logger.info(f"BiSeNet Conclusion: NO BANGS (coverage: {hair_ratio:.1%})")
            return False

    def analyze_image(self, image_path):
        try:
            logger.info(f"Analyzing: {os.path.basename(image_path)}")
            
            stream = open(image_path, "rb")
            bytes = bytearray(stream.read())
            numpy_array = np.asarray(bytes, dtype=np.uint8)
            img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

            if img is None: 
                raise ValueError("Could not read image or image is corrupted.")
            
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pil_img = Image.open(image_path).convert("RGB")

            # MediaPipe 얼굴 검출 (dlib 대체)
            faces = self.face_detector(img_gray, 1)
            if not faces:
                raise ValueError("No face detected by MediaPipe.")
            face = max(faces, key=lambda rect: rect.width() * rect.height())

            # MediaPipe 68점 랜드마크 추출 (dlib 대체)
            landmarks = self.shape_predictor(img_gray, face)
            if landmarks is None:
                raise ValueError("Failed to extract facial landmarks.")

            # Stage 1: BiSeNet 세그멘테이션
            logger.info("[Stage 1] Running BiSeNet segmentation...")
            person_silhouette = self.run_segmentation(pil_img)

            # CLIP으로 성별, 안경, 수염 검출 (성별 우선 판정)
            clip_additional = self.clip_classifier.classify_accessories_only(pil_img, face, landmarks)

            analysis_data = {}
            analysis_data['face_rect'] = face
            analysis_data['landmarks'] = landmarks
            analysis_data['clean_silhouette'] = person_silhouette
            analysis_data['hair_mask'] = self.hair_only_mask

            final_classification = "Unknown"
            viz_image = None

            # 성별에 따라 분석 경로 분기
            if clip_additional['gender'] == 'Female':
                # 여성: 헤어 길이만 분석 (Geometric Analysis 스킵)
                logger.info("[Female Detected] Analyzing hair length only (skipping geometric analysis)...")
                geo_analyzer = GeometricAnalyzer()
                final_classification = geo_analyzer.analyze_hair_length(face, landmarks, self.hair_only_mask)

                # 더미 데이터 (시각화용)
                analysis_data['classification'] = {'scores': {final_classification: 1.0}}
                analysis_data['forehead'] = {'mask': None, 'coverage': 0.0}
                analysis_data['left_right'] = {'left_ratio': 0.5, 'right_ratio': 0.5, 'symmetry': 0.0}

                viz_image = self.visualize_results(img, analysis_data, final_classification)

            else:
                # 남성: Bangs 검출 → Geometric Analysis
                logger.info("[Male Detected] Performing bangs detection and geometric analysis...")
                has_bangs = self._detect_bangs_with_bisenet(face, landmarks)

                if has_bangs:
                    # Bangs 경우
                    final_classification = "Bangs"
                    analysis_data['classification'] = {'scores': {'Bangs': 1.0}}
                    analysis_data['forehead'] = {
                        'mask': None,
                        'coverage': 0.0
                    }
                    analysis_data['left_right'] = {
                        'left_ratio': 0.5,
                        'right_ratio': 0.5,
                        'symmetry': 0.0
                    }

                    viz_image = self.visualize_results(img, analysis_data, final_classification)
                else:
                    # No Bangs - Geometric Analysis
                    logger.info("[Stage 2] No bangs detected, proceeding to geometric analysis...")
                    geo_analyzer = GeometricAnalyzer()
                    final_classification, geo_data = geo_analyzer.analyze(
                        img_gray, face, landmarks, person_silhouette, self.hair_only_mask, self.eyebrow_mask
                    )
                    analysis_data.update(geo_data)

                    background_mask = cv2.bitwise_not(person_silhouette)
                    analysis_data['background_mask'] = background_mask

                    viz_image = self.visualize_results(img, analysis_data, final_classification)

            result = {
                'classification': final_classification,
                'data': analysis_data,
                'clip_results': {
                    'bangs': 'No Bangs' if clip_additional['gender'] == 'Female' else ('Bangs' if has_bangs else 'No Bangs'),
                    'bangs_confidence': 1.0,
                    'glasses': clip_additional['glasses'],
                    'glasses_confidence': clip_additional['glasses_confidence'],
                    'beard': clip_additional['beard'],
                    'beard_confidence': clip_additional['beard_confidence'],
                    'gender': clip_additional['gender'],
                    'gender_confidence': clip_additional['gender_confidence']
                },
                # 별도 키로 추가 (test_all_samples.py 호환성)
                'gender_analysis': {
                    'gender': clip_additional['gender'],
                    'confidence': clip_additional['gender_confidence']
                },
                'glasses_analysis': {
                    'has_glasses': clip_additional['glasses'] != 'No Glasses',
                    'confidence': clip_additional['glasses_confidence']
                },
                'beard_analysis': {
                    'has_beard': clip_additional['beard'] != 'No Beard',
                    'confidence': clip_additional['beard_confidence']
                }
            }

            return result, viz_image

        except Exception as e:
            logger.error(f"Analysis Error: {e}", exc_info=True)
            return {'error': str(e)}, None

    def analyze(self, bgr_image: np.ndarray):
        """
        BGR numpy array를 직접 분석 (TCP 서버용)

        Args:
            bgr_image: BGR numpy array (1280x800x3)

        Returns:
            분석 결과 딕셔너리 (analyze_image와 동일 형식, 단 viz_image 제외)
        """
        try:
            logger.info("Analyzing BGR image (TCP server mode)")

            img = bgr_image.copy()
            if img is None or img.size == 0:
                raise ValueError("Invalid BGR image")

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # BGR → RGB → PIL
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # MediaPipe 얼굴 검출 (dlib 대체)
            faces = self.face_detector(img_gray, 1)
            if not faces:
                raise ValueError("No face detected by MediaPipe.")
            face = max(faces, key=lambda rect: rect.width() * rect.height())

            # MediaPipe 68점 랜드마크 추출 (dlib 대체)
            landmarks = self.shape_predictor(img_gray, face)
            if landmarks is None:
                raise ValueError("Failed to extract facial landmarks.")

            # Stage 1: BiSeNet 세그멘테이션
            logger.info("[Stage 1] Running BiSeNet segmentation...")
            person_silhouette = self.run_segmentation(pil_img)

            # CLIP으로 성별, 안경, 수염 검출 (성별 우선 판정)
            clip_additional = self.clip_classifier.classify_accessories_only(pil_img, face, landmarks)

            analysis_data = {}
            analysis_data['face_rect'] = face
            analysis_data['landmarks'] = landmarks
            analysis_data['clean_silhouette'] = person_silhouette
            analysis_data['hair_mask'] = self.hair_only_mask

            final_classification = "Unknown"

            # 성별에 따라 분석 경로 분기
            if clip_additional['gender'] == 'Female':
                # 여성: 헤어 길이만 분석
                logger.info("[Female Detected] Analyzing hair length only...")
                geo_analyzer = GeometricAnalyzer()
                final_classification = geo_analyzer.analyze_hair_length(face, landmarks, self.hair_only_mask)
            else:
                # 남성: Bangs 검출 → Geometric Analysis
                logger.info("[Male Detected] Performing bangs detection and geometric analysis...")
                has_bangs = self._detect_bangs_with_bisenet(face, landmarks)

                if has_bangs:
                    final_classification = "Bangs"
                else:
                    # No Bangs - Geometric Analysis
                    logger.info("[Stage 2] No bangs detected, proceeding to geometric analysis...")
                    geo_analyzer = GeometricAnalyzer()
                    final_classification, geo_data = geo_analyzer.analyze(
                        img_gray, face, landmarks, person_silhouette, self.hair_only_mask, self.eyebrow_mask
                    )
                    analysis_data.update(geo_data)

            result = {
                'classification': final_classification,
                'data': analysis_data,
                'clip_results': {
                    'bangs': 'No Bangs' if clip_additional['gender'] == 'Female' else ('Bangs' if has_bangs else 'No Bangs'),
                    'bangs_confidence': 1.0,
                    'glasses': clip_additional['glasses'],
                    'glasses_confidence': clip_additional['glasses_confidence'],
                    'beard': clip_additional['beard'],
                    'beard_confidence': clip_additional['beard_confidence'],
                    'gender': clip_additional['gender'],
                    'gender_confidence': clip_additional['gender_confidence']
                },
                'gender_analysis': {
                    'gender': clip_additional['gender'],
                    'confidence': clip_additional['gender_confidence']
                },
                'glasses_analysis': {
                    'has_glasses': clip_additional['glasses'] != 'No Glasses',
                    'confidence': clip_additional['glasses_confidence']
                },
                'beard_analysis': {
                    'has_beard': clip_additional['beard'] != 'No Beard',
                    'confidence': clip_additional['beard_confidence']
                }
            }

            return result

        except Exception as e:
            logger.error(f"BGR Analysis Error: {e}", exc_info=True)
            return {'error': str(e)}

    def visualize_results(self, image, analysis_data, final_classification):
        """
        상세한 6분할 시각화 패널을 생성하는 완전한 함수입니다.
        """
        logger.debug("Generating full 6-panel visualization...")
        h, w, _ = image.shape
        cell_h, cell_w = max(400, h), max(400, w)
        
        # 모든 경우에 6분할 패널 생성
        canvas = np.zeros((cell_h * 2, cell_w * 3, 3), dtype=np.uint8)

        def resize_to_cell(img, is_color=False):
            if img is None or img.size == 0: 
                return np.zeros((cell_h, cell_w, 3) if is_color else (cell_h, cell_w), dtype=np.uint8)
            r = min(cell_w / img.shape[1], cell_h / img.shape[0])
            dim = (int(img.shape[1] * r), int(img.shape[0] * r))
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            paste_board = np.zeros((cell_h, cell_w, 3) if is_color else (cell_h, cell_w), dtype=np.uint8)
            y_offset, x_offset = (cell_h - resized.shape[0]) // 2, (cell_w - resized.shape[1]) // 2
            if is_color and len(resized.shape) == 2: 
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            paste_board[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized
            return paste_board

        # 데이터 미리 추출
        face_rect = analysis_data.get('face_rect')
        landmarks = analysis_data.get('landmarks')
        clean_silhouette = analysis_data.get('clean_silhouette')
        forehead_data = analysis_data.get('forehead', {})
        forehead_mask = forehead_data.get('mask')  # ← 여기서 먼저 정의!
        left_right_data = analysis_data.get('left_right', {})

        # Panel 1: Original Image
        img1 = resize_to_cell(image, is_color=True)
        cv2.putText(img1, "1. Original Image", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        canvas[0:cell_h, 0:cell_w] = img1

        # Panel 2: BiSeNet Hair Detection
        img2 = resize_to_cell(image, is_color=True)
        
        if hasattr(self, 'hair_only_mask') and self.hair_only_mask is not None:
            hair_mask_resized = resize_to_cell(self.hair_only_mask)
            overlay = np.zeros_like(img2)
            overlay[hair_mask_resized > 0] = [255, 150, 0]
            img2 = cv2.addWeighted(img2, 0.6, overlay, 0.4, 0)
        
        if clean_silhouette is not None:
            silhouette_resized = resize_to_cell(clean_silhouette)
            overlay2 = np.zeros_like(img2)
            overlay2[silhouette_resized > 0] = [0, 100, 0]
            img2 = cv2.addWeighted(img2, 1.0, overlay2, 0.3, 0)
        
        cv2.putText(img2, "2. BiSeNet Hair Detection", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 150, 0), 2)
        canvas[0:cell_h, cell_w:cell_w*2] = img2

        # Panel 3: Defined Forehead Area
        img3 = resize_to_cell(image, is_color=True)
        if landmarks and face_rect:
            x_min, x_max = face_rect.left(), face_rect.right()
            eyebrow_points = np.array([[p.x, p.y] for p in landmarks.parts()[17:27]])
            dlib_y_max = int(np.min(eyebrow_points[:, 1]))
            
            if hasattr(self, 'eyebrow_mask') and self.eyebrow_mask is not None:
                search_top = max(0, dlib_y_max - int(face_rect.height() * 0.2))
                search_bottom = min(self.eyebrow_mask.shape[0], dlib_y_max + int(face_rect.height() * 0.1))
                eyebrow_region = self.eyebrow_mask[search_top:search_bottom, x_min:x_max]
                eyebrow_y_coords = np.where(eyebrow_region > 0)[0]
                
                if len(eyebrow_y_coords) > 0:
                    eyebrow_top_in_region = np.min(eyebrow_y_coords)
                    y_max = search_top + eyebrow_top_in_region
                else:
                    y_max = dlib_y_max
            else:
                y_max = dlib_y_max
            
            forehead_top = max(0, y_max - int((face_rect.height() * self.config.bangs_detection.forehead_height_ratio)))
            
            r = min(cell_w / w, cell_h / h)
            x_offset, y_offset = (cell_w - int(w * r)) // 2, (cell_h - int(h * r)) // 2
            
            x_start_s = int(face_rect.left() * r) + x_offset
            x_end_s = int(face_rect.right() * r) + x_offset
            y_start_s = int(forehead_top * r) + y_offset
            y_end_s = int(y_max * r) + y_offset

            overlay = np.zeros_like(img3)
            cv2.rectangle(overlay, (x_start_s, y_start_s), (x_end_s, y_end_s), (255, 255, 0), -1)
            img3 = cv2.addWeighted(img3, 0.6, overlay, 0.4, 0)
        cv2.putText(img3, "3. Defined Forehead Area", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        canvas[0:cell_h, cell_w*2:cell_w*3] = img3

        # Panel 4, 5, 6
        analysis_region_color = image.copy()
        if face_rect and landmarks:
            x_min, x_max = face_rect.left(), face_rect.right()
            eyebrow_points = np.array([[p.x, p.y] for p in landmarks.parts()[17:27]])
            dlib_y_max = int(np.min(eyebrow_points[:, 1]))
            
            if hasattr(self, 'eyebrow_mask') and self.eyebrow_mask is not None:
                search_top = max(0, dlib_y_max - int(face_rect.height() * 0.2))
                search_bottom = min(self.eyebrow_mask.shape[0], dlib_y_max + int(face_rect.height() * 0.1))
                eyebrow_region = self.eyebrow_mask[search_top:search_bottom, x_min:x_max]
                eyebrow_y_coords = np.where(eyebrow_region > 0)[0]
                
                if len(eyebrow_y_coords) > 0:
                    eyebrow_top_in_region = np.min(eyebrow_y_coords)
                    y_max = search_top + eyebrow_top_in_region
                else:
                    y_max = dlib_y_max
            else:
                y_max = dlib_y_max
            
            forehead_top = max(0, y_max - int((face_rect.height() * self.config.bangs_detection.forehead_height_ratio)))
            analysis_region_color = image[forehead_top:y_max, x_min:x_max]

        # Panel 4: Forehead with Hair overlay
        img4_bg = resize_to_cell(analysis_region_color, is_color=True)
        
        if final_classification != "Bangs" and forehead_mask is not None:
            # No Bangs - forehead_mask가 있을 때만
            if hasattr(self, 'hair_only_mask') and face_rect and landmarks:
                eyebrow_points = np.array([[p.x, p.y] for p in landmarks.parts()[17:27]])
                y_max = int(np.min(eyebrow_points[:, 1]))
                forehead_top = max(0, y_max - int((face_rect.height() * self.config.bangs_detection.forehead_height_ratio)))
                x_min, x_max = face_rect.left(), face_rect.right()
                
                forehead_hair = self.hair_only_mask[forehead_top:y_max, x_min:x_max]
                forehead_person = clean_silhouette[forehead_top:y_max, x_min:x_max] if clean_silhouette is not None else None
                
                hair_resized = resize_to_cell(forehead_hair)
                overlay = np.zeros_like(img4_bg)
                overlay[hair_resized > 0] = [255, 150, 0]
                img4 = cv2.addWeighted(img4_bg, 0.6, overlay, 0.4, 0)
                
                if forehead_person is not None:
                    person_resized = resize_to_cell(forehead_person)
                    overlay2 = np.zeros_like(img4)
                    overlay2[person_resized > 0] = [0, 100, 0]
                    img4 = cv2.addWeighted(img4, 1.0, overlay2, 0.3, 0)
            else:
                img4 = img4_bg
        else:
            # Bangs이거나 forehead_mask가 None
            img4 = img4_bg
        
        title = "4. Hair in Forehead (BiSeNet)" if final_classification != "Bangs" else "4. Forehead Hair Coverage"
        cv2.putText(img4, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        if final_classification != "Bangs" and forehead_mask is not None:
            skin_coverage = forehead_data.get('coverage', 0)
            cv2.putText(img4, f"Skin: {skin_coverage:.1%}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(img4, "Bangs Detected", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
        
        canvas[cell_h:cell_h*2, 0:cell_w] = img4
        
        # Panel 5: Left/Right or Bangs info
        img5_bg = resize_to_cell(analysis_region_color, is_color=True)
        
        if final_classification == "Bangs":
            cv2.putText(img5_bg, "5. Bangs Analysis", (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img5_bg, "Full bangs detected", (15, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            img5 = img5_bg
        else:
            # No Bangs: Left/Right Analysis
            if forehead_mask is not None:
                mask_vis = resize_to_cell(forehead_mask)
                
                # 리사이즈된 마스크에서 다시 중앙값 계산
                skin_coords_vis = np.where(mask_vis > 0)
                if len(skin_coords_vis[1]) > 0:
                    center_x = int(np.median(skin_coords_vis[1]))
                else:
                    center_x = mask_vis.shape[1] // 2
                
                overlay = np.zeros_like(img5_bg)
                mask_bool = mask_vis > 0
                overlay[:, 0:center_x][mask_bool[:, 0:center_x]] = [255, 0, 0]
                overlay[:, center_x:][mask_bool[:, center_x:]] = [0, 0, 255]
                img5 = cv2.addWeighted(img5_bg, 0.6, overlay, 0.4, 0)
                cv2.line(img5, (center_x, 0), (center_x, cell_h), (0, 255, 255), 2)
            else:
                img5 = img5_bg
            cv2.putText(img5, "5. Left/Right Analysis", (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(img5, f"L:{left_right_data.get('left_ratio', 0):.1%} R:{left_right_data.get('right_ratio', 0):.1%}", 
                        (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        canvas[cell_h:cell_h*2, cell_w:cell_w*2] = img5

        # Panel 6: Classification
        img6 = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        cv2.putText(img6, "6. Classification", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img6, f"Final: {final_classification}", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y_pos = 120

        # 여성이면 헤어 길이만 표시, 남성이면 기존 스코어 표시
        if final_classification in ["Short Hair", "Long Hair"]:
            # 여성: 헤어 길이만 표시
            color = (0, 255, 0)
            text = f"{final_classification}: 100.0%"
            cv2.putText(img6, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            bar_width = cell_w - 40
            cv2.rectangle(img6, (20, y_pos + 15), (20 + bar_width, y_pos + 40), color, -1)
        else:
            # 남성: 기존 스코어 표시
            scores = analysis_data.get('classification', {}).get('scores', {})
            for style, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                color = (0, 255, 0) if style == final_classification else (200, 200, 200)
                text = f"{style}: {score:.1%}"
                cv2.putText(img6, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                bar_width = int(score * (cell_w - 40))
                if bar_width > 0:
                    cv2.rectangle(img6, (20, y_pos + 15), (20 + bar_width, y_pos + 40), color, -1)
                y_pos += 70
        canvas[cell_h:cell_h*2, cell_w*2:cell_w*3] = img6
        
        return canvas