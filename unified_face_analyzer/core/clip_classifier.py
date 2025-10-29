"""
Stage 1: Bangs, Glasses, and Beard classification using OpenAI's CLIP model.
Uses the full face image for all detections in a single pass.
"""
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from utils import get_config, get_logger
from pathlib import Path

logger = get_logger(__name__)

# 로컬 모델 저장 경로 (프로젝트 내부)
MODELS_DIR = Path(__file__).parent.parent / 'data' / 'models' / 'clip-vit-base-patch32'

class CLIPClassifier:
    def __init__(self, device):
        logger.info("Loading CLIP model (this might take a moment on first run)...")
        self.device = device
        self.config = get_config()
        model_name = "openai/clip-vit-base-patch32"

        # 로컬 모델이 있으면 로컬에서 로드, 없으면 다운로드 후 저장
        if MODELS_DIR.exists() and (MODELS_DIR / "config.json").exists():
            logger.info(f"✅ 로컬 CLIP 모델 로드: {MODELS_DIR}")
            self.model = CLIPModel.from_pretrained(MODELS_DIR).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(MODELS_DIR)
        else:
            logger.info("📥 CLIP 모델 다운로드 중... (최초 1회만, 약 600MB)")
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)

            # 다운로드한 모델을 로컬에 저장
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(MODELS_DIR)
            self.processor.save_pretrained(MODELS_DIR)
            logger.info(f"✅ 모델 저장 완료: {MODELS_DIR}")

        self.results = {}

    def classify(self, pil_img, face, landmarks):
        """
        Analyzes the full face image using CLIP for bangs, glasses, and beard detection.
        Returns 'Bangs' or 'No Bangs'.
        """
        logger.info("[Stage 1] Analyzing with CLIP (Full Face Image)...")
        
        # 얼굴 전체 영역 사용 - 모든 판단에 동일한 이미지 사용
        face_image = pil_img.crop((face.left(), face.top(), face.right(), face.bottom()))

        # 모든 특징을 한 번에 판단할 프롬프트
        text_prompts = [
            # Bangs (앞머리) - 더 구체적으로
            "a person with full bangs completely covering the forehead",
            "a person with no bangs and completely visible forehead",
            # Glasses (안경)
            "a person wearing glasses",
            "a person not wearing glasses",
            # Beard (수염)
            "a person with beard or facial hair",
            "a person without beard, clean shaven"
        ]

        inputs = self.processor(text=text_prompts, images=face_image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

        self.results = {prompt: prob for prompt, prob in zip(text_prompts, probs)}

        logger.debug("--- CLIP Analysis Results (All Features) ---")
        for prompt, prob in self.results.items():
            logger.debug(f"  '{prompt}': {prob:.2%}")
        
        # 각 카테고리별로 결과 추출
        bangs_probs = [probs[0], probs[1]]  # 앞머리 있음, 없음
        glasses_probs = [probs[2], probs[3]]  # 안경 있음, 없음
        beard_probs = [probs[4], probs[5]]  # 수염 있음, 없음
        
        # 판단
        bangs_result = "Bangs" if bangs_probs[0] > bangs_probs[1] else "No Bangs"
        glasses_result = "Wearing Glasses" if glasses_probs[0] > glasses_probs[1] else "No Glasses"
        beard_result = "With Beard" if beard_probs[0] > beard_probs[1] else "No Beard"
        
        logger.info("=" * 60)
        logger.info("FINAL CLIP RESULTS:")
        logger.info(f"  Bangs: {bangs_result} (confidence: {max(bangs_probs):.2%})")
        logger.info(f"  Glasses: {glasses_result} (confidence: {max(glasses_probs):.2%})")
        logger.info(f"  Beard: {beard_result} (confidence: {max(beard_probs):.2%})")
        logger.info("=" * 60)

        if bangs_result == "Bangs":
            logger.info("CLIP Conclusion: Bangs DETECTED.")
        else:
            logger.info("CLIP Conclusion: No Bangs detected. Proceeding to Stage 2.")
        
        return bangs_result

    def get_additional_results(self):
        """안경과 수염 판단 결과를 반환합니다."""
        if not self.results:
            return {
                'glasses': 'Unknown',
                'glasses_confidence': 0.0,
                'beard': 'Unknown',
                'beard_confidence': 0.0
            }
        
        # 프롬프트에서 확률값 추출
        prompts_list = list(self.results.keys())
        probs_list = list(self.results.values())
        
        glasses_probs = [probs_list[2], probs_list[3]]  # 안경 있음, 없음
        beard_probs = [probs_list[4], probs_list[5]]  # 수염 있음, 없음
        
        glasses_status = "Wearing Glasses" if glasses_probs[0] > glasses_probs[1] else "No Glasses"
        beard_status = "With Beard" if beard_probs[0] > beard_probs[1] else "No Beard"
        
        return {
            'glasses': glasses_status,
            'glasses_confidence': max(glasses_probs),
            'beard': beard_status,
            'beard_confidence': max(beard_probs)
        }
    
    def classify_accessories_only(self, pil_img, face, landmarks):
        """
        안경, 수염, 성별 판단 (앞머리는 BiSeNet이 담당)
        """
        logger.info("[CLIP] Analyzing accessories and gender (glasses, beard, gender)...")

        face_image = pil_img.crop((face.left(), face.top(), face.right(), face.bottom()))

        text_prompts = [
            "a person wearing glasses",
            "a person not wearing glasses",
            "a person with beard or facial hair",
            "a person without beard, clean shaven",
            "a male person",
            "a female person"
        ]

        inputs = self.processor(text=text_prompts, images=face_image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

        # Threshold 기반 판정 (confidence가 너무 낮으면 신뢰하지 않음)
        glasses_confidence = max(probs[0], probs[1])
        glasses_threshold = self.config.clip.glasses_confidence_threshold
        if glasses_confidence < glasses_threshold:
            glasses_result = "No Glasses"  # 기본값
        else:
            glasses_result = "Wearing Glasses" if probs[0] > probs[1] else "No Glasses"

        # 수염 판정: 최소 threshold 이상 confidence 필요
        beard_confidence = max(probs[2], probs[3])
        beard_threshold = self.config.clip.beard_confidence_threshold
        if beard_confidence < beard_threshold:
            beard_result = "No Beard"  # 확신이 낮으면 기본값
        else:
            beard_result = "With Beard" if probs[2] > probs[3] else "No Beard"

        # 성별 판정: 상대 비교 (NIR 이미지에서는 confidence가 낮을 수 있음)
        gender_result = "Male" if probs[4] > probs[5] else "Female"

        logger.info(f"  Glasses: {glasses_result} ({max(probs[0], probs[1]):.2%})")
        logger.info(f"  Beard: {beard_result} ({max(probs[2], probs[3]):.2%})")
        logger.info(f"  Gender: {gender_result} ({max(probs[4], probs[5]):.2%})")

        return {
            'glasses': glasses_result,
            'glasses_confidence': max(probs[0], probs[1]),
            'beard': beard_result,
            'beard_confidence': max(probs[2], probs[3]),
            'gender': gender_result,
            'gender_confidence': max(probs[4], probs[5])
        }