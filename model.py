# model.py

import os
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO


class ClipMatcher(nn.Module):
    """
    CLIP 기반 매칭 모델.
    - CLIPModel 을 그대로 사용하고, logits_per_image 를 가지고
      positive > negative 가 되도록 margin ranking loss 로 학습.
    """

    def __init__(self, clip_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(clip_name)
        self.processor = CLIPProcessor.from_pretrained(clip_name)

    def forward_scores(
        self, texts: List[str], pil_images: List[Image.Image], device: torch.device
    ) -> torch.Tensor:
        """
        주어진 texts, images 에 대한 logits_per_image 의 대각 성분(diagonal)을 반환.
        (각 i-th text 와 i-th image 의 유사도 점수)
        """
        inputs = self.processor(
            text=texts, images=pil_images, return_tensors="pt", padding="max_length", truncation = True, max_length=77
        ).to(device)
        outputs = self.clip_model(**inputs)
        # logits_per_image: (batch_size, batch_size)
        logits = outputs.logits_per_image  # image-text similarity
        # 대각 원소만 사용
        diag = torch.diagonal(logits, 0)
        return diag  # (batch,)

    def compute_margin_loss(
        self,
        texts: List[str],
        pos_images: List[Image.Image],
        neg_images: List[Image.Image],
        device: torch.device,
        margin: float = 0.2,
    ) -> torch.Tensor:
        """
        positive 이미지와 negative 이미지에 대해 margin ranking loss 계산.
        pos_score > neg_score 되도록 학습.
        """
        self.train()
        pos_scores = self.forward_scores(texts, pos_images, device)  # (B,)
        neg_scores = self.forward_scores(texts, neg_images, device)  # (B,)
        target = torch.ones_like(pos_scores, device=device)
        loss = F.margin_ranking_loss(pos_scores, neg_scores, target, margin=margin)
        return loss

    @torch.no_grad()
    def score_single(
        self, text: str, pil_image: Image.Image, device: torch.device
    ) -> float:
        """
        단일 (text, image) 쌍에 대한 CLIP similarity 점수 반환.
        """
        self.eval()
        inputs = self.processor(
            text=[text], images=[pil_image], return_tensors="pt", padding="max_length", truncation=True, max_length=77
        ).to(device)
        outputs = self.clip_model(**inputs)
        logits = outputs.logits_per_image  # (1,1)
        return float(logits[0, 0].cpu().item())


class HybridYOLOCLIP:
    """
    추론(Inference)용 하이브리드 모델:
    - YOLO 를 사용해 문서 이미지에서 후보 bbox 들을 검출.
    - ClipMatcher 를 사용해 질의와 각 bbox crop 의 similarity 를 측정.
    - 가장 높은 score 를 가진 bbox 를 최종 출력.
    """

    def __init__(
        self,
        yolo_ckpt: str,
        clip_matcher: ClipMatcher,
        device: torch.device,
    ):
        self.device = device
        self.detector = YOLO(yolo_ckpt)
        # clip_matcher 는 학습된 state_dict 로 로드된 상태로 전달
        self.matcher = clip_matcher.to(device)

    def predict_single(
        self,
        image_path: str,
        query_text: str,
        conf_thres: float = 0.25,
    ) -> Tuple[float, float, float, float]:
        """
        단일 이미지 경로와 질의에 대해:
        - YOLO: 후보 bbox 들 검출
        - CLIP: 각 bbox crop 에 대해 score 계산
        - 최고 score bbox 의 (x, y, w, h) pixel 좌표 리턴
        """
        pil_img = Image.open(image_path).convert("RGB")
        W, H = pil_img.size

        # YOLO inference
        results = self.detector.predict(
            source=pil_img, conf=conf_thres, verbose=False
        )
        if not results:
            # YOLO 결과가 없으면 전체 이미지를 반환 (fallback)
            return 0.0, 0.0, float(W), float(H)

        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()  # (N,4) [x1,y1,x2,y2]
        if boxes_xyxy.shape[0] == 0:
            return 0.0, 0.0, float(W), float(H)

        best_score = -1e9
        best_bbox = (0.0, 0.0, float(W), float(H))

        for (x1, y1, x2, y2) in boxes_xyxy:
            x1i = max(0, int(x1))
            y1i = max(0, int(y1))
            x2i = min(W, int(x2))
            y2i = min(H, int(y2))
            if x2i <= x1i or y2i <= y1i:
                continue
            crop = pil_img.crop((x1i, y1i, x2i, y2i))
            score = self.matcher.score_single(query_text, crop, self.device)
            if score > best_score:
                best_score = score
                best_bbox = (float(x1i), float(y1i), float(x2i - x1i), float(y2i - y1i))

        return best_bbox


# ------------------
# Save / Load Utils
# ------------------


def save_clip_matcher(ckpt_path: str, matcher: ClipMatcher):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    state = matcher.state_dict()
    torch.save(state, ckpt_path)
    print(f"[save_clip_matcher] Saved to {ckpt_path}")


def load_clip_matcher(ckpt_path: str, clip_name: str, device: torch.device) -> ClipMatcher:
    matcher = ClipMatcher(clip_name=clip_name)
    state = torch.load(ckpt_path, map_location=device)
    matcher.load_state_dict(state)
    matcher.to(device)
    print(f"[load_clip_matcher] Loaded from {ckpt_path}")
    return matcher
