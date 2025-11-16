# train.py

import os
import argparse
import random
from typing import List

import numpy as np
import torch
from PIL import Image

from preprocess import build_dataloader
from model import ClipMatcher, save_clip_matcher


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def crop_positive(pil_img: Image.Image, bbox: List[float]) -> Image.Image:
    """GT bbox 영역을 positive crop으로 사용."""
    x, y, w, h = bbox
    W, H = pil_img.size
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(W, int(x + w))
    y2 = min(H, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return pil_img.copy()
    return pil_img.crop((x1, y1, x2, y2))


def crop_negative_random(pil_img: Image.Image, pos_bbox: List[float]) -> Image.Image:
    """
    예전 버전 random negative (fallback 용).
    pos_bbox와 IoU가 낮은 random crop을 시도.
    """
    W, H = pil_img.size
    px, py, pw, ph = pos_bbox
    px1, py1, px2, py2 = px, py, px + pw, py + ph

    def iou(box1, box2):
        x1, y1, x2, y2 = box1
        xb1, yb1, xb2, yb2 = box2
        ix1, iy1 = max(x1, xb1), max(y1, yb1)
        ix2, iy2 = min(x2, xb2), min(y2, yb2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area1 = max(0, x2 - x1) * max(0, y2 - y1)
        area2 = max(0, xb2 - xb1) * max(0, yb2 - yb1)
        union = area1 + area2 - inter
        if union <= 0:
            return 0.0
        return inter / union
    if pw <= 0 or ph <= 0:
        return pil_img.copy()

    for _ in range(50):
        w_low = max(10, int(pw / 2))
        w_high = min(W, int(pw * 2))
        h_low = max(10, int(ph / 2))
        h_high = min(H, int(ph * 2))

        # 🔴 여기서도 범위가 꼬이면 그냥 전체 이미지 반환
        if w_high <= w_low or h_high <= h_low:
            return pil_img.copy()

        rw = random.randint(w_low, w_high)
        rh = random.randint(h_low, h_high)
        rx = random.randint(0, max(0, W - rw))
        ry = random.randint(0, max(0, H - rh))
        cand = (rx, ry, rx + rw, ry + rh)
        
        if iou((px1, py1, px2, py2), cand) < 0.2:
            return pil_img.crop(cand)

    return pil_img.copy()


def crop_negative_from_other_ann(
    pil_img: Image.Image,
    other_bbox: List[float]
) -> Image.Image:
    """
    같은 이미지 내 '다른 annotation bbox'를 negative로 사용하는 함수.
    (hard negative: 같은 페이지의 다른 시각요소)
    """
    x, y, w, h = other_bbox
    W, H = pil_img.size
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(W, int(x + w))
    y2 = min(H, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return pil_img.copy()
    return pil_img.crop((x1, y1, x2, y2))


def train_clip_matcher(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device = {device}")

    # DataLoader 준비 (batch는 sample dict 리스트)
    ds, dl = build_dataloader(
        json_dir=args.json_dir,
        jpg_root=args.jpg_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    print(f"[Train] total samples in dataset: {len(ds)}")

    matcher = ClipMatcher(clip_name=args.clip_name).to(device)

    # 기존 ckpt 이어서 학습 (press → report 파인튜닝 등)
    if args.ckpt is not None and os.path.exists(args.ckpt):
        print(f"[Train] Loading checkpoint: {args.ckpt}")
        state = torch.load(args.ckpt, map_location=device)
        matcher.load_state_dict(state)
    else:
        print("[Train] No valid ckpt given or file not found; training from scratch")

    optimizer = torch.optim.AdamW(matcher.parameters(), lr=args.lr, weight_decay=1e-4)

    # 전체 학습 샘플 수 제한 (max_samples)
    max_samples = args.max_samples
    processed_global = 0

    for epoch in range(1, args.epochs + 1):
        matcher.train()
        running_loss = 0.0
        n_samples = 0

        for batch_idx, batch in enumerate(dl):
            # max_samples를 넘기면 학습 중단
            if max_samples is not None and processed_global >= max_samples:
                break

            # 1) 같은 이미지별로 인덱스 모으기
            img_to_indices = {}
            for idx, sample in enumerate(batch):
                img_path = sample["img_path"]
                bbox = sample["bbox"]
                if bbox is None:
                    continue
                img_to_indices.setdefault(img_path, []).append(idx)

            pos_images = []
            neg_images = []
            texts = []

            for idx, sample in enumerate(batch):
                img_path = sample["img_path"]
                bbox = sample["bbox"]
                qtxt = sample["query_text"]

                if bbox is None:
                    continue

                # max_samples를 넘기지 않도록 남은 개수 계산
                if max_samples is not None and processed_global + len(texts) >= max_samples:
                    break

                pil_img = Image.open(img_path).convert("RGB")

                # positive crop
                pos_img = crop_positive(pil_img, bbox)

                # 같은 이미지 내 다른 annotation을 hard negative로 사용
                cand_indices = img_to_indices.get(img_path, [])
                other_indices = [j for j in cand_indices if j != idx]

                if len(other_indices) > 0:
                    j = random.choice(other_indices)
                    other_bbox = batch[j]["bbox"]
                    neg_img = crop_negative_from_other_ann(pil_img, other_bbox)
                else:
                    # 같은 이미지의 다른 ann이 없으면 random negative fallback
                    neg_img = crop_negative_random(pil_img, bbox)

                pos_images.append(pos_img)
                neg_images.append(neg_img)
                texts.append(qtxt)

            if not texts:
                continue

            optimizer.zero_grad(set_to_none=True)
            loss = matcher.compute_margin_loss(
                texts=texts,
                pos_images=pos_images,
                neg_images=neg_images,
                device=device,
                margin=0.2,
            )
            loss.backward()
            optimizer.step()

            batch_size_eff = len(texts)
            running_loss += float(loss.item()) * batch_size_eff
            n_samples += batch_size_eff
            processed_global += batch_size_eff

            if batch_idx % 10 == 0:
                print(
                    f"[Epoch {epoch}/{args.epochs}] "
                    f"batch {batch_idx} | "
                    f"loss={loss.item():.4f} | "
                    f"processed={processed_global}"
                )

            if max_samples is not None and processed_global >= max_samples:
                break

        avg_loss = running_loss / max(1, n_samples)
        print(f"[Epoch {epoch}/{args.epochs}] avg_loss={avg_loss:.4f}, used_samples={n_samples}")

        if max_samples is not None and processed_global >= max_samples:
            print(f"[Train] Reached max_samples={max_samples}, stopping early after epoch {epoch}")
            break

    # 최종 저장
    os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)
    save_clip_matcher(args.save_ckpt, matcher)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", type=str, required=True, help="학습용 JSON 디렉토리")
    ap.add_argument("--jpg_dir", type=str, required=True, help="이미지(jpg) 디렉토리")
    ap.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--ckpt", type=str, default=None, help="이전 ckpt에서 이어서 학습할 때 사용")
    ap.add_argument("--save_ckpt", type=str, default="./outputs/clip_matcher.pth")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="학습에 사용할 최대 샘플 수 (None이면 전체)",
    )
    return ap.parse_args()


def main():
    seed_everything(42)
    args = get_args()
    train_clip_matcher(args)


if __name__ == "__main__":
    main()
