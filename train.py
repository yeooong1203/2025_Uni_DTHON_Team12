# train.py

import os
import argparse
import random
from typing import List, Tuple

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
    x, y, w, h = bbox
    W, H = pil_img.size
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(W, int(x + w))
    y2 = min(H, int(y + h))
    if x2 <= x1 or y2 <= y1:
        # 이상한 bbox면 전체 이미지 반환 (fallback)
        return pil_img.copy()
    return pil_img.crop((x1, y1, x2, y2))

def crop_negative(pil_img: Image.Image, pos_bbox: List[float], min_size=32) -> Image.Image:
    """
    이미지 전체에서 random crop을 뽑고,
    positive bbox와 IoU < 0.2 인 crop을 negative 로 사용.
    실패해도 fallback 있어 에러 없음.
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
        return inter / union if union != 0 else 0.0

    for _ in range(30):
        rw = random.randint(min_size, max(min_size + 1, int(W * 0.5)))
        rh = random.randint(min_size, max(min_size + 1, int(H * 0.5)))

        if W - rw <= 1 or H - rh <= 1:
            continue

        rx = random.randint(0, W - rw)
        ry = random.randint(0, H - rh)

        cand = (rx, ry, rx + rw, ry + rh)
        if iou((px1, py1, px2, py2), cand) < 0.2:
            return pil_img.crop(cand)

    # fallback
    return pil_img.crop((0, 0, min_size, min_size))

print("start training clip matcher...")

def train_clip_matcher(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds, dl = build_dataloader(
        json_dir=args.json_dir,
        jpg_root=args.jpg_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    # ===== NEW: Limit dataset samples =====
    if args.max_samples is not None:
        old_len = len(ds.samples)
        ds.samples = ds.samples[:args.max_samples]
        print(f"[INFO] Using only {len(ds.samples)} samples out of {old_len}.")
    # ======================================


    matcher = ClipMatcher(clip_name=args.clip_name).to(device)
    optimizer = torch.optim.AdamW(matcher.parameters(), lr=args.lr, weight_decay=1e-4)

    print("Entering epoch loop now...")

    for epoch in range(1, args.epochs + 1):
        print(f"[DEBUG] === Epoch {epoch} start ===")
        matcher.train()
        running_loss = 0.0
        n_samples = 0

        for step, batch in enumerate(dl):
            print(f"[DEBUG] Epoch {epoch} | Batch {step} START, batch size = {len(batch)}")

            pos_images = []
            neg_images = []
            texts = []

            for sample in batch:
                img_path = sample["img_path"]
                bbox = sample["bbox"]
                qtxt = sample["query_text"]

                if bbox is None:
                    continue

                pil_img = Image.open(img_path).convert("RGB")
                pos_img = crop_positive(pil_img, bbox)
                neg_img = crop_negative(pil_img, bbox)

                pos_images.append(pos_img)
                neg_images.append(neg_img)
                texts.append(qtxt)

            if not texts:
                print(f"[DEBUG] Epoch {epoch} | Batch {step} SKIP (no GT)")
                continue

            print(f"[DEBUG] Epoch {epoch} | Batch {step} BEFORE loss, n={len(texts)}")

            loss = matcher.compute_margin_loss(
                texts=texts,
                pos_images=pos_images,
                neg_images=neg_images,
                device=device,
                margin=0.2,
            )

            print(f"[DEBUG] Epoch {epoch} | Batch {step} AFTER loss, loss={loss.item():.4f}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * len(texts)
            n_samples += len(texts)

        avg_loss = running_loss / max(1, n_samples)
        print(f"[Epoch {epoch}/{args.epochs}] avg_loss={avg_loss:.4f}")

    # 학습 완료 후 모델 저장
    save_clip_matcher(args.ckpt, matcher)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", type=str, required=True, help="학습용 JSON 디렉토리")
    ap.add_argument("--jpg_dir", type=str, required=True, help="이미지(jpg) 디렉토리")
    ap.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--ckpt", type=str, default="./outputs/clip_matcher.pth")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_samples", type=int, default=None)

    return ap.parse_args()


def main():
    seed_everything(42)
    args = get_args()
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    train_clip_matcher(args)


if __name__ == "__main__":
    main()
