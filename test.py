# test.py

import os
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch

from preprocess import build_dataloader
from model import load_clip_matcher, HybridYOLOCLIP


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def iou_xywh_pixel(pred_xywh, gt_xywh) -> float:
    """
    pred_xywh, gt_xywh: (x, y, w, h) in pixel
    """
    px, py, pw, ph = pred_xywh
    gx, gy, gw, gh = gt_xywh

    px2, py2 = px + pw, py + ph
    gx2, gy2 = gx + gw, gy + gh

    ix1, iy1 = max(px, gx), max(py, gy)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    up = pw * ph + gw * gh - inter
    if up <= 0:
        up = 1e-6
    return float(inter / up)


def run_eval(args):
    """
    GT bbox 가 있는 JSON 에 대해:
    - YOLO + CLIP 로 bbox 예측
    - CSV 저장
    - mIoU 계산
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = load_clip_matcher(args.ckpt, clip_name=args.clip_name, device=device)
    hybrid = HybridYOLOCLIP(
        yolo_ckpt=args.yolo_ckpt,
        clip_matcher=matcher,
        device=device,
    )

    ds, dl = build_dataloader(
        json_dir=args.json_dir,
        jpg_root=args.jpg_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    rows = []
    ious = []
    total = len(ds)
    processed = 0

    for batch_idx, batch in enumerate(dl):
        # batch 는 sample dict 의 리스트
        for sample in batch:
            img_path = sample["img_path"]
            qtxt = sample["query_text"]
            qid = sample["query_id"]
            gt_bbox = sample["bbox"]  # [x, y, w, h] or None

            pred_x, pred_y, pred_w, pred_h = hybrid.predict_single(
                image_path=img_path,
                query_text=qtxt,
                conf_thres=args.conf_thres,
            )

            rows.append(
                {
                    "query_id": qid,
                    "query_text": qtxt,
                    "pred_x": pred_x,
                    "pred_y": pred_y,
                    "pred_w": pred_w,
                    "pred_h": pred_h,
                }
            )
            iou = None
            if gt_bbox is not None:
                gx, gy, gw, gh = [float(v) for v in gt_bbox]
                iou = iou_xywh_pixel(
                    [pred_x, pred_y, pred_w, pred_h],
                    [gx, gy, gw, gh],
                )
                ious.append(iou)
            processed += 1
            print(
                f"[Eval] {processed}/{total} | "
                f"qid={qid} | "
                f"pred=({pred_x:.1f},{pred_y:.1f},{pred_w:.1f},{pred_h:.1f})"
                + (f" | IoU={iou:.3f}" if iou is not None else "")
            )
    df = pd.DataFrame(
        rows, columns=["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"]
    )
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[Eval] Saved CSV to {args.out_csv}")

    if ious:
        print(f"[Eval] mIoU = {float(np.mean(ious)):.4f} (N={len(ious)})")
    else:
        print("[Eval] No GT bbox found; mIoU not computed.")


def run_predict(args):
    """
    테스트셋 JSON 에 대해:
    - YOLO + CLIP 로 bbox 예측
    - CSV 저장 (Dacon 제출용)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = load_clip_matcher(args.ckpt, clip_name=args.clip_name, device=device)
    hybrid = HybridYOLOCLIP(
        yolo_ckpt=args.yolo_ckpt,
        clip_matcher=matcher,
        device=device,
    )

    ds, dl = build_dataloader(
        json_dir=args.json_dir,
        jpg_root=args.jpg_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    rows = []

    for batch in dl:
        for sample in batch:
            img_path = sample["img_path"]
            qtxt = sample["query_text"]
            qid = sample["query_id"]

            pred_x, pred_y, pred_w, pred_h = hybrid.predict_single(
                image_path=img_path,
                query_text=qtxt,
                conf_thres=args.conf_thres,
            )

            rows.append(
                {
                    "query_id": qid,
                    "query_text": qtxt,
                    "pred_x": pred_x,
                    "pred_y": pred_y,
                    "pred_w": pred_w,
                    "pred_h": pred_h,
                }
            )

    df = pd.DataFrame(
        rows, columns=["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"]
    )
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[Predict] Saved CSV to {args.out_csv}")


def get_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # eval
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--json_dir", type=str, required=True)
    p_eval.add_argument("--jpg_dir", type=str, required=True)
    p_eval.add_argument("--ckpt", type=str, required=True)
    p_eval.add_argument("--yolo_ckpt", type=str, required=True)
    p_eval.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32")
    p_eval.add_argument("--out_csv", type=str, default="./outputs/eval_pred.csv")
    p_eval.add_argument("--batch_size", type=int, default=1)
    p_eval.add_argument("--num_workers", type=int, default=0)
    p_eval.add_argument("--conf_thres", type=float, default=0.25)

    # predict
    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--json_dir", type=str, required=True)
    p_pred.add_argument("--jpg_dir", type=str, required=True)
    p_pred.add_argument("--ckpt", type=str, required=True)
    p_pred.add_argument("--yolo_ckpt", type=str, required=True)
    p_pred.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32")
    p_pred.add_argument("--out_csv", type=str, default="./outputs/test_pred.csv")
    p_pred.add_argument("--batch_size", type=int, default=1)
    p_pred.add_argument("--num_workers", type=int, default=0)
    p_pred.add_argument("--conf_thres", type=float, default=0.25)

    return ap.parse_args()


def main():
    seed_everything(42)
    args = get_args()
    if args.cmd == "eval":
        run_eval(args)
    elif args.cmd == "predict":
        run_predict(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
