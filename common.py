"""Shared helpers: environment info, seeding, data loading, metrics, detectors, SAM2."""
import json
import os
import platform
import random
import sys
import time

import cv2
import numpy as np
import torch

SEED = 42
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT, "data_raw", "data"))
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)
HAND_IDS = (1, 2)  # 1=left hand, 2=right hand (verified in s1_data_checks.py); merged to one class
MIN_BOX_PIXELS = 50  # a hand instance with fewer pixels does not produce a GT box (still counted as hand pixels)
MAX_BOXES = 6
GDINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CKPT = os.path.join(ROOT, "weights", "sam2.1_hiera_small.pt")


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def print_env():
    print(f"python {platform.python_version()} | torch {torch.__version__} | CUDA (torch build) {torch.version.cuda} "
          f"| cudnn {torch.backends.cudnn.version()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} | total VRAM {p.total_memory / 1024**3:.2f} GiB")
    else:
        print("GPU: NOT AVAILABLE")
    sys.stdout.flush()


def load_json(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def save_json(obj, name):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2)


def read_rgb(path):
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def read_hand_mask(label_path):
    lab = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    if lab.ndim == 3:
        lab = lab[..., 0]
    return np.isin(lab, HAND_IDS), lab


def gt_boxes(lab):
    """One box per hand label id (left / right), class merged to 'hand'. xyxy, inclusive-exclusive."""
    boxes = []
    for hid in HAND_IDS:
        ys, xs = np.nonzero(lab == hid)
        if len(xs) >= MIN_BOX_PIXELS:
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def image_stats(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return dict(blur=float(cv2.Laplacian(g, cv2.CV_64F).var()),
                brightness=float(g.mean()), contrast=float(g.std()))


def iou_dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    s = pred.sum() + gt.sum()
    iou = inter / union if union else 1.0
    dice = 2 * inter / s if s else 1.0
    return float(iou), float(dice)


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def match_boxes(pred, scores, gt, thr=0.5):
    """Greedy (score order) matching. Returns tp flags per pred (in given order) and n_gt."""
    order = np.argsort(-scores)
    tp = np.zeros(len(pred), bool)
    used = np.zeros(len(gt), bool)
    ious = box_iou(pred, gt)
    for i in order:
        if len(gt) == 0:
            break
        cand = np.where(~used, ious[i], -1)
        j = int(np.argmax(cand))
        if cand[j] >= thr:
            tp[i] = True; used[j] = True
    return tp


def average_precision(all_scores, all_tp, n_gt):
    """VOC all-point AP."""
    if n_gt == 0:
        return float("nan")
    if len(all_scores) == 0:
        return 0.0
    o = np.argsort(-np.asarray(all_scores))
    tp = np.asarray(all_tp)[o].astype(float)
    ctp = np.cumsum(tp); cfp = np.cumsum(1 - tp)
    rec = ctp / n_gt; prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# ---------------------------------------------------------------- models
class GDino:
    """Grounding DINO tiny. Returns ALL boxes above low floor thresholds with both scores,
    so box/text thresholds can be applied afterwards without re-running the model."""

    def __init__(self, device="cuda", prompt="hand."):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.proc = AutoProcessor.from_pretrained(GDINO_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to(device).eval()
        self.device, self.prompt = device, prompt

    @torch.inference_mode()
    def __call__(self, rgbs, floor=0.05):
        from PIL import Image
        ims = [Image.fromarray(r) for r in rgbs]
        inp = self.proc(images=ims, text=[self.prompt] * len(ims), return_tensors="pt", padding=True).to(self.device)
        out = self.model(**inp)
        logits = out.logits.sigmoid()  # (B, Q, T)
        ids = inp["input_ids"]
        res = []
        for b, r in enumerate(rgbs):
            h, w = r.shape[:2]
            # text tokens of the prompt excluding special tokens and '.'
            tok = [i for i, t in enumerate(ids[b].tolist()) if t not in (101, 102, 1012, 0)]
            box_score = logits[b].max(-1).values  # used against box_threshold (as in HF post-processing)
            text_score = logits[b][:, tok].max(-1).values  # used against text_threshold
            keep = box_score > floor
            bx = out.pred_boxes[b][keep]  # cx cy w h normalized
            xyxy = torch.stack([(bx[:, 0] - bx[:, 2] / 2) * w, (bx[:, 1] - bx[:, 3] / 2) * h,
                                (bx[:, 0] + bx[:, 2] / 2) * w, (bx[:, 1] + bx[:, 3] / 2) * h], -1)
            res.append(dict(boxes=xyxy.float().cpu().numpy(), box_score=box_score[keep].float().cpu().numpy(),
                            text_score=text_score[keep].float().cpu().numpy()))
        return res


def gdino_filter(det, box_thr, text_thr):
    k = (det["box_score"] > box_thr) & (det["text_score"] > text_thr)
    return det["boxes"][k], det["box_score"][k]


class Yolo:
    def __init__(self, weights, device="cuda", imgsz=640):
        from ultralytics import YOLO
        self.m = YOLO(weights)
        self.device, self.imgsz = device, imgsz

    def __call__(self, rgbs, floor=0.01):
        # ultralytics expects BGR numpy arrays
        out = self.m.predict([r[..., ::-1] for r in rgbs], conf=floor, imgsz=self.imgsz, device=self.device,
                             verbose=False, max_det=50)
        return [dict(boxes=o.boxes.xyxy.cpu().numpy(), box_score=o.boxes.conf.cpu().numpy()) for o in out]


def yolo_filter(det, conf):
    k = det["box_score"] > conf
    return det["boxes"][k], det["box_score"][k]


class Sam2:
    def __init__(self, device="cuda"):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.pred = SAM2ImagePredictor(build_sam2(SAM2_CFG, SAM2_CKPT, device=device))

    @torch.inference_mode()
    def masks_for_boxes(self, rgbs, boxes_list):
        """Batched SAM2: one embedding per image, one mask per box. Returns list of (N,H,W) bool arrays."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.pred.set_image_batch(list(rgbs))
            nonempty = [i for i, b in enumerate(boxes_list) if len(b)]
            outs = [np.zeros((0,) + r.shape[:2], bool) for r in rgbs]
            if nonempty:
                # predict_batch needs every image; give empty images a dummy box and drop the result
                bl = [b if len(b) else np.array([[0, 0, 1, 1]], np.float32) for b in boxes_list]
                m, _, _ = self.pred.predict_batch(box_batch=bl, multimask_output=False)
                for i in nonempty:
                    mi = np.asarray(m[i])
                    if mi.ndim == 4:
                        mi = mi[:, 0]
                    elif mi.ndim == 2:
                        mi = mi[None]
                    outs[i] = mi > 0
        return outs


def union_mask(masks, keep, shape):
    m = np.zeros(shape, bool)
    for i in np.nonzero(keep)[0]:
        m |= masks[i]
    return m


class Timer:
    def __init__(self):
        self.t = {}

    def add(self, k, dt):
        self.t[k] = self.t.get(k, 0.0) + dt


def cuda_time(fn, *a, **kw):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    r = fn(*a, **kw)
    torch.cuda.synchronize()
    return r, time.perf_counter() - t0
