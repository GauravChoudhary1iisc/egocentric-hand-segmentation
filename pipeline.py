"""Run detector -> SAM2 on a list of items, evaluating one or many threshold settings in a single pass.
SAM2 is run once per candidate box (score above a low floor, top MAX_CAND); each threshold setting
then just selects a subset of those boxes and unions their masks."""
import time

import numpy as np
import torch
from tqdm import tqdm

import common as C

MAX_CAND = 20


def candidates(version, det_out):
    """Sort by score, cap to MAX_CAND. Returns boxes, score (ranking/AP score), text_score (gdino only)."""
    s = det_out["box_score"]
    o = np.argsort(-s)[:MAX_CAND]
    ts = det_out.get("text_score", s)[o]
    return det_out["boxes"][o], s[o], ts


def keep_flags(version, score, text_score, setting):
    if version == "A":
        k = (score > setting["box_threshold"]) & (text_score > setting["text_threshold"])
    else:
        k = score > setting["conf"]
    # boxes are already in score order; cap the number used
    idx = np.nonzero(k)[0][:C.MAX_BOXES]
    out = np.zeros(len(score), bool); out[idx] = True
    return out


def run(version, items, detector, sam, settings, batch=4, desc=""):
    """items: list of dicts with image, label. Returns list of per-image records.
    record['res'][j] = dict(iou, dice, n_boxes, detected, keep) for settings[j]."""
    recs = []
    for i in tqdm(range(0, len(items), batch), desc=desc, mininterval=30):
        chunk = items[i:i + batch]
        rgbs = [C.read_rgb(it["image"]) for it in chunk]
        dets = detector(rgbs)
        cands = [candidates(version, d) for d in dets]
        masks = sam.masks_for_boxes(rgbs, [c[0] for c in cands])
        for it, r, (bx, sc, ts), m in zip(chunk, rgbs, cands, masks):
            gt, lab = C.read_hand_mask(it["label"])
            gtb = C.gt_boxes(lab)
            res = []
            for st in settings:
                k = keep_flags(version, sc, ts, st)
                pm = C.union_mask(m, k, gt.shape)
                iou, dice = C.iou_dice(pm, gt)
                res.append(dict(iou=iou, dice=dice, n_boxes=int(k.sum()), detected=bool(k.any()), keep=k,
                                pred_area=float(pm.mean())))
            recs.append(dict(name=it["name"], boxes=bx, score=sc, text_score=ts, gt_boxes=gtb,
                             has_hand=bool(gt.any()), res=res))
    return recs


def summarize(version, recs, j, setting):
    """Metrics for setting index j."""
    hand = [r for r in recs if r["has_hand"]]
    free = [r for r in recs if not r["has_hand"]]
    ious = np.array([r["res"][j]["iou"] for r in hand])
    dices = np.array([r["res"][j]["dice"] for r in hand])
    # operating-point detection precision/recall at box IoU 0.5
    tp = fp = 0; n_gt = 0
    ap_scores, ap_tp = [], []
    for r in recs:
        n_gt += len(r["gt_boxes"])
        k = r["res"][j]["keep"]
        t = C.match_boxes(r["boxes"][k], r["score"][k], r["gt_boxes"])
        tp += int(t.sum()); fp += int((~t).sum())
        # AP: rank every candidate allowed by the non-score filter (text filter for GDINO)
        if version == "A":
            ak = r["text_score"] > setting["text_threshold"]
        else:
            ak = np.ones(len(r["score"]), bool)
        ta = C.match_boxes(r["boxes"][ak], r["score"][ak], r["gt_boxes"])
        ap_scores += list(r["score"][ak]); ap_tp += list(ta)
    return dict(
        n_images=len(recs), n_hand_images=len(hand), n_hand_free_images=len(free),
        mean_iou=float(ious.mean()) if len(ious) else None,
        mean_dice=float(dices.mean()) if len(dices) else None,
        share_iou_lt_0_5=float((ious < 0.5).mean()) if len(ious) else None,
        det_precision=tp / (tp + fp) if tp + fp else None,
        det_recall=tp / n_gt if n_gt else None,
        ap50=C.average_precision(ap_scores, ap_tp, n_gt),
        n_gt_boxes=n_gt,
        fp_rate_hand_free=float(np.mean([r["res"][j]["detected"] for r in free])) if free else None,
    )


def selection_score(recs, j):
    """Objective used to choose thresholds on VALIDATION: mean over ALL images of IoU, where a
    hand-free image scores 1 if nothing is predicted and 0 otherwise (penalises false positives)."""
    v = []
    for r in recs:
        if r["has_hand"]:
            v.append(r["res"][j]["iou"])
        else:
            v.append(0.0 if r["res"][j]["detected"] else 1.0)
    return float(np.mean(v))
