"""Auto-annotate hands: YOLO (fine-tuned, version B) -> boxes -> SAM2-small box prompt -> union mask.
Usage: python annotate.py --input DIR --output DIR [--weights best.pt] [--conf X] [--batch N]
Writes <output>/<stem>.png (0/255 hand mask) and <output>/annotations.csv (image, hand_area_frac, n_boxes)."""
import argparse
import csv
import json
import os

import cv2
import numpy as np

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default=os.path.join(C.ROOT, "weights", "yolo_hand_best.pt"))
    ap.add_argument("--config", default=os.path.join(C.OUT, "config.json"))
    ap.add_argument("--conf", type=float, default=None, help="default: yolo_conf from config.json")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    C.seed_all()
    conf = a.conf if a.conf is not None else json.load(open(a.config))["yolo_conf"]
    os.makedirs(a.output, exist_ok=True)
    files = sorted(f for f in os.listdir(a.input) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    det, sam = C.Yolo(a.weights), C.Sam2()
    rows = []
    for i in range(0, len(files), a.batch):
        names = files[i:i + a.batch]
        rgbs = [C.read_rgb(os.path.join(a.input, n)) for n in names]
        dets = det(rgbs)
        boxes = [C.yolo_filter(d, conf)[0][:C.MAX_BOXES] for d in dets]
        masks = sam.masks_for_boxes(rgbs, boxes)
        for n, r, b, m in zip(names, rgbs, boxes, masks):
            u = C.union_mask(m, np.ones(len(m), bool), r.shape[:2])
            cv2.imwrite(os.path.join(a.output, os.path.splitext(n)[0] + ".png"), u.astype(np.uint8) * 255)
            rows.append([n, f"{u.mean():.6f}", len(b)])
    with open(os.path.join(a.output, "annotations.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "hand_area_frac", "n_boxes"]); w.writerows(rows)
    print(f"wrote {len(rows)} masks to {a.output}")


if __name__ == "__main__":
    main()
