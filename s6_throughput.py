"""STEP 6b: end-to-end throughput of version B (YOLO + SAM2 + postprocessing) on test_indomain.
Timed region per run: disk read + decode, detector (incl. its preprocessing), SAM2 (incl. preprocessing),
mask union postprocessing, PNG write to a temp dir. cuda.synchronize() around every timed segment."""
import json
import os
import tempfile
import time

import cv2
import numpy as np
import torch

import common as C


def main():
    C.print_env(); C.seed_all()
    split, cfg = C.load_json("split.json"), C.load_json("config.json")
    items = split["test_indomain"]
    bs = cfg["batch_size"]
    det, sam = C.Yolo(f"{C.ROOT}/weights/yolo_hand_best.pt"), C.Sam2()
    print(f"timing includes: disk reading+decode, detector preprocessing+inference+NMS, SAM2 preprocessing+inference, "
          f"mask union postprocessing, PNG writing. batch size fixed at {bs}; {len(items)} images; 1 warm-up pass of 5 batches.")

    def one_pass(lst, outdir):
        t = dict(read=0.0, det=0.0, sam=0.0, post_write=0.0)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for i in range(0, len(lst), bs):
            ch = lst[i:i + bs]
            a = time.perf_counter(); rgbs = [C.read_rgb(it["image"]) for it in ch]; t["read"] += time.perf_counter() - a
            dets, dt = C.cuda_time(det, rgbs); t["det"] += dt
            boxes = [C.yolo_filter(d, cfg["yolo_conf"])[0][:C.MAX_BOXES] for d in dets]
            masks, dt = C.cuda_time(sam.masks_for_boxes, rgbs, boxes); t["sam"] += dt
            a = time.perf_counter()
            for it, r, m in zip(ch, rgbs, masks):
                u = C.union_mask(m, np.ones(len(m), bool), r.shape[:2])
                cv2.imwrite(os.path.join(outdir, it["name"] + ".png"), u.astype(np.uint8) * 255)
            t["post_write"] += time.perf_counter() - a
        torch.cuda.synchronize()
        t["total"] = time.perf_counter() - t0
        return t

    runs = []
    with tempfile.TemporaryDirectory() as td:
        one_pass(items[:5 * bs], td)  # warm-up
        for k in range(3):
            t = one_pass(items, td)
            runs.append(t)
            print(f"run {k+1}: {len(items)/t['total']:.2f} img/s, {1000*t['total']/len(items):.1f} ms/img | "
                  + ", ".join(f"{n} {1000*t[n]/len(items):.1f} ms/img" for n in ("read", "det", "sam", "post_write")))
    tot = np.array([r["total"] for r in runs])
    ips = len(items) / tot
    res = dict(n_images=len(items), batch_size=bs, gpu=torch.cuda.get_device_name(0),
               included="disk read+decode, detector pre/infer/NMS, SAM2 pre/infer, mask union, PNG write",
               img_per_s=dict(min=float(ips.min()), median=float(np.median(ips)), max=float(ips.max())),
               ms_per_img=dict(min=float(1000 * tot.min() / len(items)), median=float(1000 * np.median(tot) / len(items)),
                               max=float(1000 * tot.max() / len(items))),
               component_ms_per_img_median={n: float(np.median([1000 * r[n] / len(items) for r in runs]))
                                            for n in ("read", "det", "sam", "post_write")})
    print(json.dumps(res, indent=1))
    C.save_json(res, "throughput.json")


if __name__ == "__main__":
    main()
