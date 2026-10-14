"""STEP 2: timing estimate, Grounding DINO thresholds on VALIDATION, bucket cut-offs from TRAIN, gate.json.
Touches only train (features) and val. Never reads test."""
import itertools
import json

import numpy as np
import pandas as pd

import common as C
import pipeline as P


def main():
    C.print_env(); C.seed_all()
    split = C.load_json("split.json")
    feats = pd.read_csv(f"{C.OUT}/features.csv")

    # ---- bucket cut-offs (tertiles) on TRAIN only; hand-size tertiles on train images that contain hands
    tr = feats[feats.split == "train"]
    blur_cut = np.quantile(tr.blur, [1 / 3, 2 / 3]).tolist()
    size_cut = np.quantile(tr[tr.hand_area_frac > 0].hand_area_frac, [1 / 3, 2 / 3]).tolist()
    print(f"blur tertile cut-offs (train): {blur_cut}")
    print(f"hand-size tertile cut-offs (train, hand images): {size_cut}")

    val = split["val"]
    det, sam = C.GDino(), C.Sam2()

    # ---- timing estimate before committing to sizes / batch
    probe = val[:16]
    rgbs = [C.read_rgb(it["image"]) for it in probe]
    print("timing probe (16 val images, after 1 warm-up batch):")
    est = {}
    for bs in (1, 2, 4):
        det(rgbs[:bs]); sam.masks_for_boxes(rgbs[:bs], [np.array([[10, 10, 100, 100]], np.float32)] * bs)
        td = ts = 0.0
        for i in range(0, 16, bs):
            d, t1 = C.cuda_time(det, rgbs[i:i + bs])
            bx = [P.candidates("A", x)[0] for x in d]
            _, t2 = C.cuda_time(sam.masks_for_boxes, rgbs[i:i + bs], bx)
            td += t1; ts += t2
        import torch
        mem = torch.cuda.max_memory_allocated() / 1024**3
        est[bs] = (td + ts) / 16
        print(f"  batch {bs}: GDINO {1000*td/16:.1f} ms/img, SAM2 {1000*ts/16:.1f} ms/img, "
              f"peak alloc {mem:.2f} GiB -> est. {est[bs]*len(val)/60:.1f} min for val ({len(val)} imgs)")
    bs = min(est, key=est.get)
    print(f"chosen batch size: {bs} (fastest in probe; model sizes are fixed by the task: GDINO tiny, SAM2 small)")

    # ---- threshold grid on VALIDATION
    grid = [dict(box_threshold=b, text_threshold=t) for b, t in
            itertools.product([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], [0.2, 0.25, 0.3, 0.35, 0.4])]
    recs = P.run("A", val, det, sam, grid, batch=bs, desc="A/val")
    scores = [P.selection_score(recs, j) for j in range(len(grid))]
    for g, s in zip(grid, scores):
        print(f"  box {g['box_threshold']:.2f} text {g['text_threshold']:.2f} -> selection score {s:.4f}")
    jb = int(np.argmax(scores))
    best = grid[jb]
    m = P.summarize("A", recs, jb, best)
    print(f"chosen GDINO thresholds: {best} (val selection score {scores[jb]:.4f})")
    print("baseline A validation metrics:", json.dumps(m, indent=1))

    cfg = dict(seed=C.SEED, gdino_model=C.GDINO_ID, gdino_prompt="hand.", gdino_box_threshold=best["box_threshold"],
               gdino_text_threshold=best["text_threshold"], selection_objective=P.selection_score.__doc__.strip(),
               sam2_cfg=C.SAM2_CFG, max_boxes=C.MAX_BOXES, max_candidates=P.MAX_CAND, batch_size=bs,
               blur_cutoffs_train_tertiles=blur_cut, hand_size_cutoffs_train_tertiles=size_cut,
               baseline_val_metrics=m, threshold_grid_scores=[dict(**g, score=s) for g, s in zip(grid, scores)])
    C.save_json(cfg, "config.json")

    gate = dict(
        derived_from="baseline A on VALIDATION (written before any test evaluation)",
        rules=[dict(name="mean_iou_hand_images", op=">=", threshold=m["mean_iou"]),
               dict(name="share_iou_lt_0_5", op="<=", threshold=m["share_iou_lt_0_5"])])
    C.save_json(gate, "gate.json")
    print("gate.json:", json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
