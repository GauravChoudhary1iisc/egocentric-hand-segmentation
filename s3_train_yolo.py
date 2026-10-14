"""STEP 3: fine-tune YOLO11s on train-split boxes derived from GT hand masks; early stopping on val;
then choose the YOLO confidence threshold on VALIDATION (same objective as GDINO). Never reads test."""
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch

import common as C
import pipeline as P

YDIR = os.path.join(C.ROOT, "yolo_data")
RUN = os.path.join(C.ROOT, "runs", "yolo_hand")


def build_dataset(split):
    for s in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(f"{YDIR}/{s}/{sub}", exist_ok=True)
        n_box = n_empty = 0
        for it in split[s]:
            stem = it["name"]
            ext = os.path.splitext(it["image"])[1]
            dst = f"{YDIR}/{s}/images/{stem}{ext}"
            if not os.path.exists(dst):
                os.symlink(it["image"], dst)
            _, lab = C.read_hand_mask(it["label"])
            h, w = lab.shape
            b = C.gt_boxes(lab)
            n_box += len(b); n_empty += len(b) == 0
            with open(f"{YDIR}/{s}/labels/{stem}.txt", "w") as f:  # empty file = background image
                for x1, y1, x2, y2 in b:
                    f.write(f"0 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n")
        print(f"YOLO {s}: {len(split[s])} images, {n_box} hand boxes, {n_empty} images without boxes (kept as background)")
    with open(f"{YDIR}/data.yaml", "w") as f:
        f.write(f"path: {YDIR}\ntrain: train/images\nval: val/images\nnames:\n  0: hand\n")


def time_batches(bs=16, n=5):
    from ultralytics import YOLO
    m = YOLO("yolo11s.pt").model.cuda().train()
    for p in m.parameters():  # checkpoint loads with requires_grad=False
        p.requires_grad_(True)
    opt = torch.optim.SGD(m.parameters(), lr=1e-3)
    x = torch.randn(bs, 3, 640, 640, device="cuda")
    for i in range(n + 1):
        if i == 1:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda"):
            out = m(x)
            loss = sum(o.float().mean() for o in out)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n
    mem = torch.cuda.max_memory_allocated() / 1024**3
    del m, opt; torch.cuda.empty_cache()
    return dt, mem


def main():
    C.print_env(); C.seed_all()
    split = C.load_json("split.json")
    build_dataset(split)

    ntr = len(split["train"])
    for bs in (8, 16):
        dt, mem = time_batches(bs)
        print(f"timing YOLO11s fwd+bwd (synthetic 640px, AMP): batch {bs}: {dt*1000:.0f} ms/batch, peak {mem:.2f} GiB "
              f"-> ~{dt*ntr/bs/60:.1f} min/epoch compute only (data loading/val excluded)")
    batch = 16 if mem < 6.0 else 8
    print(f"chosen YOLO batch {batch} (variant fixed by task: small)")

    from ultralytics import YOLO
    if os.path.isdir(RUN):
        shutil.rmtree(RUN)
    model = YOLO("yolo11s.pt")
    model.train(data=f"{YDIR}/data.yaml", epochs=40, patience=8, imgsz=640, batch=batch, seed=C.SEED,
                deterministic=True, workers=6, project=os.path.dirname(RUN), name="yolo_hand", exist_ok=True,
                save_period=1, single_cls=True, verbose=False, plots=False)
    res = pd.read_csv(f"{RUN}/results.csv")
    res.columns = [c.strip() for c in res.columns]
    res["fitness"] = 0.1 * res["metrics/mAP50(B)"] + 0.9 * res["metrics/mAP50-95(B)"]
    res.to_csv(f"{C.OUT}/training_log.csv", index=False)
    best_epoch = int(res.loc[res.fitness.idxmax(), "epoch"])
    print(res.to_string())
    print(f"epochs run: {len(res)}; chosen epoch (best val fitness = 0.1*mAP50+0.9*mAP50-95): {best_epoch}")
    os.makedirs(f"{C.ROOT}/weights", exist_ok=True)
    shutil.copy(f"{RUN}/weights/best.pt", f"{C.ROOT}/weights/yolo_hand_best.pt")

    # ---- confidence threshold on VALIDATION, full pipeline
    cfg = C.load_json("config.json")
    det, sam = C.Yolo(f"{C.ROOT}/weights/yolo_hand_best.pt"), C.Sam2()
    grid = [dict(conf=c) for c in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7)]
    recs = P.run("B", split["val"], det, sam, grid, batch=cfg["batch_size"], desc="B/val")
    scores = [P.selection_score(recs, j) for j in range(len(grid))]
    for g, s in zip(grid, scores):
        print(f"  conf {g['conf']:.2f} -> selection score {s:.4f}")
    jb = int(np.argmax(scores))
    m = P.summarize("B", recs, jb, grid[jb])
    print(f"chosen YOLO conf: {grid[jb]['conf']}; B validation metrics:", json.dumps(m, indent=1))
    cfg.update(yolo_model="yolo11s.pt fine-tuned", yolo_best_epoch=best_epoch, yolo_epochs_run=int(len(res)),
               yolo_batch=batch, yolo_conf=grid[jb]["conf"], finetuned_val_metrics=m,
               yolo_conf_grid_scores=[dict(**g, score=s) for g, s in zip(grid, scores)])
    C.save_json(cfg, "config.json")


if __name__ == "__main__":
    main()
