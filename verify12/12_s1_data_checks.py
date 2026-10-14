"""STEP 1: inspect layout + label encoding, counts per split/source, video leak check, hand-free images,
leak-free video-level split -> split.json; per-image features (blur, brightness, contrast, hand area) -> features.csv."""
import os
import re
from collections import Counter
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

import common as C

SPLITS = ["train", "val", "test_indomain", "test_outdomain"]
VAL_FRAC = 0.11  # roughly the official val share of train+val (1124 / 10117)


def source_of(name):
    p = name.split("_")[0]
    return {"ego4d": "ego4d", "epic": "epic_kitchens", "thu": "thu_read", "escape": "authors_own(escape)",
            "youtube": "youtube"}.get(p, "unknown")


def video_of(name):
    """'Same video' = identical filename after removing the trailing frame index (and EPIC's '_frame')."""
    v = re.sub(r"_\d+$", "", name)
    return re.sub(r"_frame$", "", v)


def tree(root, depth=2):
    for dp, dn, fn in sorted(os.walk(root)):
        lvl = dp[len(root):].count(os.sep)
        if lvl > depth:
            continue
        exts = Counter(os.path.splitext(f)[1] for f in fn)
        print("  " * lvl + f"{os.path.basename(dp) or dp}/  ({len(fn)} files {dict(exts)})")


def feat(args):
    s, name, img, lab = args
    rgb = C.read_rgb(img)
    hand, l = C.read_hand_mask(lab)
    st = C.image_stats(rgb)
    return dict(split=s, name=name, h=rgb.shape[0], w=rgb.shape[1], label_shape_ok=l.shape == rgb.shape[:2],
                hand_area_frac=float(hand.mean()), left_px=int((l == 1).sum()), right_px=int((l == 2).sum()),
                values=tuple(np.unique(l).tolist()), label_ndim=cv2.imread(lab, cv2.IMREAD_UNCHANGED).ndim, **st)


def main():
    C.print_env(); C.seed_all()
    print(f"DATA_DIR = {C.DATA_DIR}\nfolder structure:")
    tree(C.DATA_DIR)
    present = [s for s in SPLITS if os.path.isdir(f"{C.DATA_DIR}/{s}")]
    print("official splits found:", present)
    print("NOTE: the repo README lists a 'contact' sub-folder per split; it is",
          "present" if os.path.isdir(f"{C.DATA_DIR}/train/contact") else "NOT present in this download (not needed here).")

    items, jobs = {}, []
    for s in present:
        ims = sorted(os.listdir(f"{C.DATA_DIR}/{s}/image"))
        labs = set(os.listdir(f"{C.DATA_DIR}/{s}/label"))
        missing = [i for i in ims if os.path.splitext(i)[0] + ".png" not in labs]
        print(f"{s}: {len(ims)} images, {len(labs)} labels, images without label: {len(missing)}")
        items[s] = []
        for i in ims:
            n = os.path.splitext(i)[0]
            if n + ".png" not in labs:
                continue
            it = dict(name=n, image=f"{C.DATA_DIR}/{s}/image/{i}", label=f"{C.DATA_DIR}/{s}/label/{n}.png",
                      source=source_of(n), video=video_of(n))
            items[s].append(it); jobs.append((s, n, it["image"], it["label"]))

    with Pool(8) as p:
        F = pd.DataFrame(p.map(feat, jobs, chunksize=32))
    allv = Counter(v for vs in F["values"] for v in vs)
    print("\nlabel files: ndim counts", dict(Counter(F.label_ndim)), "| label/image size mismatch:", int((~F.label_shape_ok).sum()))
    print("unique pixel values over ALL label masks (value: #masks containing it):", dict(sorted(allv.items())))
    for s in present:
        vs = Counter(v for vs in F[F.split == s]["values"] for v in vs)
        print(f"  {s}: {dict(sorted(vs.items()))}")
    print("README encoding: 0 bg, 1 left hand, 2 right hand, 3-8 interacting objects. Using ONLY 1 and 2, merged into one 'hand' class"
          " (left/right are separate ids, so merging is always possible).")
    print(f"images with both left and right hand pixels: {int(((F.left_px > 0) & (F.right_px > 0)).sum())}, "
          f"left only: {int(((F.left_px > 0) & (F.right_px == 0)).sum())}, right only: {int(((F.left_px == 0) & (F.right_px > 0)).sum())}")
    print("image sizes (h x w): ", dict(Counter(zip(F.h, F.w)).most_common(8)))

    meta = pd.DataFrame([dict(split=s, **it) for s in present for it in items[s]])
    print("\ncounts per split x source:")
    print(pd.crosstab(meta.split, meta.source, margins=True).to_string())
    print("\nvideos per split x source:")
    print(meta.groupby(["split", "source"]).video.nunique().unstack(fill_value=0).to_string())

    # ---- leak check
    vids = {s: set(meta[meta.split == s].video) for s in present}
    print("\nvideo leak check ('same video' = same filename minus trailing frame number):")
    leak = {}
    for a in present:
        for b in present:
            if a < b:
                common = vids[a] & vids[b]
                na = int(meta[(meta.split == a) & meta.video.isin(common)].shape[0])
                nb = int(meta[(meta.split == b) & meta.video.isin(common)].shape[0])
                leak[f"{a}&{b}"] = dict(shared_videos=len(common), images_in_first=na, images_in_second=nb)
                print(f"  {a} & {b}: {len(common)} shared videos ({na} {a} images, {nb} {b} images)")

    # ---- hand-free images
    F["has_hand"] = F.hand_area_frac > 0
    print("\nhand-free images per split:", F[~F.has_hand].groupby("split").size().to_dict(),
          "| handling: kept everywhere; YOLO sees them as background (empty label file); IoU/Dice are reported on hand images "
          "only; hand-free images are used for the false-positive rate.")

    # ---- leak-free video-level split: official test_indomain kept unchanged; train+val pooled, every frame of a video
    # that also appears in test_indomain removed; remaining videos split into train/val per source (seeded).
    test_v = vids.get("test_indomain", set())
    pool = meta[meta.split.isin(["train", "val"]) & ~meta.video.isin(test_v)]
    dropped = meta[meta.split.isin(["train", "val"]) & meta.video.isin(test_v)]
    rng = np.random.default_rng(C.SEED)
    val_v = set()
    for src, g in pool.groupby("source"):
        vv = sorted(g.video.unique())
        rng.shuffle(vv)
        target, acc = VAL_FRAC * len(g), 0
        sizes = g.video.value_counts()
        for v in vv:
            if acc >= target:
                break
            val_v.add(v); acc += sizes[v]
    new = {"train": [], "val": []}
    by_name = {(s, it["name"]): it for s in present for it in items[s]}
    for _, r in pool.iterrows():
        new["val" if r.video in val_v else "train"].append(by_name[(r.split, r["name"])])
    for s in present:
        if s.startswith("test"):
            new[s] = items[s]
    print(f"\nleak-free video-level split: dropped {len(dropped)} train/val images from {dropped.video.nunique()} videos shared with "
          f"test_indomain; official val replaced by a seeded video-level val ({VAL_FRAC:.0%} of images per source).")
    nm = pd.DataFrame([dict(split=s, **it) for s in new for it in new[s]])
    print(pd.crosstab(nm.split, nm.source, margins=True).to_string())
    for a in new:
        for b in new:
            if a < b:
                print(f"  check {a} & {b}: shared videos = {len(set(nm[nm.split == a].video) & set(nm[nm.split == b].video))}")
    # name collisions across splits would break joins
    assert not nm.duplicated(["split", "name"]).any()

    C.save_json(dict(description="final split used by all steps; test splits are the official ones; train/val are the "
                     "leak-free video-level re-split", seed=C.SEED, video_definition=video_of.__doc__, leak_official=leak,
                     n_dropped_for_leak=int(len(dropped)), **new), "split.json")
    # features keyed by (final split, name)
    key = {}
    for s in new:
        for it in new[s]:
            key[it["image"]] = s
    F["image_path"] = [by_name[(s, n)]["image"] for s, n in zip(F.split, F.name)]
    F["official_split"] = F.split
    F["split"] = [key.get(p, "dropped_leak") for p in F.image_path]
    F.drop(columns=["values"]).to_csv(f"{C.OUT}/features.csv", index=False)
    print("hand-free images per FINAL split:", F[~F.has_hand].groupby("split").size().to_dict())
    print("saved outputs/split.json and outputs/features.csv")


if __name__ == "__main__":
    main()
