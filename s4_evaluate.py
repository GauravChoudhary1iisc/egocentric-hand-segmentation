"""STEPS 4+5: single evaluation of A and B on the test splits with settings frozen in config.json / gate.json;
per_image.csv, metrics.json, gate check, regression list, drift.json. All from this one run."""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import common as C
import pipeline as P

N_BOOT = 1000


def bucket(v, cuts, names=("low", "mid", "high")):
    return names[int(np.searchsorted(cuts, v, side="right"))]


def boot_ci(df, col, cluster, rng):
    """Cluster bootstrap of the mean of df[col]; cluster column = video id (or image name)."""
    groups = [g[col].to_numpy() for _, g in df.groupby(cluster)]
    sums = np.array([g.sum() for g in groups]); cnts = np.array([len(g) for g in groups])
    k = len(groups)
    idx = rng.integers(0, k, size=(N_BOOT, k))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def psi(ref, new, bins=10):
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    r = np.histogram(ref, edges)[0] / len(ref); n = np.histogram(new, edges)[0] / len(new)
    r = np.clip(r, 1e-4, None); n = np.clip(n, 1e-4, None)
    return float(np.sum((n - r) * np.log(n / r)))


def main():
    C.print_env(); C.seed_all()
    split, cfg, gate = C.load_json("split.json"), C.load_json("config.json"), C.load_json("gate.json")
    feats = pd.read_csv(f"{C.OUT}/features.csv").set_index(["split", "name"])
    test_splits = [s for s in split if s.startswith("test")]
    print("evaluating once on:", test_splits, "| frozen config:", {k: cfg[k] for k in
          ("gdino_box_threshold", "gdino_text_threshold", "yolo_conf", "yolo_best_epoch", "batch_size")})

    versions = {
        "A": (lambda: C.GDino(), [dict(box_threshold=cfg["gdino_box_threshold"], text_threshold=cfg["gdino_text_threshold"])]),
        "B": (lambda: C.Yolo(f"{C.ROOT}/weights/yolo_hand_best.pt"), [dict(conf=cfg["yolo_conf"])]),
    }
    sam = C.Sam2()
    rows, recs_all = [], {}
    for v, (mk, setting) in versions.items():
        det = mk()
        for s in test_splits:
            recs = P.run(v, split[s], det, sam, setting, batch=cfg["batch_size"], desc=f"{v}/{s}")
            recs_all[(v, s)] = recs
            meta = {it["name"]: it for it in split[s]}
            for r in recs:
                f = feats.loc[(s, r["name"])]
                rr = r["res"][0]
                rows.append(dict(version=v, split=s, image=r["name"], source=meta[r["name"]]["source"],
                                 video=meta[r["name"]]["video"], blur=f.blur, hand_area_frac=f.hand_area_frac,
                                 has_hand=r["has_hand"], iou=rr["iou"] if r["has_hand"] else np.nan,
                                 dice=rr["dice"] if r["has_hand"] else np.nan, detected=rr["detected"],
                                 n_boxes=rr["n_boxes"], pred_area_frac=rr["pred_area"]))
        del det
    df = pd.DataFrame(rows)
    df["blur_bucket"] = [bucket(b, cfg["blur_cutoffs_train_tertiles"]) for b in df.blur]
    df["size_bucket"] = [bucket(a, cfg["hand_size_cutoffs_train_tertiles"]) if a > 0 else "no_hand" for a in df.hand_area_frac]
    df.to_csv(f"{C.OUT}/per_image.csv", index=False)

    rng = np.random.default_rng(C.SEED)
    metrics = dict(config_used={k: cfg[k] for k in ("gdino_box_threshold", "gdino_text_threshold", "yolo_conf",
                                                   "yolo_best_epoch")}, splits={})
    for s in test_splits:
        sd = df[df.split == s]
        hand = sd[sd.has_hand]
        vid_known = hand.video.notna().all()
        cluster = "video" if vid_known else "image"
        out = dict(bootstrap_unit="video/sequence" if vid_known else "image", versions={})
        for v in versions:
            m = P.summarize(v, recs_all[(v, s)], 0, versions[v][1][0])
            hv = hand[hand.version == v]
            m["mean_iou_ci95"] = boot_ci(hv, "iou", cluster, rng)
            m["by_source"] = {k: dict(n=int(len(g)), mean_iou=float(g.iou.mean())) for k, g in hv.groupby("source")}
            m["by_blur_bucket"] = {k: dict(n=int(len(g)), mean_iou=float(g.iou.mean())) for k, g in hv.groupby("blur_bucket")}
            m["by_size_bucket"] = {k: dict(n=int(len(g)), mean_iou=float(g.iou.mean())) for k, g in hv.groupby("size_bucket")}
            fv = sd[(sd.version == v) & ~sd.has_hand]
            m["fp_rate_hand_free_by_source"] = {k: dict(n=int(len(g)), fp_rate=float(g.detected.mean()))
                                                for k, g in fv.groupby("source")}
            out["versions"][v] = m
        # paired A vs B on hand images
        a = hand[hand.version == "A"].set_index("image"); b = hand[hand.version == "B"].set_index("image")
        d = (b.iou - a.iou.reindex(b.index)).rename("diff").to_frame()
        d[cluster] = d.index if cluster == "image" else b.video
        w = stats.wilcoxon(b.iou, a.iou.reindex(b.index))
        out["paired_B_minus_A"] = dict(mean_iou_diff=float(d["diff"].mean()),
                                       paired_bootstrap_ci95=boot_ci(d.reset_index(drop=True), "diff", cluster, rng),
                                       wilcoxon_statistic=float(w.statistic), wilcoxon_p=float(w.pvalue),
                                       n_pairs=int(len(d)))
        reg = d[d["diff"] < -0.2]
        out["regressions_B_worse_than_A_by_iou_gt_0_2"] = [
            dict(image=i, iou_A=float(a.iou[i]), iou_B=float(b.iou[i])) for i in reg.sort_values("diff").index]
        metrics["splits"][s] = out

    # ---- gate (primary test split = in-domain)
    gate_split = "test_indomain" if "test_indomain" in test_splits else test_splits[0]
    metrics["gate"] = dict(split=gate_split, results={})
    print(f"\n=== QA gate on {gate_split} (gate.json written before test) ===")
    for v in versions:
        m = metrics["splits"][gate_split]["versions"][v]
        res = []
        for rule in gate["rules"]:
            val = m["mean_iou" if rule["name"] == "mean_iou_hand_images" else rule["name"]]
            ok = val >= rule["threshold"] if rule["op"] == ">=" else val <= rule["threshold"]
            res.append(dict(rule=rule["name"], op=rule["op"], threshold=rule["threshold"], value=val, passed=bool(ok)))
            print(f"  {v}: {rule['name']} = {val:.4f} {rule['op']} {rule['threshold']:.4f} -> {'PASS' if ok else 'FAIL'}")
        metrics["gate"]["results"][v] = dict(rules=res, overall="PASS" if all(r["passed"] for r in res) else "FAIL")
        print(f"  {v}: OVERALL {metrics['gate']['results'][v]['overall']}")
    C.save_json(metrics, "metrics.json")

    # ---- print metrics
    for s in test_splits:
        o = metrics["splits"][s]
        print(f"\n=== {s} (bootstrap unit: {o['bootstrap_unit']}) ===")
        for v, m in o["versions"].items():
            print(f"  [{v}] " + ", ".join(f"{k}={m[k]}" for k in ("n_hand_images", "n_hand_free_images", "mean_iou",
                  "mean_iou_ci95", "mean_dice", "share_iou_lt_0_5", "det_precision", "det_recall", "ap50", "fp_rate_hand_free")))
            for key in ("by_source", "by_blur_bucket", "by_size_bucket", "fp_rate_hand_free_by_source"):
                print(f"      {key}: {m[key]}")
        print(f"  paired B-A: {o['paired_B_minus_A']}")
        print(f"  regressions (B worse than A by IoU > 0.2): {len(o['regressions_B_worse_than_A_by_iou_gt_0_2'])}")
        for r in o["regressions_B_worse_than_A_by_iou_gt_0_2"]:
            print(f"      {r['image']}: A={r['iou_A']:.3f} B={r['iou_B']:.3f}")

    # ---- drift: reference = val (in-domain); new = test_outdomain
    f = feats.reset_index()
    ref = f[f.split == "val"]
    new_name = "test_outdomain" if "test_outdomain" in split else None
    if new_name is None:
        src_counts = f[f.split != "val"].source.value_counts()
        new_name = f"source:{src_counts.idxmin()}"
        new = f[f.source == src_counts.idxmin()]
    else:
        new = f[f.split == new_name]
    drift = dict(reference="val (in-domain)", new_batch=new_name, n_ref=int(len(ref)), n_new=int(len(new)),
                 note="PSI > 0.2 is a common rule-of-thumb convention, not a proven threshold.", features={})
    print(f"\n=== drift: reference=val (n={len(ref)}) vs {new_name} (n={len(new)}) ===")
    for k in ("brightness", "contrast", "blur"):
        p = psi(ref[k].to_numpy(), new[k].to_numpy())
        ks = stats.ks_2samp(ref[k], new[k])
        drift["features"][k] = dict(psi=p, ks_statistic=float(ks.statistic), ks_p=float(ks.pvalue),
                                    ref_mean=float(ref[k].mean()), new_mean=float(new[k].mean()),
                                    psi_gt_0_2_convention=bool(p > 0.2))
        print(f"  {k}: PSI={p:.4f} KS D={ks.statistic:.4f} p={ks.pvalue:.3g} (ref mean {ref[k].mean():.2f}, new mean {new[k].mean():.2f})")
    C.save_json(drift, "drift.json")


if __name__ == "__main__":
    main()
