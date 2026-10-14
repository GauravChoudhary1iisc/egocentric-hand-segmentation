"""Builds README.md. Every number is read from the JSON/CSV outputs; nothing is typed by hand."""
import json
import os

import pandas as pd

import common as C


def j(name):
    p = os.path.join(C.OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, d=4):
    return "n/a" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def main():
    split, cfg, gate, met, drift = j("split.json"), j("config.json"), j("gate.json"), j("metrics.json"), j("drift.json")
    thr, dock, face = j("throughput.json"), j("docker_check.json"), j("face_blur.json")
    L = ["# EgoHOS hand auto-annotation: Grounding DINO + SAM2 vs fine-tuned YOLO + SAM2", ""]
    L += ["## Data and licences",
          "- Dataset: EgoHOS (Zhang et al., ECCV 2022), https://github.com/owenzlz/EgoHOS, downloaded via the Google Drive link in "
          "`download_datasets.sh`. Its frames come from **Ego4D, EPIC-KITCHENS and THU-READ** (plus the authors' own frames, "
          "prefix `escape_`, and YouTube frames in the out-of-domain test); the terms of those source datasets apply.",
          "- Models: Grounding DINO tiny (`IDEA-Research/grounding-dino-tiny`, Apache-2.0), SAM 2.1 hiera-small (Apache-2.0), "
          "Ultralytics YOLO11s (AGPL-3.0; commercial use needs an Ultralytics licence), OpenCV Haar cascade (Apache-2.0 / BSD).",
          "- Labels: 1=left hand, 2=right hand used and merged into one `hand` class; object classes 3-8 ignored.", ""]
    L += ["## Split", f"- {split['description']}. Video = {split['video_definition']}",
          f"- Train/val images removed because their video also appears in official test_indomain: {split['n_dropped_for_leak']}",
          "- Official-split leak check: " + "; ".join(f"{k}: {v['shared_videos']} shared videos" for k, v in split['leak_official'].items()),
          "- Final counts: " + ", ".join(f"{s}={len(split[s])}" for s in split if isinstance(split[s], list)), ""]
    L += ["## Settings fixed on validation (before test)",
          f"- Grounding DINO box_threshold={cfg['gdino_box_threshold']}, text_threshold={cfg['gdino_text_threshold']}; "
          f"YOLO conf={cfg.get('yolo_conf')}; YOLO chosen epoch={cfg.get('yolo_best_epoch')} of {cfg.get('yolo_epochs_run')} run",
          f"- Selection objective: {cfg['selection_objective']}",
          f"- Blur tertile cut-offs (train): {cfg['blur_cutoffs_train_tertiles']}; hand-size cut-offs (train): "
          f"{cfg['hand_size_cutoffs_train_tertiles']}",
          "- Gate (gate.json, derived from baseline A on validation): " +
          "; ".join(f"{r['name']} {r['op']} {f(r['threshold'])}" for r in gate["rules"]), ""]
    if met:
        L += ["## Test results (single run)"]
        cols = ["n_hand_images", "n_hand_free_images", "mean_iou", "mean_iou_ci95", "mean_dice", "share_iou_lt_0_5",
                "det_precision", "det_recall", "ap50", "fp_rate_hand_free"]
        for s, o in met["splits"].items():
            L += [f"### {s} (bootstrap unit: {o['bootstrap_unit']})", "", "| version | " + " | ".join(cols) + " |",
                  "|" + "---|" * (len(cols) + 1)]
            for v, m in o["versions"].items():
                L.append(f"| {v} | " + " | ".join(f(m[c]) if not isinstance(m[c], list) else
                                                   "[" + ", ".join(f(x) for x in m[c]) + "]" for c in cols) + " |")
            p = o["paired_B_minus_A"]
            L += ["", f"Paired B-A mean IoU diff {f(p['mean_iou_diff'])}, 95% CI [{f(p['paired_bootstrap_ci95'][0])}, "
                  f"{f(p['paired_bootstrap_ci95'][1])}], Wilcoxon p={p['wilcoxon_p']:.3g} (n={p['n_pairs']}). "
                  f"Regressions (B worse than A by IoU>0.2): {len(o['regressions_B_worse_than_A_by_iou_gt_0_2'])}.", ""]
            for key in ("by_source", "by_blur_bucket", "by_size_bucket"):
                L.append(f"- {key}: " + "; ".join(f"{v}: " + ", ".join(f"{k} (n={d['n']}) {f(d['mean_iou'])}" for k, d in
                                                                     o['versions'][v][key].items()) for v in o["versions"]))
            L.append("")
        L += [f"### QA gate on {met['gate']['split']}"]
        for v, g in met["gate"]["results"].items():
            L.append(f"- {v}: **{g['overall']}** — " + "; ".join(
                f"{r['rule']} {f(r['value'])} {r['op']} {f(r['threshold'])} {'PASS' if r['passed'] else 'FAIL'}" for r in g["rules"]))
        L.append("")
    else:
        L += ["## Test results", "NOT DONE (metrics.json missing).", ""]
    if drift:
        L += [f"## Drift: {drift['reference']} (n={drift['n_ref']}) vs {drift['new_batch']} (n={drift['n_new']})",
              f"{drift['note']}", ""] + [f"- {k}: PSI {f(d['psi'])}, KS D {f(d['ks_statistic'])}, p {d['ks_p']:.3g}"
                                         for k, d in drift["features"].items()] + [""]
    L += ["## Throughput (version B, end to end)"]
    if thr:
        L += [f"- {thr['gpu']}, batch {thr['batch_size']}, {thr['n_images']} images, 3 runs after warm-up. Included: {thr['included']}",
              f"- img/s min/median/max: {f(thr['img_per_s']['min'],2)}/{f(thr['img_per_s']['median'],2)}/{f(thr['img_per_s']['max'],2)}; "
              f"ms/img min/median/max: {f(thr['ms_per_img']['min'],1)}/{f(thr['ms_per_img']['median'],1)}/{f(thr['ms_per_img']['max'],1)}",
              "- median component ms/img: " + ", ".join(f"{k} {f(v,1)}" for k, v in thr["component_ms_per_img_median"].items())]
    else:
        L.append("NOT DONE")
    L += ["", "## Docker", "```", "docker build -t egohos-hands .",
          "docker run --rm --gpus all -v $PWD/sample50:/in:ro -v $PWD/docker_out:/out egohos-hands --input /in --output /out", "```"]
    L.append(f"- Result: {dock['status']}" + (f" — {dock.get('reason')}" if dock and dock.get('reason') else
                                             (f" — max pixel diff {dock['max_pixel_diff']}, images differing {dock['n_images_differing']}/50" if dock else "")) if dock else "- NOT DONE")
    if face:
        L += ["", f"## Face blur (qualitative only, no ground truth)", f"- {face['detector']}: {face['n_frames']} frames, "
              f"{sum(n > 0 for n in face['faces_per_frame'])} with a detection; see face_blur_examples/."]
    L += ["", "## Reproducibility", "- SEED=42 for python/numpy/torch/ultralytics and for all splits/bootstraps. GPU runs (cuDNN, "
          "bf16 autocast in SAM2, YOLO AMP training) are **not bit-identical** across runs or machines.",
          "- Run everything with `bash run_all.sh`; logs in logs/, outputs in outputs/."]
    open(os.path.join(C.ROOT, "README.md"), "w").write("\n".join(L) + "\n")
    print("README.md written")


if __name__ == "__main__":
    main()
