# egocentric-hand-segmentation

Hand masks for egocentric (first-person) video frames, comparing two auto-annotation pipelines on EgoHOS:

- **A**: zero-shot Grounding DINO tiny (text prompt) → SAM 2.1 hiera-small
- **B**: fine-tuned YOLO11s hand detector → SAM 2.1 hiera-small

Labels 1 (left hand) and 2 (right hand) are merged into one `hand` class. Object classes 3–8 are ignored.

> **Note:** `run_all.sh` finishes by running `make_readme.py`, which **overwrites this README** with an auto-generated one built from `outputs/*.json`. Save a copy of this file before re-running the pipeline.

## Hardware and software

- GPU: NVIDIA GeForce RTX 3050, 7.65 GiB VRAM (`logs/s1_data_checks.log`)
- Python 3.12.3, torch 2.5.1+cu121 (CUDA 12.1, cuDNN 90100), Ultralytics 8.3.40, transformers 4.46.3, sam2 1.1.0
- Full pinned list in `requirements.txt`. Install with `bash setup_env.sh`, which creates `.venv/` and installs SAM2 without build isolation and without its optional CUDA extension.

## Data and model weights (not in this repo)

The dataset and all weights are git-ignored. To set them up:

1. **EgoHOS dataset** (Zhang et al., ECCV 2022): https://github.com/owenzlz/EgoHOS. Download it with the Google Drive link in that repo's `download_datasets.sh` and unpack it so the folders look like this:
   ```
   data_raw/data/{train,val,test_indomain,test_outdomain}/{image,label}/
   ```
   To use another location, set `DATA_DIR` (default `$PWD/data_raw/data`). The download used here has 8,993 train, 1,124 val, 1,126 test_indomain and 500 test_outdomain images. It has no `contact` sub-folder, and none is needed.
2. **SAM 2.1 hiera-small**: download `sam2.1_hiera_small.pt` from the official SAM 2 repo (https://github.com/facebookresearch/sam2, checkpoints section) and put it at `weights/sam2.1_hiera_small.pt`. The config `configs/sam2.1/sam2.1_hiera_s.yaml` comes with the `sam2` package.
3. **Grounding DINO tiny**: `IDEA-Research/grounding-dino-tiny` on Hugging Face. There is nothing to place by hand: `transformers` downloads it into the Hugging Face cache on first use.
4. **YOLO11s base weights**: `s3_train_yolo.py` loads `yolo11s.pt`, which Ultralytics downloads automatically if it is missing. The fine-tuned detector is written to `weights/yolo_hand_best.pt`.

Licences: Grounding DINO and SAM 2.1 are Apache-2.0. Ultralytics YOLO11 is AGPL-3.0. EgoHOS frames come from Ego4D, EPIC-KITCHENS, THU-READ, the authors' own recordings (`escape_` prefix) and YouTube (out-of-domain test), and those datasets' terms apply.

## Run order

```bash
bash setup_env.sh
bash run_all.sh
```

`run_all.sh` sets `PYTHONHASHSEED=42` and runs these steps in order, each logging to `logs/<step>.log`:

| # | step | writes |
|---|---|---|
| 1 | `s1_data_checks` | `outputs/split.json`, `outputs/features.csv` |
| 2 | `s2_val_baseline` | `outputs/config.json` (GDINO thresholds, bucket cut-offs), `outputs/gate.json` (validation only) |
| 3 | `s3_train_yolo` | `outputs/training_log.csv`, YOLO weights, `yolo_conf` in `config.json` (train/val only) |
| 4 | `s4_evaluate` | `outputs/per_image.csv`, `outputs/metrics.json`, `outputs/drift.json` (test) |
| 5 | `s6_throughput` | `outputs/throughput.json` |
| 6 | `s6_docker_check` | `outputs/docker_check.json` |
| 7 | `s7_face_blur` | `outputs/face_blur.json` (qualitative, optional) |
| 8 | `make_readme` | `README.md` (overwrites this file) |

## Leakage audit and final split

In EgoHOS's official split, the same videos appear in several splits ("same video" means the same filename once the trailing frame index is removed; `logs/s1_data_checks.log`):

- **1,016 of the 1,126 test_indomain images** come from videos that are also in train (**633 shared videos**).
- train and val share 625 videos, and test_indomain and val share 333. test_outdomain shares no videos with any other split.

To fix this, **5,588 train/val images from 641 videos** that also appear in test_indomain were dropped. Validation was then rebuilt as a seeded video-level split (11% of images per source). Both test sets are the official ones, unchanged. Afterwards every pair of splits shares **0 videos**.

| split | images | escape | ego4d | epic_kitchens | thu_read | youtube |
|---|---|---|---|---|---|---|
| train | 4,011 | 0 | 2,107 | 1,268 | 636 | 0 |
| val | 518 | 16 | 262 | 161 | 79 | 0 |
| test_indomain | 1,126 | 47 | 774 | 217 | 88 | 0 |
| test_outdomain | 500 | 0 | 0 | 0 | 0 | 500 |

`outputs/split.json` lists every image's split, so the audit can be re-checked.

## Settings chosen on validation

Thresholds were fixed on validation before the test evaluation:

- Grounding DINO box_threshold = 0.2, text_threshold = 0.2
- YOLO conf = 0.2; the epoch chosen was 39 of 40
- QA gate (`gate.json`, from pipeline A on validation): mean IoU ≥ 0.5571 and share of images with IoU < 0.5 ≤ 0.4614

The logs are overwritten on every run, so an earlier test evaluation can't be ruled out from the files alone.

## Results (test sets, hand images only)

The 95% CIs come from a video-clustered bootstrap with 1,000 resamples (`s4_evaluate.py`). Clusters: 742 videos in test_indomain, 27 in test_outdomain. All test images contain hands.

| domain | pipeline | n | mean IoU | 95% CI | mean Dice | share IoU < 0.5 | QA gate |
|---|---|---|---|---|---|---|---|
| test_indomain | A: GDINO + SAM2 | 1,126 | 0.5000 | [0.4806, 0.5200] | 0.6158 | 0.5471 | FAIL |
| test_indomain | B: YOLO + SAM2 | 1,126 | 0.8593 | [0.8474, 0.8695] | 0.9136 | 0.0560 | PASS |
| test_outdomain | A: GDINO + SAM2 | 500 | 0.6484 | [0.6063, 0.6998] | 0.7562 | 0.3300 | – |
| test_outdomain | B: YOLO + SAM2 | 500 | 0.9207 | [0.9116, 0.9285] | 0.9564 | 0.0100 | – |

Paired difference (B − A, per image):

| domain | mean IoU diff | 95% CI (paired cluster bootstrap) | Wilcoxon p | n pairs |
|---|---|---|---|---|
| test_indomain | 0.3592 | [0.3391, 0.3794] | 5.46e-155 | 1,126 |
| test_outdomain | 0.2723 | [0.2240, 0.3099] | 2.36e-70 | 500 |

The CIs above are the ones in `outputs/metrics.json`. `s4_evaluate.py` uses one random generator for all bootstrap calls in turn. A separate recompute from `per_image.csv`, with a fresh generator per estimate, differs only in the third decimal place.

Results by source, blur and hand size are in `outputs/metrics.json`. Throughput for pipeline B, end to end, batch 2, RTX 3050: median 11.41 img/s (87.6 ms/img), of which SAM2 takes 69.3 ms (`outputs/throughput.json`).

## Limitations

- **The false-positive rate was never measured.** The final validation and test sets contain no hand-free images (only 2 exist in the official train/val split, and both were dropped by the leak removal). The `fp_rate_hand_free` metric is therefore n/a.
- **After the leak removal, train has no images from the authors' own recordings** (`escape_`, 0 in train). That source is scored on only 47 test images and 16 validation images.
- **YOLO + SAM2 is much worse than GDINO + SAM2 on some images.** B scores more than 0.2 IoU below A on **18 of 1,126** in-domain images and **3 of 500** out-of-domain images (`logs/s4_evaluate.log`, matching a recount from `per_image.csv`).
- **Narrow data:** only EgoHOS, and only one out-of-domain set (500 YouTube frames from 27 videos, which makes its CI wide).
- **One random seed (42).** GPU runs (cuDNN, bf16 autocast in SAM2, YOLO AMP training) are not bit-identical across runs or machines.
- Docker check: **not done**, because the NVIDIA container runtime is not installed (`outputs/docker_check.json`).
