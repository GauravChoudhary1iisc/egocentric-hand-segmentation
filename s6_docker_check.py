"""STEP 6a: docker build + run on a 50-image sample and compare with host masks.
Stops (NOT DONE, with the exact error) if Docker cannot give the container GPU access."""
import os
import random
import shutil
import subprocess

import cv2
import numpy as np

import common as C


def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def main():
    C.seed_all()
    split = C.load_json("split.json")
    sample = os.path.join(C.ROOT, "sample50")
    shutil.rmtree(sample, ignore_errors=True); os.makedirs(sample)
    items = random.Random(C.SEED).sample(split["test_indomain"], 50)
    for it in items:
        shutil.copy(it["image"], sample)
    host_out = os.path.join(C.ROOT, "host_out")
    shutil.rmtree(host_out, ignore_errors=True)
    rc, out = sh(f"{C.ROOT}/.venv/bin/python {C.ROOT}/annotate.py --input {sample} --output {host_out}")
    print("host annotate:", out.splitlines()[-1] if out else rc)

    res = dict(status="NOT DONE")
    rc, out = sh("docker run --rm --gpus all hello-world")
    if rc != 0:
        res["reason"] = ("Docker cannot give containers GPU access (NVIDIA container runtime / nvidia-container-toolkit is not "
                         "installed; `docker info` lists only runc). The image was not built or run. Exact error from "
                         f"`docker run --rm --gpus all hello-world`: {out}")
        print("Docker step NOT DONE:", res["reason"])
        C.save_json(res, "docker_check.json")
        return
    rc, out = sh(f"cd {C.ROOT} && docker build -t egohos-hands .")
    if rc != 0:
        res["reason"] = "docker build failed: " + out[-2000:]
        print(res["reason"]); C.save_json(res, "docker_check.json"); return
    dock_out = os.path.join(C.ROOT, "docker_out")
    shutil.rmtree(dock_out, ignore_errors=True); os.makedirs(dock_out)
    rc, out = sh(f"docker run --rm --gpus all --user {os.getuid()}:{os.getgid()} -v {sample}:/in:ro -v {dock_out}:/out "
                 "egohos-hands --input /in --output /out")
    if rc != 0:
        res["reason"] = "docker run failed: " + out[-2000:]
        print(res["reason"]); C.save_json(res, "docker_check.json"); return
    maxdiff, ndiff = 0, 0
    for f in sorted(os.listdir(host_out)):
        if f.endswith(".png"):
            a = cv2.imread(f"{host_out}/{f}", 0).astype(int); b = cv2.imread(f"{dock_out}/{f}", 0).astype(int)
            d = np.abs(a - b)
            maxdiff = max(maxdiff, int(d.max())); ndiff += int(d.any())
    res = dict(status="DONE", n_images=50, max_pixel_diff=maxdiff, n_images_differing=ndiff)
    print(res); C.save_json(res, "docker_check.json")


if __name__ == "__main__":
    main()
