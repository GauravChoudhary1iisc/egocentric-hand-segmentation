"""STEP 7 (optional, QUALITATIVE ONLY): blur faces with OpenCV's bundled Haar frontal-face cascade on a few frames.
There is no face ground truth in EgoHOS, so no accuracy is claimed."""
import os
import random

import cv2

import common as C


def main():
    C.seed_all()
    split = C.load_json("split.json")
    det = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    out = os.path.join(C.ROOT, "face_blur_examples"); os.makedirs(out, exist_ok=True)
    items = random.Random(C.SEED).sample(split["test_outdomain"] + split["test_indomain"], 30)
    n_faces = []
    for it in items:
        img = cv2.imread(it["image"])
        faces = det.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        for x, y, w, h in faces:
            img[y:y + h, x:x + w] = cv2.GaussianBlur(img[y:y + h, x:x + w], (0, 0), sigmaX=max(w, h) / 6)
        n_faces.append(len(faces))
        cv2.imwrite(os.path.join(out, it["name"] + ".jpg"), img)
    print(f"face blur (qualitative only, no ground truth): {len(items)} frames, faces detected per frame: {n_faces}; "
          f"frames with >=1 detection: {sum(n > 0 for n in n_faces)}. Outputs in face_blur_examples/ for visual inspection.")
    C.save_json(dict(qualitative_only=True, detector="OpenCV Haar frontalface_default", n_frames=len(items),
                     faces_per_frame=n_faces), "face_blur.json")


if __name__ == "__main__":
    main()
