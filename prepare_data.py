#!/usr/bin/env python3

# Builds data.txt
import os
import sys
import glob
import numpy as np
import cv2

from utils import FaceMeshExtractor, landmarks_to_feature

def main():
    if len(sys.argv) < 2:
        print("Usage: python prepare_data.py <dataset_root>")
        sys.exit(1)

    data_dir = sys.argv[1]
    if not os.path.isdir(data_dir):
        print(f"Error: '{data_dir}' is not a directory")
        sys.exit(1)

    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        print("No class folders found.")
        sys.exit(1)

    print("Classes:", class_names)
    np.save('class_names.npy', np.array(class_names, dtype=object))

    fm = FaceMeshExtractor(static_image_mode=True)
    rows = []
    kept_per_class = {c: 0 for c in class_names}
    dropped_per_class = {c: 0 for c in class_names}

    for label_idx, cname in enumerate(class_names):
        folder = os.path.join(data_dir, cname)
        paths = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            paths.extend(glob.glob(os.path.join(folder, ext)))

        if not paths:
            print(f"[WARN] No images found in {folder}")

        for p in paths:
            img = cv2.imread(p)
            if img is None:
                dropped_per_class[cname] += 1
                continue

            lm = fm.get_landmarks(img)
            if lm is None or len(lm) == 0:
                dropped_per_class[cname] += 1
                continue

            feat = landmarks_to_feature(lm)
            if feat is None:
                dropped_per_class[cname] += 1
                continue

            rows.append(np.concatenate([feat, np.array([label_idx])]))
            kept_per_class[cname] += 1

    if not rows:
        print("No usable samples. Aborting.")
        sys.exit(1)

    data = np.vstack(rows).astype(np.float32)
    np.savetxt("data.txt", data, fmt="%.6f")

    print("\n== Summary ==")
    for cname in class_names:
        print(f"{cname:12s} kept={kept_per_class[cname]:4d}  dropped={dropped_per_class[cname]:4d}")
    print(f"Total samples: {len(rows)}")
    print("Wrote data.txt and class_names.npy")

if __name__ == "__main__":
    main()
