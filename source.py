import subprocess
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path


OPENFACE_EXE = r"openface/FeatureExtraction.exe"
IMAGES_DIR = Path("images")
OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(exist_ok=True)


def RunOpenFace(image_path: Path):
    subprocess.run(
        [
            OPENFACE_EXE,
            "-f", str(image_path),
            "-out_dir", str(OUTPUT_DIR),
            "-2Dfp"
        ],
        check=True
    )

def LoadLandmarks(csv_path: Path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    x_cols = [c for c in df.columns if c.startswith("x_")]
    y_cols = [c for c in df.columns if c.startswith("y_")]

    if not x_cols or not y_cols:
        raise RuntimeError(f"No landmark columns found in {csv_path}")

    landmark_strength = df[x_cols].sum(axis=1)
    row = df.loc[landmark_strength.idxmax()]

    return row[x_cols].values, row[y_cols].values


def ShowLandmarks(image_path: Path, xs, ys):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.scatter(xs, ys, c="red", s=4)
    plt.title(image_path.name)
    plt.axis("off")
    plt.show()


def main():
    images = sorted(IMAGES_DIR.glob("face*.jpg"))

    if not images:
        raise RuntimeError("No images found in images/")

    for image_path in images:
        print(f"Processing {image_path.name}")

        RunOpenFace(image_path)

        csv_path = OUTPUT_DIR / (image_path.stem + ".csv")

        xs, ys = LoadLandmarks(csv_path)
        ShowLandmarks(image_path, xs, ys)

if __name__ == "__main__":
    main()
