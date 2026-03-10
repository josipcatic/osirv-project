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

def LoadPointsForTransform(csv_path: Path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    x_cols = [c for c in df.columns if c.startswith("x_")]
    y_cols = [c for c in df.columns if c.startswith("y_")]

    row = df.iloc[0]

    pts = np.column_stack((row[x_cols].values, row[y_cols].values))
    return pts.astype(np.float32)


def ComputeAffine(A, B):
    M, _ = cv2.estimateAffinePartial2D(B, A)
    return M


def WarpImage(img_src, img_dst, M):
    h, w = img_dst.shape[:2]
    warped = cv2.warpAffine(img_src, M, (w, h))
    return warped

def TransformPoints(points, M):
    pts = points.reshape(-1, 1, 2)
    pts_t = cv2.transform(pts, M)
    return pts_t.reshape(-1, 2)

def ShowAligned(imgA, imgB_aligned, ptsA, ptsB_transformed):

    A = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
    B = cv2.cvtColor(imgB_aligned, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(A)
    plt.scatter(ptsA[:, 0], ptsA[:, 1], c="red", s=4)
    plt.title("Image A")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(B)
    plt.scatter(ptsB_transformed[:, 0], ptsB_transformed[:, 1], c="red", s=4)
    plt.title("Image B aligned")
    plt.axis("off")

    plt.show()



def main():

    imgA_path = Path("images/face3.jpg")
    imgB_path = Path("images/face1.jpg")

    RunOpenFace(imgA_path)
    RunOpenFace(imgB_path)
    
    csvA = OUTPUT_DIR / "face3.csv"
    csvB = OUTPUT_DIR / "face1.csv"

    ptsA = LoadPointsForTransform(csvA)
    ptsB = LoadPointsForTransform(csvB)

    imgA = cv2.imread(str(imgA_path))
    imgB = cv2.imread(str(imgB_path))

    M = ComputeAffine(ptsA, ptsB)

    alignedB = WarpImage(imgB, imgA, M)

    ptsB_t = TransformPoints(ptsB, M)

    ShowAligned(imgA, alignedB, ptsA, ptsB_t)


if __name__ == "__main__":
    main()
