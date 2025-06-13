#!/usr/bin/env python3
"""
kmeanspp.py – main interface for HW2


Usage:
    python3 kmeanspp.py  K  [max_iter]  eps  file1  file2

Where:
    K          – number of clusters (int > 1)
    max_iter   – optional, default = 300 (int, 1 < iter < 1000)
    eps        – convergence threshold (float ≥ 0)
    file1      – path to file with N samples (txt/csv)
    file2      – path to second file with matching keys
"""

from __future__ import annotations
import sys
import csv
import math
import numpy as np
from typing import Tuple

ERR_INVALID_K    = "Invalid number of clusters!"
ERR_INVALID_ITER = "Invalid maximum iteration!"
ERR_INVALID_EPS  = "Invalid epsilon!"
ERR_GENERAL      = "An Error Has Occurred Py"

# ---------- helpers ---------------------------------------------------------

def parse_cli(argv: list[str]) -> Tuple[int, int, float, str, str]:
    """Parse & validate command-line arguments."""
    argc = len(argv)
    if argc not in (5, 6):
        print(ERR_GENERAL)
        sys.exit(1)

    try:
        K = int(argv[1])
        if argc == 6:
            max_iter = int(argv[2])
            eps      = float(argv[3])
            file1, file2 = argv[4], argv[5]
        else:
            max_iter = 300
            eps      = float(argv[2])
            file1, file2 = argv[3], argv[4]
    except ValueError:
        print("hoemo")
        print(ERR_GENERAL)
        sys.exit(1)

    if K <= 1:
        print(ERR_INVALID_K); sys.exit(1)
    if not (1 < max_iter < 1000):
        print(ERR_INVALID_ITER); sys.exit(1)
    if eps < 0:
        print(ERR_INVALID_EPS); sys.exit(1)

    return K, max_iter, eps, file1, file2


def read_points(path: str) -> List[List[float]]:
    """Read a txt/csv file whose first column is the key, rest = vector coords."""
    points = []
    try:
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and any(cell.strip() for cell in row):  # skip empty line
                    points.append([float(cell) for cell in row])
    except FileNotFoundError:
        print(f"Error: File not found → {path}")
        exit(1)
    except ValueError:
        print(f"Error: Failed to convert a value to float in file → {path}")
        exit(1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

    return points


def inner_join(a: list[list[float]], b: list[list[float]]) -> np.ndarray:
    """Inner-join on first column and return np.ndarray sorted by key."""
    dict_a = {row[0]: row[1:] for row in a}
    dict_b = {row[0]: row[1:] for row in b}
    common_keys = sorted(set(dict_a) & set(dict_b))
    joined = [dict_a[k] + dict_b[k] for k in common_keys]
    return np.array(joined, dtype=float)


# ---------- K-means++ --------------------------------------------------------

def kmeans_pp_init(points: np.ndarray, K: int) -> list[int]:
    """Return list of indices (in points) chosen by K-means++."""
    np.random.seed(1234)
    N = points.shape[0]
    indices = []

    # choose first centre uniformly
    idx0 = np.random.choice(N)
    indices.append(idx0)

    # choose K-1 additional centres
    for _ in range(1, K):
        dists = np.min(
            np.linalg.norm(points - points[indices][:, None], axis=2) ** 2,
            axis=0
        )
        probs = dists / dists.sum()
        next_idx = np.random.choice(N, p=probs)
        indices.append(next_idx)

    return indices


# ---------- main -------------------------------------------------------------

def main() -> None:
    try:

        K, max_iter, eps, file1, file2 = parse_cli(sys.argv)
        raw1, raw2 = read_points(file1), read_points(file2)
        N = len(raw1)

        if not (K < N == len(raw2)):
            print(ERR_INVALID_K); sys.exit(1)

        data = inner_join(raw1, raw2)     # shape = (N, d)
        dim = data.shape[1]               # number of features per point
        init_indices = kmeans_pp_init(data, K)
        init_centroids = data[init_indices].tolist()

        # ---- call C extension (6 parameters) ----
        import mykmeanspp

        final_centroids = mykmeanspp.fit(
        data.tolist(),        # points: list[list[float]]
        init_centroids,       # centroids: list[list[float]]
        K,                    # int
        max_iter,             # int
        dim,        # dim: int
        eps                   # float
        )

        # ---- output ----
        print(','.join(str(i) for i in init_indices))
        for vec in final_centroids:
            print(','.join(f"{coord:.4f}" for coord in vec))

    except Exception:
        print(ERR_GENERAL)
        sys.exit(1)


if __name__ == "__main__":
    main()