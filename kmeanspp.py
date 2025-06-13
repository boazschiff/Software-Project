#!/usr/bin/env python3
from __future__ import annotations
import sys
import csv
import numpy as np
import mykmeanspp
from typing import Tuple

np.random.seed(1234)

ERR_INVALID_K    = "Invalid number of clusters!"
ERR_INVALID_ITER = "Invalid maximum iteration!"
ERR_INVALID_EPS  = "Invalid epsilon!"
ERR_GENERAL      = "An Error Has Occurred"


class Point:
    def __init__(self, key: int, coords: list[float]):
        self.key = key
        self.coords = coords
        self.minDist = float('inf')

    def update_min_dist(self, centroid_coords: list[float]) -> float:
        dist = np.linalg.norm(np.array(self.coords) - np.array(centroid_coords))
        self.minDist = min(self.minDist, dist)
        return self.minDist


def parse_cli(argv: list[str]) -> Tuple[int, int, float, str, str]:
    argc = len(argv)
    if argc not in (5, 6):
        print(ERR_GENERAL); sys.exit(1)

    try:
        K = int(argv[1])
        if argc == 6:
            max_iter = int(argv[2])
            eps = float(argv[3])
            file1, file2 = argv[4], argv[5]
        else:
            max_iter = 300
            eps = float(argv[2])
            file1, file2 = argv[3], argv[4]
    except ValueError:
        print(ERR_GENERAL); sys.exit(1)

    if K <= 1:
        print(ERR_INVALID_K); sys.exit(1)
    if not (1 < max_iter < 1000):
        print(ERR_INVALID_ITER); sys.exit(1)
    if eps < 0:
        print(ERR_INVALID_EPS); sys.exit(1)

    return K, max_iter, eps, file1, file2


def read_points(file1: str, file2: str) -> list[Point]:
    def load(path: str) -> dict[int, list[float]]:
        data = {}
        with open(path, 'r') as f:
            for line in f:
                if line.strip() == "":
                    continue
                parts = line.strip().split(',')
                key = int(float(parts[0]))
                coords = list(map(float, parts[1:]))
                data[key] = coords
        return data

    d1 = load(file1)
    d2 = load(file2)
    common_keys = sorted(set(d1) & set(d2))

    return [Point(k, d1[k] + d2[k]) for k in common_keys]


def kmeans_pp_init(points: list[Point], K: int) -> list[int]:
    N = len(points)
    centroids = []
    indices = []

    first = np.random.choice(points)
    centroids.append(first.coords)
    indices.append(first.key)

    for _ in range(1, K):
        # Update min distances
        total = 0
        dists = []
        for p in points:
            d = p.update_min_dist(centroids[-1])
            dists.append(d)
            total += d

        probs = [d / total for d in dists]
        next_point = np.random.choice(points, p=probs)
        centroids.append(next_point.coords)
        indices.append(next_point.key)
    return indices, centroids


def main():
    try:
        K, max_iter, eps, file1, file2 = parse_cli(sys.argv)
        points = read_points(file1, file2)

        if K >= len(points):
            print(ERR_INVALID_K); sys.exit(1)

        indices, init_centroids = kmeans_pp_init(points, K)

        final_centroids = mykmeanspp.fit(
            [p.coords for p in points],
            init_centroids,
            K,
            max_iter,
            len(points[0].coords),
            eps
        )

        print(','.join(str(i) for i in indices))
        for c in final_centroids:
            print(','.join(f"{x:.4f}" for x in c))

    except Exception:
        print(ERR_GENERAL)
        sys.exit(1)


if __name__ == "__main__":
    main()