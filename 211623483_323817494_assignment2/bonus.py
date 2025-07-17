import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Parameters
MAX_K = 10

def compute_inertia(data):
    inertia_values = []
    for k in range(1, MAX_K + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)
    return np.arange(1, MAX_K + 1), np.array(inertia_values)

def detect_elbow(k_vals, inertia_vals):
    start = np.array([k_vals[0], inertia_vals[0]])
    end = np.array([k_vals[-1], inertia_vals[-1]])
    distances = []

    for i in range(len(k_vals)):
        point = np.array([k_vals[i], inertia_vals[i]])
        distance = np.abs(np.cross(end - start, start - point)) / np.linalg.norm(end - start)
        distances.append(distance)

    idx = int(np.argmax(distances))
    return k_vals[idx], inertia_vals[idx]

def draw_elbow_chart(k_vals, inertia_vals, elbow):
    plt.figure(figsize=(8, 5))
    plt.plot(k_vals, inertia_vals, marker='o', linestyle='--', color='purple')
    plt.scatter(*elbow, color='red', s=100, label=f"Elbow at k={elbow[0]}")
    plt.title("Elbow Method for K-Means Clustering")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Sum of Squared Distances)")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.savefig("elbow.png")
    plt.close()

if __name__ == "__main__":
    iris_data = load_iris().data
    k_range, inertias = compute_inertia(iris_data)
    elbow_k = detect_elbow(k_range, inertias)
    draw_elbow_chart(k_range, inertias, elbow_k)
