import sys                         

                         
def read_points():                 
    points = []                    
    for line in sys.stdin:         
        stripped = line.strip()    
        if stripped:               
            parts = stripped.split(",")         
            vector = [float(x) for x in parts]  
            points.append(vector)               
    if not points:                 
        sys.exit("No input provided")    
    return points    

def parse_cmdline(argv, n_points):
    if len(argv) not in (2, 3):              
        sys.exit("Usage: python3 kmeans.py K [max_iter]")

    try:
        K = int(argv[1])                      
    except ValueError:
        sys.exit("Incorrect number of clusters!")

    if not (1 < K < n_points):                
        sys.exit("Incorrect number of clusters!")

    if len(argv) == 3:                        
        try:
            max_iter = int(argv[2])
        except ValueError:
            sys.exit("Incorrect maximum iteration!")
        if not (1 < max_iter < 1000):         
            sys.exit("Incorrect maximum iteration!")
    else:
        max_iter = 400                        

    return K, max_iter

def kmeans(points, K, max_iter=100, eps=1e-3):
    """
    Run k-means on `points`
      points   : list of N vectors (list[float])
      K        : number of clusters (int, 1 < K < N)
      max_iter : maximum iterations to try
      eps      : convergence tolerance (centroid shift < eps)

    returns    : list of K centroids (each a vector with same dimension)
    """
    dim = len(points[0])                      # data dimension  (d)
    centroids = [p[:] for p in points[:K]]    # first K points → initial µ_k

    # ------------- repeat assignment + update until convergence -------------
    for _ in range(max_iter):

        # ---------- assignment step ----------
        clusters = [[] for _ in range(K)]     # reset K empty buckets
        for p in points:                      # for every data point…
            best_k   = 0
            best_dist = float("inf")
            for k in range(K):                # scan every centroid
                dist = euclidean(p, centroids[k])
                if dist < best_dist:
                    best_dist = dist
                    best_k    = k
            clusters[best_k].append(p)        # assign to nearest µ_k

        # ---------- update step ----------
        new_centroids = []
        for k in range(K):
            cluster = clusters[k]
            if cluster:                       # non-empty cluster
                # compute coordinate-wise mean
                sums = [0.0] * dim
                for vec in cluster:
                    for i in range(dim):
                        sums[i] += vec[i]
                centroid = [s / len(cluster) for s in sums]
                new_centroids.append(centroid)
            else:                             # empty cluster → keep old µ_k
                new_centroids.append(centroids[k])

        # ---------- convergence test ----------
        largest_shift = max(
            euclidean(old, new) for old, new in zip(centroids, new_centroids)
        )
        if largest_shift < eps:
            break                             # centroids stopped moving
        centroids = new_centroids             # otherwise iterate again

    return centroids

def euclidean(p1, p2):
    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5




def main():
    points = read_points()
    K, max_iter = parse_cmdline(sys.argv, len(points))
    centroids = kmeans(points, K, max_iter)
    for c in centroids:
        print("," .join(f"{x:.4f}" for x in c))
if __name__ == "__main__":
    main()
    


