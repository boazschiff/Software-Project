import sys                         

def read_points():                 
    points = []                    
    for line in sys.stdin:         
        stripped = line.strip()    
        if stripped:               
            parts = stripped.split(",")         
            try:
                vector = [float(x) for x in parts]  
            except ValueError:
                print("Invalid input format")
                sys.exit(1)
            points.append(vector)               
    if not points:                 
        print("No input provided")    
        sys.exit(1)
    return points    

def parse_cmdline(argv, n_points):
    if len(argv) not in (2, 3):              
        print("Usage: python3 kmeans.py K [max_iter]")
        sys.exit(1)

    try:
        K = float(argv[1])
        if not K.is_integer():
            raise ValueError
        K = int(K)
    except ValueError:
        print("Incorrect number of clusters!")
        sys.exit(1)

    if not (1 < K < n_points):                
        print("Incorrect number of clusters!")
        sys.exit(1)

    if len(argv) == 3:                        
        try:
            max_iter = float(argv[2])
            if not max_iter.is_integer():
                raise ValueError
            max_iter = int(max_iter)
        except ValueError:
            print("Incorrect maximum iteration!")
            sys.exit(1)
        if not (1 < max_iter < 1000):         
            print("Incorrect maximum iteration!")
            sys.exit(1)
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
    dim = len(points[0])                      
    centroids = [p[:] for p in points[:K]]    

    for _ in range(max_iter):
        # Assignment step
        clusters = [[] for _ in range(K)]
        for p in points:
            best_k   = 0
            best_dist = float("inf")
            for k in range(K):
                dist = euclidean(p, centroids[k])
                if dist < best_dist:
                    best_dist = dist
                    best_k    = k
            clusters[best_k].append(p)

        new_centroids = []
        for k in range(K):
            cluster = clusters[k]
            if cluster:
                sums = [0.0] * dim
                for vec in cluster:
                    for i in range(dim):
                        sums[i] += vec[i]
                centroid = [s / len(cluster) for s in sums]
                new_centroids.append(centroid)
            else:
                new_centroids.append(centroids[k])

        largest_shift = max(
            euclidean(old, new) for old, new in zip(centroids, new_centroids)
        )
        if largest_shift < eps:
            break
        centroids = new_centroids

    return centroids

def euclidean(p1, p2):
    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5

def main():
    points = read_points()
    K, max_iter = parse_cmdline(sys.argv, len(points))
    centroids = kmeans(points, K, max_iter)
    for c in centroids:
        print(",".join(f"{x:.4f}" for x in c))

if __name__ == "__main__":
    main()
