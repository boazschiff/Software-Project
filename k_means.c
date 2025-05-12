#include <stdio.h>
#include <stdlib.h>

#define MAX_LINE_LEN 1024

void parse_cmdline(int argc, char *argv[], int n_points, int *K, int *max_iter);
int read_points(double ***points_ptr, int *n_points_ptr, int *dim_ptr);
double euclidean(const double *p1, const double *p2, int dim);
double **kmeans(double **points, int n_points, int dim, int K, int max_iter, double eps);

int main(int argc, char *argv[])
{
    double **points = NULL;
    int n_points = 0;
    int dim = 0;

    // Read points from stdin
    if (read_points(&points, &n_points, &dim) != 0)
    {
        return 1;
    }

    int K = 0;
    int max_iter = 0;

    // Parse command-line arguments
    parse_cmdline(argc, argv, n_points, &K, &max_iter);

    // Run k-means
    double **centroids = kmeans(points, n_points, dim, K, max_iter, 1e-3);

    // Print centroids
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            printf("%.4f", centroids[i][j]);
            if (j < dim - 1)
                printf(",");
        }
        printf("\n");
    }

    // Free centroids
    for (int i = 0; i < K; i++)
    {
        free(centroids[i]);
    }
    free(centroids);

    // Free points
    for (int i = 0; i < n_points; i++)
    {
        free(points[i]);
    }
    free(points);

    return 0;
}

int read_points(double ***points_ptr, int *n_points_ptr, int *dim_ptr)
{
    char line[MAX_LINE_LEN];
    int n_points = 0;
    int capacity = 10;
    int dim = 0; /* local working copy of dimension */

    double **points = (double **)malloc(capacity * sizeof(double *));
    if (points == NULL)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    while (fgets(line, MAX_LINE_LEN, stdin) != NULL)
    {
        int i, current_dim = 0;
        char *token;
        char *line_copy;
        char *tmp;

        line[strcspn(line, "\r\n")] = '\0';
        line_copy = (char *)malloc(strlen(line) + 1);
        if (line_copy == NULL)
        {
            fprintf(stderr, "Memory allocation failed.\n");
            return 1;
        }
        strcpy(line_copy, line);

        token = strtok(line, ",");
        while (token != NULL)
        {
            current_dim++;
            token = strtok(NULL, ",");
        }

        if (n_points == 0)
        {
            dim = current_dim;
        }
        else if (dim != current_dim)
        {
            fprintf(stderr, "Inconsistent dimensions at line %d.\n", n_points + 1);
            free(line_copy);
            return 1;
        }
        points[n_points] = (double *)malloc(dim * sizeof(double));
        if (points[n_points] == NULL)
        {
            fprintf(stderr, "Memory allocation failed.\n");
            free(line_copy);
            return 1;
        }
        token = strtok(line_copy, ",");
        for (i = 0; i < dim && token != NULL; i++)
        {
            points[n_points][i] = strtod(token, &tmp);
            token = strtok(NULL, ",");
        }

        if (i != dim)
        {
            fprintf(stderr, "Line %d: expected %d values but got %d.\n", n_points + 1, dim, i);
            free(line_copy);
            return 1;
        }

        free(line_copy);
        n_points++;
        /* Resize points array if needed */
        if (n_points >= capacity)
        {
            double **new_points;
            capacity *= 2;
            new_points = (double **)realloc(points, capacity * sizeof(double *));
            if (new_points == NULL)
            {
                fprintf(stderr, "Reallocation failed.\n");
                return 1;
            }
            points = new_points;
        }
    }

    if (n_points == 0)
    {
        fprintf(stderr, "No input provided.\n");
        return 1;
    }
    *points_ptr = points;
    *n_points_ptr = n_points;
    *dim_ptr = dim;

    return 0;
}

void parse_cmdline(int argc, char *argv[], int n_points, int *K, int *max_iter)
{
    if (argc != 2 && argc != 3)
    {
        fprintf(stderr, "Usage: ./kmeans K [1000]\n");
        exit(1);
    }

    *K = atoi(argv[1]);
    if (*K <= 1 || *K >= n_points)
    {
        fprintf(stderr, "Incorrect number of clusters!\n");
        exit(1);
    }

    if (argc == 3)
    {
        *max_iter = atoi(argv[2]);
        if (*max_iter <= 1 || *max_iter >= 1000)
        {
            fprintf(stderr, "Incorrect maximum iteration!\n");
            exit(1);
        }
    }
    else
    {
        *max_iter = 400; // default value
    }
}

double euclidean(const double *p1, const double *p2, int dim)
{
    double sum = 0.0;
    for (int i = 0; i < dim; i++)
    {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Main K-means function
double **kmeans(double **points, int n_points, int dim, int K, int max_iter, double eps)
{
    // Allocate memory
    double **centroids = malloc(K * sizeof(double *));
    double **new_centroids = malloc(K * sizeof(double *));
    int *cluster_sizes = calloc(K, sizeof(int));

    // Check memory allocations
    if (!centroids || !new_centroids || !cluster_sizes)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    // Initialize centroids with the first K points
    for (int i = 0; i < K; i++)
    {
        centroids[i] = malloc(dim * sizeof(double));
        new_centroids[i] = calloc(dim, sizeof(double));
        if (!centroids[i] || !new_centroids[i])
        {
            fprintf(stderr, "Memory allocation failed.\n");
            exit(1);
        }
        for (int j = 0; j < dim; j++)
        {
            centroids[i][j] = points[i][j];
        }
    }

    // Iterate
    for (int iter = 0; iter < max_iter; iter++)
    {
        // Reset new centroids and cluster sizes
        for (int i = 0; i < K; i++)
        {
            cluster_sizes[i] = 0;
            for (int j = 0; j < dim; j++)
            {
                new_centroids[i][j] = 0.0;
            }
        }

        // Assign each point to the nearest centroid
        for (int i = 0; i < n_points; i++)
        {
            double min_dist = euclidean(points[i], centroids[0], dim);
            int best_k = 0;

            for (int k = 1; k < K; k++)
            {
                double dist = euclidean(points[i], centroids[k], dim);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    best_k = k;
                }
            }

            cluster_sizes[best_k]++;

            for (int j = 0; j < dim; j++)
            {
                new_centroids[best_k][j] += points[i][j];
            }
        }

        // Update centroids by averaging the points in each cluster
        for (int k = 0; k < K; k++)
        {
            if (cluster_sizes[k] > 0)
            {
                for (int j = 0; j < dim; j++)
                {
                    new_centroids[k][j] /= cluster_sizes[k];
                }
            }
            else
            {
                // Keep old centroid if cluster is empty
                for (int j = 0; j < dim; j++)
                {
                    new_centroids[k][j] = centroids[k][j];
                }
            }
        }

        // Check for convergence
        double max_shift = 0.0;
        for (int k = 0; k < K; k++)
        {
            double shift = euclidean(centroids[k], new_centroids[k], dim);
            if (shift > max_shift)
            {
                max_shift = shift;
            }
        }

        if (max_shift < eps)
        {
            break;
        }

        // Copy new_centroids → centroids for next iteration
        for (int k = 0; k < K; k++)
        {
            for (int j = 0; j < dim; j++)
            {
                centroids[k][j] = new_centroids[k][j];
            }
        }
    }

    // Free temp buffers
    for (int i = 0; i < K; i++)
    {
        free(new_centroids[i]);
    }
    free(new_centroids);
    free(cluster_sizes);

    return centroids; // caller must free
}