#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_LINE_LEN 1024

void parse_cmdline(int argc, char *argv[], int n_points, int *K, int *max_iter);
int read_points(double ***points_ptr, int *n_points_ptr, int *dim_ptr);
double euclidean(const double *p1, const double *p2, int dim);
double **kmeans(double **points, int n_points, int dim, int K, int max_iter, double eps);

int main(int argc, char *argv[]) {
    double **points = NULL;
    double **centroids;
    int n_points = 0;
    int dim = 0;
    int K = 0;
    int max_iter = 0;
    int i, j;

    /* Read points from stdin */
    if (read_points(&points, &n_points, &dim) != 0) {
        return 1;
    }

    /* Parse command-line arguments */
    parse_cmdline(argc, argv, n_points, &K, &max_iter);

    /* Run k-means */
    centroids = kmeans(points, n_points, dim, K, max_iter, 1e-3);

    /* Print centroids */
    for (i = 0; i < K; i++) {
        for (j = 0; j < dim; j++) {
            printf("%.4f", centroids[i][j]);
            if (j < dim - 1) {
                printf(",");
            }
        }
        printf("\n");
    }

    /* Free centroids */
    for (i = 0; i < K; i++) {
        free(centroids[i]);
    }
    free(centroids);

    /* Free points */
    for (i = 0; i < n_points; i++) {
        free(points[i]);
    }
    free(points);

    return 0;
}

void parse_cmdline(int argc, char *argv[], int n_points, int *K, int *max_iter) {
    if (argc != 2 && argc != 3) {
        fprintf(stderr, "Usage: ./kmeans K [max_iter]\n");
        exit(1);
    }

    *K = atoi(argv[1]);
    if (*K <= 1 || *K >= n_points) {
        fprintf(stderr, "Incorrect number of clusters!\n");
        exit(1);
    }

    if (argc == 3) {
        *max_iter = atoi(argv[2]);
        if (*max_iter <= 1 || *max_iter >= 1000) {
            fprintf(stderr, "Incorrect maximum iteration!\n");
            exit(1);
        }
    } else {
        *max_iter = 400;
    }
}

double euclidean(const double *p1, const double *p2, int dim) {
    int i;
    double sum = 0.0;
    for (i = 0; i < dim; i++) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

double **kmeans(double **points, int n_points, int dim, int K, int max_iter, double eps) {
    int i, j, k, iter;
    double max_shift;
    double shift;

    double **centroids = malloc(K * sizeof(double *));
    double **new_centroids = malloc(K * sizeof(double *));
    int *cluster_sizes = calloc(K, sizeof(int));

    if (!centroids || !new_centroids || !cluster_sizes) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    for (i = 0; i < K; i++) {
        centroids[i] = malloc(dim * sizeof(double));
        new_centroids[i] = calloc(dim, sizeof(double));
        if (!centroids[i] || !new_centroids[i]) {
            fprintf(stderr, "Memory allocation failed.\n");
            exit(1);
        }
        for (j = 0; j < dim; j++) {
            centroids[i][j] = points[i][j];
        }
    }

    for (iter = 0; iter < max_iter; iter++) {
        for (i = 0; i < K; i++) {
            cluster_sizes[i] = 0;
            for (j = 0; j < dim; j++) {
                new_centroids[i][j] = 0.0;
            }
        }

        for (i = 0; i < n_points; i++) {
            double min_dist = euclidean(points[i], centroids[0], dim);
            int best_k = 0;
            for (k = 1; k < K; k++) {
                double dist = euclidean(points[i], centroids[k], dim);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_k = k;
                }
            }
            cluster_sizes[best_k]++;
            for (j = 0; j < dim; j++) {
                new_centroids[best_k][j] += points[i][j];
            }
        }

        for (k = 0; k < K; k++) {
            if (cluster_sizes[k] > 0) {
                for (j = 0; j < dim; j++) {
                    new_centroids[k][j] /= cluster_sizes[k];
                }
            } else {
                for (j = 0; j < dim; j++) {
                    new_centroids[k][j] = centroids[k][j];
                }
            }
        }

        max_shift = 0.0;
        for (k = 0; k < K; k++) {
            shift = euclidean(centroids[k], new_centroids[k], dim);
            if (shift > max_shift) {
                max_shift = shift;
            }
        }

        if (max_shift < eps) {
            break;
        }

        for (k = 0; k < K; k++) {
            for (j = 0; j < dim; j++) {
                centroids[k][j] = new_centroids[k][j];
            }
        }
    }

    for (i = 0; i < K; i++) {
        free(new_centroids[i]);
    }
    free(new_centroids);
    free(cluster_sizes);

    return centroids;
}

int read_points(double ***points_ptr, int *n_points_ptr, int *dim_ptr) {
    char line[MAX_LINE_LEN];
    int n_points = 0;
    int capacity = 10;
    int dim = 0;
    double **points = (double **)malloc(capacity * sizeof(double *));

    if (points == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    while (fgets(line, MAX_LINE_LEN, stdin) != NULL) {
        int i = 0;
        int current_dim = 0;
        char *token;
        char *line_copy;
        char *tmp;

        line[strcspn(line, "\r\n")] = '\0';
        line_copy = (char *)malloc(strlen(line) + 1);
        if (line_copy == NULL) {
            fprintf(stderr, "Memory allocation failed.\n");
            return 1;
        }
        strcpy(line_copy, line);

        token = strtok(line, ",");
        while (token != NULL) {
            current_dim++;
            token = strtok(NULL, ",");
        }

        if (n_points == 0) {
            dim = current_dim;
        } else if (dim != current_dim) {
            fprintf(stderr, "Inconsistent dimensions at line %d.\n", n_points + 1);
            free(line_copy);
            return 1;
        }

        points[n_points] = (double *)malloc(dim * sizeof(double));
        if (points[n_points] == NULL) {
            fprintf(stderr, "Memory allocation failed.\n");
            free(line_copy);
            return 1;
        }

        token = strtok(line_copy, ",");
        for (i = 0; i < dim && token != NULL; i++) {
            points[n_points][i] = strtod(token, &tmp);
            token = strtok(NULL, ",");
        }

        if (i != dim) {
            fprintf(stderr, "Line %d: expected %d values but got %d.\n", n_points + 1, dim, i);
            free(line_copy);
            return 1;
        }

        free(line_copy);
        n_points++;

        if (n_points >= capacity) {
            double **new_points;
            capacity *= 2;
            new_points = (double **)realloc(points, capacity * sizeof(double *));
            if (new_points == NULL) {
                fprintf(stderr, "Reallocation failed.\n");
                return 1;
            }
            points = new_points;
        }
    }

    if (n_points == 0) {
        fprintf(stderr, "No input provided.\n");
        return 1;
    }

    *points_ptr = points;
    *n_points_ptr = n_points;
    *dim_ptr = dim;

    return 0;
}