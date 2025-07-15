#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_LINE_LEN 1024

void free_points(double **points, int n_points);
int parse_cmdline(int argc, char *argv[], int n_points, int *K, int *max_iter);
int read_points(double ***points_ptr, int *n_points_ptr, int *dim_ptr);
double euclidean(const double *p1, const double *p2, int dim);
double **kmeans(double **points, int n_points, int dim, int K, int max_iter, double eps);
int safe_parse_int(const char *str, int *out);

int main(int argc, char *argv[]) {
    double **points = NULL;
    double **centroids = NULL;
    int n_points = 0;
    int dim = 0;
    int K = 0;
    int max_iter = 0;
    int i, j;

    if (read_points(&points, &n_points, &dim) != 0) {
        return 1;
    }

    if (parse_cmdline(argc, argv, n_points, &K, &max_iter) != 0) {
        free_points(points, n_points);
        return 1;
    }

    centroids = kmeans(points, n_points, dim, K, max_iter, 1e-3);
    if (centroids == NULL) {
        printf("An Error Has Occurred\n");
        free_points(points, n_points);
        return 1;
    }

    for (i = 0; i < K; i++) {
        for (j = 0; j < dim; j++) {
            printf("%.4f", centroids[i][j]);
            if (j < dim - 1) {
                printf(",");
            }
        }
        printf("\n");
    }

    free_points(centroids, K);
    free_points(points, n_points);

    return 0;
}

void free_points(double **points, int n_points) {
    int i;
    if (points == NULL) return;
    for (i = 0; i < n_points; i++) {
        free(points[i]);
    }
    free(points);
}

int safe_parse_int(const char *str, int *out) {
    char *endptr;
    double val;

    val = strtod(str, &endptr);

    if (*endptr != '\0') {
        return 0;
    }

    if (floor(val) != val) {
        return 0;
    }

    if (val <= 1.0 || val >= 65536.0) {
        return 0;
    }

    *out = (int)val;
    return 1;
}

int parse_cmdline(int argc, char *argv[], int n_points, int *K, int *max_iter) {
    if (argc != 2 && argc != 3) {
        printf("An Error Has Occurred\n");
        return 1;
    }

    if (!safe_parse_int(argv[1], K) || *K <= 1 || *K >= n_points) {
        printf("Incorrect number of clusters!\n");
        return 1;
    }

    if (argc == 3) {
        if (!safe_parse_int(argv[2], max_iter) || *max_iter <= 1 || *max_iter >= 1000) {
            printf("Incorrect maximum iteration!\n");
            return 1;
        }
    } else {
        *max_iter = 400;
    }
    return 0;
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
        printf("An Error Has Occurred\n");
        return NULL;
    }

    for (i = 0; i < K; i++) {
        centroids[i] = malloc(dim * sizeof(double));
        new_centroids[i] = calloc(dim, sizeof(double));
        if (!centroids[i] || !new_centroids[i]) {
            printf("An Error Has Occurred\n");
            return NULL;
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
    int n_points = 0, capacity = 10, dim = 0;
    double **points = malloc(capacity * sizeof(double *));

    if (!points) {
        printf("An Error Has Occurred\n");
        return 1;
    }

    while (fgets(line, MAX_LINE_LEN, stdin)) {
        char *line_copy = NULL, *token, *tmp;
        int i = 0, current_dim = 0;

        line[strcspn(line, "\r\n")] = '\0';

        line_copy = malloc(strlen(line) + 1);
        if (!line_copy) {
            printf("An Error Has Occurred\n");
            free_points(points, n_points);
            return 1;
        }
        strcpy(line_copy, line);

        token = strtok(line, ",");
        while (token) {
            current_dim++;
            token = strtok(NULL, ",");
        }

        if (n_points == 0)
            dim = current_dim;
        else if (dim != current_dim) {
            printf("An Error Has Occurred\n");
            free(line_copy);
            free_points(points, n_points);
            return 1;
        }

        points[n_points] = malloc(dim * sizeof(double));
        if (!points[n_points]) {
            printf("An Error Has Occurred\n");
            free(line_copy);
            free_points(points, n_points);
            return 1;
        }

        token = strtok(line_copy, ",");
        for (i = 0; i < dim && token; ++i) {
            points[n_points][i] = strtod(token, &tmp);
            token = strtok(NULL, ",");
        }
        if (i != dim) {
            printf("An Error Has Occurred\n");
            free(points[n_points]);
            free(line_copy);
            free_points(points, n_points);
            return 1;
        }

        free(line_copy);
        n_points++;

        if (n_points == capacity) {
            double **tmp_points;
            capacity *= 2;
            tmp_points = realloc(points, capacity * sizeof(double *));
            if (!tmp_points) {
                printf("An Error Has Occurred\n");
                free_points(points, n_points);
                return 1;
            }
            points = tmp_points;
        }
    }

    if (n_points == 0) {
        printf("An Error Has Occurred\n");
        free(points);
        return 1;
    }

    *points_ptr = points;
    *n_points_ptr = n_points;
    *dim_ptr = dim;
return 0;
}
