#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <math.h>


double euclidean(const double *p1, const double *p2, int dim) {
    int i;
    double sum = 0.0;
    for (i = 0; i < dim; i++) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

void kmeans(double **points, double **centroids, int n_points, int K, int dim, int max_iter, double eps) {
    int i, j, k, iter;
    double max_shift;
    double shift;

    double **new_centroids = malloc(K * sizeof(double *));
    int *cluster_sizes = calloc(K, sizeof(int));

    if (!new_centroids || !cluster_sizes) {
        printf("An Error Has Occurred\n");
        return;
    }

    for (i = 0; i < K; i++) {
        new_centroids[i] = calloc(dim, sizeof(double));
        if (!new_centroids[i]) {
            printf("An Error Has Occurred\n");
            return;
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
}





static PyObject* fit(PyObject *self, PyObject *args) {
    PyObject *py_points, *py_centroids;
    int n_points, K, dim, max_iter;
    double eps;
    int i, j;
    double **points;
    double **centroids;
    PyObject *row;
    PyObject *result;

    if (!PyArg_ParseTuple(args, "OOiiid", &py_points, &py_centroids, &K, &max_iter, &dim, &eps)) {
        return NULL;
    }

    if (!PyList_Check(py_points) || PyList_Size(py_points) == 0) {
        PyErr_SetString(PyExc_ValueError, "points must be a non-empty list of lists");
        return NULL;
    }

    n_points = PyList_Size(py_points);
    points = malloc(n_points * sizeof(double *));
    centroids = malloc(K * sizeof(double *));
    if (!points || !centroids) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed");
        return NULL;
    }

    for (i = 0; i < n_points; i++) {
        row = PyList_GetItem(py_points, i);
        if (!PyList_Check(row) || PyList_Size(row) != dim) {
            PyErr_SetString(PyExc_ValueError, "All points must have the same dimension");
            return NULL;
        }
        points[i] = malloc(dim * sizeof(double));
        for (j = 0; j < dim; j++) {
            points[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }

    for (i = 0; i < K; i++) {
        row = PyList_GetItem(py_centroids, i);
        if (!PyList_Check(row) || PyList_Size(row) != dim) {
            PyErr_SetString(PyExc_ValueError, "All centroids must have the same dimension");
            return NULL;
        }
        centroids[i] = malloc(dim * sizeof(double));
        for (j = 0; j < dim; j++) {
            centroids[i][j] = PyFloat_AsDouble(PyList_GetItem(row, j));
        }
    }

    kmeans(points, centroids, n_points, K, dim, max_iter, eps);

    result = PyList_New(K);
    for (i = 0; i < K; i++) {
        row = PyList_New(dim);
        for (j = 0; j < dim; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(centroids[i][j]));
        }
        PyList_SetItem(result, i, row);
    }

    for (i = 0; i < n_points; i++) free(points[i]);
    for (i = 0; i < K; i++) free(centroids[i]);
    free(points);
    free(centroids);

    return result;
}

static PyMethodDef methods[] = {
    {"fit", (PyCFunction)fit, METH_VARARGS, "Run K-means clustering"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanspp",  
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_mykmeanspp(void) {
    return PyModule_Create(&moduledef);
}
